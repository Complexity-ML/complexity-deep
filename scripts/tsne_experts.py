"""
T-SNE visualization of expert activations.

Loads model on GPU, hooks TokenRoutedMLP layers to capture per-expert
hidden states during forward pass, then runs PCA + T-SNE.

Shows that despite modulo-based routing (token_id % num_experts),
each expert learns distinct representations in activation space.

Usage:
    python scripts/tsne_experts.py --checkpoint checkpoints/step_1000000.pt
    python scripts/tsne_experts.py --checkpoint checkpoints/step_1000000.pt --tokenizer ./tokenizer --num-samples 512

Outputs: expert_tsne.png

INL - 2025
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from safetensors.torch import load_file
from pathlib import Path


def load_checkpoint(checkpoint_path, device="cuda"):
    """Load model state dict from checkpoint."""
    path = Path(checkpoint_path)

    if path.suffix == ".pt":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = data.get("model_state_dict", data.get("model", data))
        config = data.get("config", None)
    else:
        state_dict = load_file(str(path))
        config = None

    # Strip "model." prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("model.", "") if k.startswith("model.") else k
        cleaned[key] = v

    return cleaned, config


def build_model(state_dict, config_dict=None, num_experts=4, vocab_size=100000):
    """Build model from state dict using complexity-deep."""
    from complexity_deep.models.config import ComplexityConfig
    from complexity_deep.models.modeling import ComplexityForCausalLM

    if config_dict is not None:
        config = ComplexityConfig(**config_dict)
    else:
        # Infer from state dict
        hidden = state_dict["layers.0.attention.q_proj.weight"].shape[0]
        num_layers = max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")
        ) + 1
        vocab = state_dict["embed_tokens.weight"].shape[0]
        inter = state_dict["layers.0.mlp.gate_up_proj"].shape[-1] // 2

        config = ComplexityConfig(
            hidden_size=hidden,
            num_hidden_layers=num_layers,
            num_attention_heads=16,
            num_key_value_heads=8,
            intermediate_size=inter,
            vocab_size=vocab,
            num_experts=num_experts,
            max_position_embeddings=2048,
        )

    model = ComplexityForCausalLM(config)
    model.model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class ExpertActivationHook:
    """Hook TokenRoutedMLP layers to capture per-expert activations."""

    def __init__(self, num_experts=4):
        self.num_experts = num_experts
        self.activations = []  # list of (expert_id, layer_idx, activation_vector)
        self.handles = []

    def register(self, model):
        """Register forward hooks on all TokenRoutedMLP layers."""
        for layer_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            # Check if it's a token-routed MLP (has gate_up_proj with 3 dims)
            if hasattr(mlp, "gate_up_proj") and mlp.gate_up_proj.dim() == 3:
                handle = mlp.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self.handles.append(handle)
        print(f"Registered hooks on {len(self.handles)} TokenRoutedMLP layers")

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output shape: [batch, seq_len, hidden]
            # input[0] shape: [batch, seq_len, hidden]
            x = input[0]  # [batch, seq_len, hidden]
            batch_size, seq_len, hidden = x.shape

            # Route tokens to experts: token_id % num_experts
            # We use position index as proxy for token routing
            gate_up = module.gate_up_proj  # [num_experts, hidden, 2*inter]
            down = module.down_proj        # [num_experts, inter, hidden]
            num_experts = gate_up.shape[0]

            # Compute per-expert outputs for each token
            for expert_id in range(num_experts):
                # Get tokens routed to this expert
                mask = torch.arange(seq_len, device=x.device) % num_experts == expert_id
                if not mask.any():
                    continue

                tokens = x[:, mask, :]  # [batch, num_tokens, hidden]

                # Expert forward: gate_up then down
                gu = torch.matmul(tokens, gate_up[expert_id])  # [batch, num_tokens, 2*inter]
                inter_size = gu.shape[-1] // 2
                gate = gu[..., :inter_size]
                up = gu[..., inter_size:]
                activated = torch.nn.functional.silu(gate) * up  # [batch, num_tokens, inter]
                expert_out = torch.matmul(activated, down[expert_id])  # [batch, num_tokens, hidden]

                # Store mean activation per expert per batch item
                for b in range(batch_size):
                    act = expert_out[b].mean(dim=0).detach().cpu().float().numpy()
                    self.activations.append((expert_id, layer_idx, act))

        return hook_fn

    def clear(self):
        self.activations = []

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def collect_activations(model, tokenizer, num_samples=256, max_length=512, device="cuda"):
    """Run text through model and collect expert activations via hooks."""
    hook = ExpertActivationHook(num_experts=4)
    hook.register(model)
    model = model.to(device)

    # Generate random token sequences (we just need activations, not quality)
    print(f"Collecting activations from {num_samples} samples...")
    for i in range(num_samples):
        input_ids = torch.randint(
            3, tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer),
            (1, max_length), device=device
        )
        model(input_ids)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{num_samples} samples ({len(hook.activations)} activations)")

    hook.remove()
    print(f"Total activations collected: {len(hook.activations)}")
    return hook.activations


def plot_tsne(X_2d, expert_ids, layer_ids, num_experts, output_path):
    """Create 2-panel T-SNE plot."""
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by expert
    ax = axes[0]
    for eid in range(num_experts):
        mask = expert_ids == eid
        if not mask.any():
            continue
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[eid % len(colors)], label=f"Expert {eid}",
                   alpha=0.7, s=40, edgecolors="white", linewidth=0.3)
    ax.set_title("Expert Activation T-SNE", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")
    ax.grid(True, alpha=0.2)

    # Right: colored by layer
    ax = axes[1]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=layer_ids, cmap="viridis",
                         alpha=0.7, s=40, edgecolors="white", linewidth=0.3)
    plt.colorbar(scatter, ax=ax, label="Layer index")
    ax.set_title("Layer Depth T-SNE", fontsize=14, fontweight="bold")
    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="T-SNE of expert activations (GPU)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Number of random sequences to feed through model")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output", type=str, default="expert_tsne.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load tokenizer
    from transformers import PreTrainedTokenizerFast
    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    print(f"Tokenizer: {len(tokenizer)} tokens")

    # Load checkpoint and build model
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict, config_dict = load_checkpoint(args.checkpoint)
    model = build_model(state_dict, config_dict, num_experts=args.num_experts,
                        vocab_size=len(tokenizer))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params")

    # Collect activations via hooks
    activations = collect_activations(
        model, tokenizer,
        num_samples=args.num_samples,
        max_length=args.max_length,
        device=args.device,
    )

    if not activations:
        print("No activations collected!")
        return

    # Unpack
    expert_ids = np.array([a[0] for a in activations])
    layer_ids = np.array([a[1] for a in activations])
    vectors = np.array([a[2] for a in activations])

    print(f"\nActivation matrix: {vectors.shape}")
    for eid in range(args.num_experts):
        count = (expert_ids == eid).sum()
        print(f"  Expert {eid}: {count} activations")

    # PCA -> T-SNE
    pca_dim = min(50, vectors.shape[0] - 1, vectors.shape[1])
    print(f"\nPCA: {vectors.shape[1]} -> {pca_dim} dims")
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(vectors)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    perplexity = min(args.perplexity, X_pca.shape[0] - 1)
    print(f"T-SNE on {X_pca.shape[0]} vectors (perplexity={perplexity:.0f})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_pca)

    # Plot
    plot_tsne(X_2d, expert_ids, layer_ids, args.num_experts, args.output)


if __name__ == "__main__":
    main()
