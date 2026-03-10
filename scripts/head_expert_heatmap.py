"""
Attention head → Expert distribution heatmap.

For each attention head, measures the mean L2 norm of its output
on tokens routed to each expert. Shows whether certain heads
specialize for certain experts.

Usage:
    python scripts/head_expert_heatmap.py --checkpoint /path/to/model.safetensors --device cuda

Outputs: head_expert_heatmap.png

INL - 2025
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_checkpoint(checkpoint_path):
    path = Path(checkpoint_path)
    if path.suffix == ".pt":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = data.get("model_state_dict", data.get("model", data))
    else:
        from safetensors.torch import load_file
        state_dict = load_file(str(path))
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("model.", "") if k.startswith("model.") else k
        cleaned[key] = v
    return cleaned


def build_model(state_dict):
    sys.path.insert(0, str(Path(__file__).parents[2] / "complexity-framework"))
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    q_key = next(k for k in state_dict if "q_proj.weight" in k and "layers.0." in k)
    hidden = state_dict[q_key].shape[0]
    num_layers = max(int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")) + 1
    vocab = state_dict["embed_tokens.weight"].shape[0]

    gate_key = next((k for k in state_dict if "experts.0.gate_proj.weight" in k and "layers.0." in k), None)
    if gate_key:
        expert_inter = state_dict[gate_key].shape[0]
        num_experts = sum(1 for k in state_dict if "layers.0.mlp.experts." in k and "gate_proj.weight" in k)
        inter = expert_inter * num_experts
        mlp_type = "token_routed"
    else:
        num_experts = 1
        inter_key = next(k for k in state_dict if "gate_proj.weight" in k and "layers.0." in k)
        inter = state_dict[inter_key].shape[0]
        mlp_type = "swiglu"

    k_key = next(k for k in state_dict if "k_proj.weight" in k and "layers.0." in k)
    num_kv_heads = state_dict[k_key].shape[0] // (hidden // 12)

    config = ModelConfig(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=12,
        num_key_value_heads=num_kv_heads,
        intermediate_size=inter,
        vocab_size=vocab,
        num_experts=num_experts,
        max_position_embeddings=2048,
        mlp_type=mlp_type,
        use_inl_dynamics=True,
        use_qk_norm=True,
    )
    model = ComplexityModel(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


class HeadExpertCollector:
    """
    Hook attention output per head + MLP input per expert.

    For each (layer, head, expert): records mean norm of head output
    on tokens that will be routed to that expert.
    """

    def __init__(self, num_heads, num_experts):
        self.num_heads = num_heads
        self.num_experts = num_experts
        # accumulator[layer][head][expert] = (sum_norm, count)
        self.data = {}
        self.handles = []
        self._attn_outputs = {}  # layer_idx -> attn output tensor

    def register(self, model):
        layers = model.layers if hasattr(model, "layers") else model.model.layers
        for layer_idx, layer in enumerate(layers):
            # Hook attention output
            handle = layer.self_attn.register_forward_hook(self._make_attn_hook(layer_idx))
            self.handles.append(handle)
            # Hook MLP input to know expert assignment
            mlp = layer.mlp
            is_routed = (
                (hasattr(mlp, "gate_up_proj") and mlp.gate_up_proj.dim() == 3)
                or hasattr(mlp, "experts")
            )
            if is_routed:
                handle = mlp.register_forward_hook(self._make_mlp_hook(layer_idx))
                self.handles.append(handle)
        print(f"Registered hooks on {len(layers)} layers")

    def _make_attn_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is the attention result [batch, seq, hidden]
            # Reshape to per-head: [batch, seq, num_heads, head_dim]
            out = output[0] if isinstance(output, tuple) else output
            batch, seq, hidden = out.shape
            head_dim = hidden // self.num_heads
            # Store for MLP hook to use
            self._attn_outputs[layer_idx] = out.detach()  # [B, S, H]
        return hook_fn

    def _make_mlp_hook(self, layer_idx):
        def hook_fn(module, input, output):
            x = input[0]  # [batch, seq, hidden]
            batch, seq, hidden = x.shape
            head_dim = hidden // self.num_heads

            if hasattr(module, "gate_up_proj") and module.gate_up_proj.dim() == 3:
                num_experts = module.gate_up_proj.shape[0]
            elif hasattr(module, "experts"):
                num_experts = len(module.experts)
            else:
                return

            attn_out = self._attn_outputs.get(layer_idx)
            if attn_out is None:
                return

            if layer_idx not in self.data:
                self.data[layer_idx] = np.zeros((self.num_heads, num_experts))
            counts = np.zeros((self.num_heads, num_experts))

            # Reshape attn_out to [B, S, num_heads, head_dim]
            attn_per_head = attn_out.view(batch, seq, self.num_heads, head_dim)

            for expert_id in range(num_experts):
                mask = torch.arange(seq, device=x.device) % num_experts == expert_id
                if not mask.any():
                    continue
                # attn output for tokens going to this expert: [B, n_tok, num_heads, head_dim]
                tokens_attn = attn_per_head[:, mask, :, :]  # [B, n_tok, H, D]
                # per-head norm: [B, n_tok, H]
                norms = tokens_attn.norm(dim=-1)
                # mean over batch and tokens
                mean_norm = norms.mean(dim=(0, 1)).cpu().float().numpy()  # [H]
                self.data[layer_idx][:, expert_id] += mean_norm
                counts[:, expert_id] += 1

            # Normalize
            counts = np.maximum(counts, 1)
            self.data[layer_idx] /= counts

        return hook_fn

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def collect(model, config, num_samples=64, seq_len=256, device="cuda"):
    collector = HeadExpertCollector(
        num_heads=config.num_attention_heads,
        num_experts=config.num_experts,
    )
    collector.register(model)
    model = model.to(device)

    print(f"Collecting ({num_samples} samples x {seq_len} tokens)...")
    for i in range(num_samples):
        input_ids = torch.randint(3, config.vocab_size, (1, seq_len), device=device)
        model(input_ids)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{num_samples}")

    collector.remove()
    return collector.data


def plot_heatmap(data, num_heads, num_experts, num_layers, output_path):
    """
    Two plots:
    (a) Per-layer heatmaps (head x expert norm)
    (b) Aggregated across all layers
    """
    colors_expert = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    # Aggregate: mean across layers
    all_layers = [data[l] for l in sorted(data.keys())]
    agg = np.mean(all_layers, axis=0)  # [num_heads, num_experts]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Aggregated heatmap
    ax = axes[0]
    im = ax.imshow(agg, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean head output norm")
    ax.set_xlabel("Expert", fontsize=12)
    ax.set_ylabel("Attention Head", fontsize=12)
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f"Expert {e}" for e in range(num_experts)])
    ax.set_yticks(range(num_heads))
    ax.set_yticklabels([f"Head {h}" for h in range(num_heads)])
    ax.set_title("(a) Head → Expert affinity (all layers)", fontsize=13, fontweight="bold")

    # Annotate values
    for h in range(num_heads):
        for e in range(num_experts):
            ax.text(e, h, f"{agg[h, e]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    # (b) Per-head: which expert does each head prefer?
    ax = axes[1]
    preferred_expert = agg.argmax(axis=1)  # [num_heads]
    head_max_norm = agg.max(axis=1)        # [num_heads]
    head_min_norm = agg.min(axis=1)

    bar_colors = [colors_expert[e % len(colors_expert)] for e in preferred_expert]
    bars = ax.barh(range(num_heads), head_max_norm - head_min_norm,
                   left=head_min_norm, color=bar_colors, alpha=0.8)

    ax.set_yticks(range(num_heads))
    ax.set_yticklabels([f"Head {h}" for h in range(num_heads)])
    ax.set_xlabel("Norm range (min → max across experts)", fontsize=11)
    ax.set_title("(b) Per-head expert preference\n(color = preferred expert)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_expert[e], label=f"Expert {e}")
                       for e in range(num_experts)]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.suptitle("Attention Head → Expert Distribution", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Attention head → expert distribution heatmap")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="head_expert_heatmap.png")
    args = parser.parse_args()

    print(f"Loading: {args.checkpoint}")
    state_dict = load_checkpoint(args.checkpoint)
    model, config = build_model(state_dict)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{config.num_hidden_layers} layers, {config.num_attention_heads} heads, "
          f"{config.num_experts} experts")

    data = collect(model, config,
                   num_samples=args.num_samples,
                   seq_len=args.seq_len,
                   device=args.device)

    num_layers = max(data.keys()) + 1 if data else 0
    print(f"\nCollected data from {num_layers} layers")

    plot_heatmap(data, config.num_attention_heads, config.num_experts,
                 num_layers, args.output)


if __name__ == "__main__":
    main()
