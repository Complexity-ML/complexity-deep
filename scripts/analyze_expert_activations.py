"""
Expert Activation Analysis — CPU only, no GPU needed.

Collects per-expert activations from a trained model, then analyzes:
1. PCA variance explained per expert (how complex is each expert's manifold?)
2. Intrinsic dimensionality per expert (participation ratio)
3. Expert specialization: cosine similarity between expert mean activations
4. Activation norms per expert per layer

Can run on Colab CPU in minutes.

Usage:
    python scripts/analyze_expert_activations.py --checkpoint checkpoints/step_1000000.pt
    python scripts/analyze_expert_activations.py --checkpoint checkpoints/step_1000000.pt --num-samples 512

INL - 2025
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from collections import defaultdict


def load_checkpoint(checkpoint_path):
    """Load model state dict."""
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


def build_model(state_dict, num_experts=4):
    """Build model from state dict."""
    from complexity_deep.models.config import ComplexityConfig
    from complexity_deep.models.modeling import ComplexityForCausalLM

    q_key = next(k for k in state_dict if "q_proj.weight" in k and "layers.0." in k)
    hidden = state_dict[q_key].shape[0]
    num_layers = max(int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")) + 1
    vocab = state_dict["embed_tokens.weight"].shape[0]
    gu_key = next(k for k in state_dict if "gate_up_proj" in k and "layers.0." in k)
    gu_tensor = state_dict[gu_key]
    detected_experts = gu_tensor.shape[0] if gu_tensor.dim() == 3 else 1
    expert_inter = gu_tensor.shape[-1] // 2
    inter = expert_inter * detected_experts

    k_key = next(k for k in state_dict if "k_proj.weight" in k and "layers.0." in k)
    head_dim = hidden // 16
    num_kv_heads = state_dict[k_key].shape[0] // head_dim

    config = ComplexityConfig(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=16,
        num_key_value_heads=num_kv_heads,
        intermediate_size=inter,
        vocab_size=vocab,
        num_experts=detected_experts,
        max_position_embeddings=2048,
    )

    model = ComplexityForCausalLM(config)
    model.model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


class ActivationCollector:
    """Hook to collect per-expert activations per layer."""

    def __init__(self, num_experts=4):
        self.num_experts = num_experts
        # {layer_idx: {expert_id: [activation_vectors]}}
        self.data = defaultdict(lambda: defaultdict(list))
        self.handles = []

    def register(self, model):
        for layer_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            if hasattr(mlp, "gate_up_proj") and mlp.gate_up_proj.dim() == 3:
                handle = mlp.register_forward_hook(self._make_hook(layer_idx))
                self.handles.append(handle)
        print(f"Hooks on {len(self.handles)} TokenRoutedMLP layers")

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            x = input[0]  # [batch, seq, hidden]
            batch_size, seq_len, hidden = x.shape
            num_experts = module.gate_up_proj.shape[0]

            for expert_id in range(num_experts):
                mask = torch.arange(seq_len, device=x.device) % num_experts == expert_id
                if not mask.any():
                    continue
                tokens = x[:, mask, :]  # [batch, n_tokens, hidden]
                # Store per-token activations (flatten batch)
                flat = tokens.reshape(-1, hidden).detach().cpu().float().numpy()
                self.data[layer_idx][expert_id].append(flat)
        return hook_fn

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_matrices(self):
        """Return {layer: {expert: np.array of shape [n_tokens, hidden]}}."""
        result = {}
        for layer_idx in sorted(self.data.keys()):
            result[layer_idx] = {}
            for expert_id in sorted(self.data[layer_idx].keys()):
                chunks = self.data[layer_idx][expert_id]
                result[layer_idx][expert_id] = np.concatenate(chunks, axis=0)
        return result


@torch.no_grad()
def collect(model, num_samples=256, seq_len=512, vocab_size=32000, device="cpu"):
    """Collect activations on random sequences."""
    collector = ActivationCollector()
    collector.register(model)
    model = model.to(device)

    print(f"Collecting activations ({num_samples} × {seq_len} tokens)...")
    for i in range(num_samples):
        input_ids = torch.randint(3, vocab_size, (1, seq_len), device=device)
        model(input_ids)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{num_samples}")

    collector.remove()
    return collector.get_matrices()


# ── Analysis functions ────────────────────────────────────────────────────

def participation_ratio(eigenvalues):
    """
    Intrinsic dimensionality via participation ratio.
    PR = (sum(lambda_i))^2 / sum(lambda_i^2)
    High PR = activations spread across many dims.
    Low PR = activations concentrated in few dims.
    """
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0
    s = eigenvalues.sum()
    s2 = (eigenvalues ** 2).sum()
    return (s ** 2) / s2 if s2 > 0 else 0.0


def analyze_pca(activations, label="", top_k=10):
    """PCA analysis of activation matrix [n_tokens, hidden]."""
    from sklearn.decomposition import PCA

    n_tokens, hidden = activations.shape
    n_components = min(top_k, n_tokens - 1, hidden)
    if n_components < 2:
        return None

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(activations)

    var_explained = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_explained)
    pr = participation_ratio(pca.explained_variance_)

    return {
        "label": label,
        "n_tokens": n_tokens,
        "var_top1": float(var_explained[0]),
        "var_top3": float(cumvar[min(2, len(cumvar)-1)]),
        "var_top5": float(cumvar[min(4, len(cumvar)-1)]),
        "var_top10": float(cumvar[-1]),
        "participation_ratio": float(pr),
        "mean_norm": float(np.linalg.norm(activations, axis=1).mean()),
        "std_norm": float(np.linalg.norm(activations, axis=1).std()),
    }


def cosine_sim_matrix(means):
    """Cosine similarity between expert mean activations."""
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = means / norms
    return normalized @ normalized.T


def analyze_all(matrices, num_experts=4):
    """Full analysis across layers and experts."""
    from sklearn.decomposition import PCA

    all_results = []
    layer_summaries = []

    for layer_idx in sorted(matrices.keys()):
        print(f"\n{'─'*60}")
        print(f"  Layer {layer_idx}")
        print(f"{'─'*60}")

        layer_means = []
        layer_prs = []
        layer_norms = []

        for expert_id in sorted(matrices[layer_idx].keys()):
            act = matrices[layer_idx][expert_id]
            label = f"L{layer_idx}_E{expert_id}"
            result = analyze_pca(act, label=label)
            if result is None:
                continue
            all_results.append(result)
            layer_prs.append(result["participation_ratio"])
            layer_norms.append(result["mean_norm"])
            layer_means.append(act.mean(axis=0))

            print(f"  Expert {expert_id}: "
                  f"PR={result['participation_ratio']:.1f}, "
                  f"Var1={result['var_top1']:.1%}, "
                  f"Var5={result['var_top5']:.1%}, "
                  f"Norm={result['mean_norm']:.2f} ± {result['std_norm']:.2f}, "
                  f"N={result['n_tokens']}")

        # Cross-expert similarity
        if len(layer_means) >= 2:
            means_matrix = np.stack(layer_means)
            sim = cosine_sim_matrix(means_matrix)
            # Average off-diagonal similarity
            mask = ~np.eye(sim.shape[0], dtype=bool)
            avg_sim = sim[mask].mean()
            print(f"\n  Cross-expert cosine similarity: {avg_sim:.4f}")
            print(f"  (0 = fully specialized, 1 = identical experts)")

            layer_summaries.append({
                "layer": layer_idx,
                "avg_pr": float(np.mean(layer_prs)),
                "avg_norm": float(np.mean(layer_norms)),
                "cross_expert_sim": float(avg_sim),
            })

    return all_results, layer_summaries


def print_summary(all_results, layer_summaries, num_experts):
    """Print final summary."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    # Global stats per expert
    expert_prs = defaultdict(list)
    expert_norms = defaultdict(list)
    for r in all_results:
        eid = int(r["label"].split("_E")[1])
        expert_prs[eid].append(r["participation_ratio"])
        expert_norms[eid].append(r["mean_norm"])

    print(f"\n  Per-expert averages (across layers):")
    print(f"  {'Expert':<10} {'Avg PR':<12} {'Avg Norm':<12}")
    print(f"  {'-'*34}")
    for eid in sorted(expert_prs.keys()):
        print(f"  E{eid:<9} {np.mean(expert_prs[eid]):<12.1f} {np.mean(expert_norms[eid]):<12.2f}")

    # Layer depth trend
    if layer_summaries:
        print(f"\n  Layer depth trend:")
        print(f"  {'Layer':<8} {'Avg PR':<12} {'Cross-Sim':<12} {'Avg Norm':<12}")
        print(f"  {'-'*44}")
        for ls in layer_summaries:
            print(f"  L{ls['layer']:<7} {ls['avg_pr']:<12.1f} {ls['cross_expert_sim']:<12.4f} {ls['avg_norm']:<12.2f}")

    # Key findings
    print(f"\n  KEY FINDINGS:")
    if layer_summaries:
        sims = [ls["cross_expert_sim"] for ls in layer_summaries]
        avg_sim = np.mean(sims)
        if avg_sim < 0.3:
            print(f"  - Experts are HIGHLY SPECIALIZED (avg sim = {avg_sim:.3f})")
        elif avg_sim < 0.7:
            print(f"  - Experts show MODERATE specialization (avg sim = {avg_sim:.3f})")
        else:
            print(f"  - Experts are SIMILAR (avg sim = {avg_sim:.3f}) — weak specialization")

        # PR trend
        prs_early = [ls["avg_pr"] for ls in layer_summaries[:len(layer_summaries)//2]]
        prs_late = [ls["avg_pr"] for ls in layer_summaries[len(layer_summaries)//2:]]
        if prs_early and prs_late:
            if np.mean(prs_late) > np.mean(prs_early) * 1.2:
                print(f"  - Intrinsic dim INCREASES with depth (early={np.mean(prs_early):.1f}, late={np.mean(prs_late):.1f})")
            elif np.mean(prs_late) < np.mean(prs_early) * 0.8:
                print(f"  - Intrinsic dim DECREASES with depth (early={np.mean(prs_early):.1f}, late={np.mean(prs_late):.1f})")
            else:
                print(f"  - Intrinsic dim STABLE across depth (early={np.mean(prs_early):.1f}, late={np.mean(prs_late):.1f})")


def main():
    parser = argparse.ArgumentParser(description="Expert activation analysis (CPU)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="expert_analysis.json")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPERT ACTIVATION ANALYSIS")
    print("=" * 70)

    # Load model
    print(f"\nLoading: {args.checkpoint}")
    state_dict = load_checkpoint(args.checkpoint)
    model, config = build_model(state_dict, num_experts=args.num_experts)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params")

    # Collect activations
    matrices = collect(
        model,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
        device=args.device,
    )

    # Free model memory
    del model
    import gc; gc.collect()

    # Analyze
    all_results, layer_summaries = analyze_all(matrices, num_experts=args.num_experts)

    # Summary
    print_summary(all_results, layer_summaries, args.num_experts)

    # Save
    output = {
        "per_expert_per_layer": all_results,
        "layer_summaries": layer_summaries,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
