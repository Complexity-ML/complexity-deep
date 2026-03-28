"""
Expert Specialization Analysis for Token-Routed MLP.

Measures whether experts in the Token-Routed MLP develop distinct
representations despite deterministic routing.  This addresses the
reviewer question: "If routing is arbitrary w.r.t. semantic content,
how is this different from simply splitting an MLP's weights?"

Metrics computed per layer:
1. Cosine similarity between all expert-pair weight matrices
2. Normalized Euclidean distance between expert weights
3. Per-expert weight norms

If experts specialize, cosine similarity should decrease and Euclidean
distance should increase over training.

Usage:
    python analyze_expert_specialization.py --checkpoint ./checkpoints/final.pt
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations


def load_state_dict(path: str) -> dict:
    """Load checkpoint state dict from various formats."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def extract_expert_weights(state_dict: dict, num_experts: int = 4) -> dict:
    """
    Find expert weight tensors (3-D: [E, in, out]) per layer.

    Returns: {layer_idx: {"gate": Tensor, "up": Tensor, "down": Tensor}}
    """
    layers = {}
    for key, tensor in state_dict.items():
        if tensor.dim() != 3 or tensor.shape[0] != num_experts:
            continue
        if "mlp" not in key:
            continue

        # Extract layer index from key like "layers.5.mlp.gate_proj_w"
        parts = key.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    break
                except ValueError:
                    pass
        if layer_idx is None:
            continue

        if layer_idx not in layers:
            layers[layer_idx] = {}

        if "gate" in key:
            layers[layer_idx]["gate"] = tensor
        elif "up" in key:
            layers[layer_idx]["up"] = tensor
        elif "down" in key:
            layers[layer_idx]["down"] = tensor

    return layers


def cosine_sim(w1: torch.Tensor, w2: torch.Tensor) -> float:
    """Cosine similarity between two flattened weight tensors."""
    a = w1.flatten().float()
    b = w2.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


def euclidean_dist(w1: torch.Tensor, w2: torch.Tensor) -> float:
    """Normalized Euclidean distance."""
    a = w1.flatten().float()
    b = w2.flatten().float()
    avg_norm = (a.norm() + b.norm()) / 2
    return ((a - b).norm() / (avg_norm + 1e-8)).item()


def analyze(checkpoint_path: str, num_experts: int = 4):
    """Run the full analysis and return results dict."""
    sd = load_state_dict(checkpoint_path)
    expert_weights = extract_expert_weights(sd, num_experts)

    if not expert_weights:
        print("No expert weights found in checkpoint.")
        return None

    num_layers = max(expert_weights.keys()) + 1
    print(f"Found expert weights in {len(expert_weights)} layers (max index {num_layers - 1})")

    pairs = list(combinations(range(num_experts), 2))
    cos_per_layer = []
    euc_per_layer = []
    norms_per_layer = {e: [] for e in range(num_experts)}

    for layer_idx in sorted(expert_weights.keys()):
        w = expert_weights[layer_idx]
        # Use gate weights as representative (largest tensor)
        t = w.get("gate", w.get("up", w.get("down")))
        if t is None:
            continue

        layer_cos = []
        layer_euc = []
        for i, j in pairs:
            layer_cos.append(cosine_sim(t[i], t[j]))
            layer_euc.append(euclidean_dist(t[i], t[j]))

        cos_per_layer.append(np.mean(layer_cos))
        euc_per_layer.append(np.mean(layer_euc))

        for e in range(num_experts):
            norms_per_layer[e].append(t[e].float().norm().item())

    return {
        "layers": list(range(len(cos_per_layer))),
        "cos_sim": cos_per_layer,
        "euclidean": euc_per_layer,
        "norms": norms_per_layer,
        "num_experts": num_experts,
    }


def plot_results(results: dict, output_path: str):
    """Create a 2x2 summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    L = results["layers"]
    E = results["num_experts"]

    # 1. Mean cosine similarity
    ax = axes[0, 0]
    ax.plot(L, results["cos_sim"], "b-", linewidth=2)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="identical")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Expert similarity (lower = more specialized)")
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Mean Euclidean distance
    ax = axes[0, 1]
    ax.plot(L, results["euclidean"], "g-", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Euclidean distance")
    ax.set_title("Expert distance (higher = more specialized)")
    ax.grid(True, alpha=0.3)

    # 3. Per-expert norms
    ax = axes[1, 0]
    for e in range(E):
        ax.plot(L, results["norms"][e], label=f"Expert {e}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Weight norm")
    ax.set_title("Per-expert weight norms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis("off")
    mean_cos = np.mean(results["cos_sim"])
    mean_euc = np.mean(results["euclidean"])
    if mean_cos > 0.98:
        verdict = "No specialization (experts nearly identical)"
    elif mean_cos > 0.90:
        verdict = "Low specialization"
    elif mean_cos > 0.80:
        verdict = "Moderate specialization"
    else:
        verdict = "Strong specialization"

    summary = (
        f"Mean cosine similarity: {mean_cos:.4f}\n"
        f"Mean Euclidean distance: {mean_euc:.4f}\n\n"
        f"Verdict: {verdict}\n\n"
        f"Early layers (0-2): cos_sim = {np.mean(results['cos_sim'][:3]):.4f}\n"
        f"Late layers (-3:):  cos_sim = {np.mean(results['cos_sim'][-3:]):.4f}"
    )
    ax.text(0.1, 0.5, summary, fontsize=12, family="monospace",
            verticalalignment="center", transform=ax.transAxes)

    plt.suptitle("Token-Routed MLP: Expert Specialization Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze expert specialization")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final.pt")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--output", type=str, default="./expert_specialization.png")
    args = parser.parse_args()

    results = analyze(args.checkpoint, args.num_experts)
    if results is not None:
        plot_results(results, args.output)

        mean_cos = np.mean(results["cos_sim"])
        print(f"\nMean cosine similarity: {mean_cos:.4f}")
        if mean_cos < 0.90:
            print("Experts are well differentiated.")
        elif mean_cos < 0.98:
            print("Experts show some differentiation.")
        else:
            print("Experts are nearly identical (no specialization).")


if __name__ == "__main__":
    main()
