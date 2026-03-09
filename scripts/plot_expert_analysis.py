"""
Visualize expert activation analysis from JSON output.

Reads expert_analysis.json and generates publication-ready plots:
1. Barplot: Participation Ratio per expert (averaged across layers)
2. Heatmap: Cross-expert cosine similarity per layer
3. Line plot: PR and norms evolution across layer depth
4. Stacked variance: PCA variance explained (top 1/3/5/10) per expert

Usage:
    python scripts/plot_expert_analysis.py
    python scripts/plot_expert_analysis.py --input expert_analysis.json --output plots/

INL - 2025
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

EXPERT_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data["per_expert_per_layer"], data["layer_summaries"]


def parse_label(label):
    """Parse 'L0_E2' -> (layer=0, expert=2)."""
    parts = label.split("_")
    layer = int(parts[0][1:])
    expert = int(parts[1][1:])
    return layer, expert


# ── Plot 1: PR barplot per expert ─────────────────────────────────────────

def plot_pr_barplot(results, output_dir):
    """Barplot of average participation ratio per expert."""
    expert_prs = defaultdict(list)
    for r in results:
        _, eid = parse_label(r["label"])
        expert_prs[eid].append(r["participation_ratio"])

    experts = sorted(expert_prs.keys())
    means = [np.mean(expert_prs[e]) for e in experts]
    stds = [np.std(expert_prs[e]) for e in experts]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [f"Expert {e}" for e in experts], means, yerr=stds,
        color=EXPERT_COLORS[:len(experts)], edgecolor="white",
        linewidth=1.5, capsize=5, alpha=0.9,
    )

    ax.set_ylabel("Participation Ratio (intrinsic dim)")
    ax.set_title("Expert Intrinsic Dimensionality")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontweight="bold")

    path = output_dir / "expert_pr_barplot.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 2: Similarity heatmap per layer ──────────────────────────────────

def plot_similarity_heatmap(results, output_dir):
    """Heatmap of cross-expert cosine similarity evolving by layer."""
    # Group by layer
    layer_experts = defaultdict(dict)
    for r in results:
        lid, eid = parse_label(r["label"])
        layer_experts[lid][eid] = r["mean_norm"]

    layers = sorted(layer_experts.keys())
    num_experts = max(max(layer_experts[l].keys()) for l in layers) + 1

    # Build mean activation norms matrix [layers × experts]
    matrix = np.zeros((len(layers), num_experts))
    for i, l in enumerate(layers):
        for e in range(num_experts):
            matrix[i, e] = layer_experts[l].get(e, 0)

    # Normalize per layer for similarity view
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = matrix / norms

    # Cosine sim between experts across layers
    sim_matrix = np.zeros((num_experts, num_experts))
    for i in range(num_experts):
        for j in range(num_experts):
            sim_matrix[i, j] = np.corrcoef(matrix[:, i], matrix[:, j])[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(num_experts))
    ax.set_xticklabels([f"E{i}" for i in range(num_experts)])
    ax.set_yticklabels([f"E{i}" for i in range(num_experts)])
    ax.set_title("Expert Norm Correlation Across Layers")

    # Add values
    for i in range(num_experts):
        for j in range(num_experts):
            ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(sim_matrix[i,j]) > 0.5 else "black", fontsize=11)

    path = output_dir / "expert_similarity_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 3: Depth evolution ───────────────────────────────────────────────

def plot_depth_evolution(results, layer_summaries, output_dir):
    """Line plot: PR, similarity, and norms vs layer depth."""
    if not layer_summaries:
        print("No layer summaries, skipping depth plot")
        return

    layers = [ls["layer"] for ls in layer_summaries]
    prs = [ls["avg_pr"] for ls in layer_summaries]
    sims = [ls["cross_expert_sim"] for ls in layer_summaries]
    norms = [ls["avg_norm"] for ls in layer_summaries]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PR vs depth
    ax = axes[0]
    ax.plot(layers, prs, "o-", color="#e74c3c", linewidth=2, markersize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Intrinsic Dimensionality vs Depth")
    ax.grid(alpha=0.3)

    # Similarity vs depth
    ax = axes[1]
    ax.plot(layers, sims, "s-", color="#3498db", linewidth=2, markersize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-Expert Cosine Sim")
    ax.set_title("Expert Specialization vs Depth")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Norms vs depth
    ax = axes[2]
    ax.plot(layers, norms, "^-", color="#2ecc71", linewidth=2, markersize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Activation Norm")
    ax.set_title("Activation Magnitude vs Depth")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "expert_depth_evolution.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 4: PCA variance per expert ───────────────────────────────────────

def plot_pca_variance(results, output_dir):
    """Grouped barplot of PCA variance explained per expert."""
    expert_vars = defaultdict(lambda: {"v1": [], "v3": [], "v5": [], "v10": []})
    for r in results:
        _, eid = parse_label(r["label"])
        expert_vars[eid]["v1"].append(r["var_top1"])
        expert_vars[eid]["v3"].append(r["var_top3"])
        expert_vars[eid]["v5"].append(r["var_top5"])
        expert_vars[eid]["v10"].append(r["var_top10"])

    experts = sorted(expert_vars.keys())
    x = np.arange(len(experts))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = [
        ("Top 1", "v1", "#e74c3c"),
        ("Top 3", "v3", "#f39c12"),
        ("Top 5", "v5", "#3498db"),
        ("Top 10", "v10", "#2ecc71"),
    ]

    for i, (label, key, color) in enumerate(categories):
        means = [np.mean(expert_vars[e][key]) for e in experts]
        ax.bar(x + i * width, means, width, label=label, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.8)

    ax.set_xlabel("Expert")
    ax.set_ylabel("Variance Explained")
    ax.set_title("PCA Variance Explained per Expert")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"Expert {e}" for e in experts])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = output_dir / "expert_pca_variance.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 5: Per-layer per-expert PR heatmap ───────────────────────────────

def plot_pr_heatmap(results, output_dir):
    """Heatmap of participation ratio: layers × experts."""
    layer_expert_pr = defaultdict(dict)
    for r in results:
        lid, eid = parse_label(r["label"])
        layer_expert_pr[lid][eid] = r["participation_ratio"]

    layers = sorted(layer_expert_pr.keys())
    experts = sorted(set(e for l in layers for e in layer_expert_pr[l].keys()))

    matrix = np.zeros((len(layers), len(experts)))
    for i, l in enumerate(layers):
        for j, e in enumerate(experts):
            matrix[i, j] = layer_expert_pr[l].get(e, 0)

    fig, ax = plt.subplots(figsize=(8, max(6, len(layers) * 0.4)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Participation Ratio")

    ax.set_xticks(range(len(experts)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels([f"E{e}" for e in experts])
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")
    ax.set_title("Intrinsic Dimensionality (Layer × Expert)")

    path = output_dir / "expert_pr_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot expert analysis results")
    parser.add_argument("--input", type=str, default="expert_analysis.json")
    parser.add_argument("--output", type=str, default="./plots")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    results, layer_summaries = load_data(args.input)
    print(f"  {len(results)} expert×layer entries")
    print(f"  {len(layer_summaries)} layer summaries")

    print(f"\nGenerating plots in {output_dir}/\n")

    plot_pr_barplot(results, output_dir)
    plot_similarity_heatmap(results, output_dir)
    plot_depth_evolution(results, layer_summaries, output_dir)
    plot_pca_variance(results, output_dir)
    plot_pr_heatmap(results, output_dir)

    print(f"\nDone! {5} plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
