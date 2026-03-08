"""
T-SNE visualization of expert representations.

Shows that despite modulo-based routing (token_id % num_experts),
each expert learns distinct representations.

Usage:
    python scripts/tsne_experts.py --checkpoint checkpoints/step_200000.safetensors
    python scripts/tsne_experts.py --checkpoint checkpoints/pacific-prime-chat/checkpoint_epoch400.pt

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


def load_expert_weights(checkpoint_path):
    """Load expert weights from checkpoint."""
    path = Path(checkpoint_path)

    if path.suffix == ".pt":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = data.get("model_state_dict", data.get("model", data))
    else:
        state_dict = load_file(str(path))

    # Strip "model." prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("model.", "") if k.startswith("model.") else k
        cleaned[key] = v

    return cleaned


def extract_expert_representations(state_dict, num_experts=4):
    """Extract per-expert weight vectors from all layers."""
    expert_vectors = {i: [] for i in range(num_experts)}
    layer_labels = {i: [] for i in range(num_experts)}

    for key, tensor in state_dict.items():
        # Match expert weights: layers.X.mlp.experts.Y.{gate,up,down}_proj.weight
        if ".mlp.experts." in key and ".weight" in key:
            parts = key.split(".")
            try:
                expert_idx = int(parts[parts.index("experts") + 1])
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue

            if expert_idx < num_experts:
                flat = tensor.flatten().float().numpy()
                expert_vectors[expert_idx].append(flat)
                layer_labels[expert_idx].append(layer_idx)

        # Match fused expert weights: layers.X.mlp.gate_up_proj [num_experts, hidden, 2*inter]
        elif ".mlp.gate_up_proj" in key and tensor.dim() == 3:
            parts = key.split(".")
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue

            for expert_idx in range(min(tensor.shape[0], num_experts)):
                flat = tensor[expert_idx].flatten().float().numpy()
                expert_vectors[expert_idx].append(flat)
                layer_labels[expert_idx].append(layer_idx)

        elif ".mlp.down_proj" in key and tensor.dim() == 3:
            parts = key.split(".")
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue

            for expert_idx in range(min(tensor.shape[0], num_experts)):
                flat = tensor[expert_idx].flatten().float().numpy()
                expert_vectors[expert_idx].append(flat)
                layer_labels[expert_idx].append(layer_idx)

    return expert_vectors, layer_labels


def truncate_to_min_length(vectors_dict):
    """Truncate all vectors to same length for T-SNE."""
    min_len = min(
        min(len(v) for v in vecs) if vecs else float('inf')
        for vecs in vectors_dict.values()
        if vecs
    )
    if min_len == float('inf'):
        return vectors_dict

    truncated = {}
    for expert_id, vecs in vectors_dict.items():
        truncated[expert_id] = [v[:min_len] for v in vecs]
    return truncated


def main():
    parser = argparse.ArgumentParser(description="T-SNE of expert representations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--output", type=str, default="expert_tsne.png")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = load_expert_weights(args.checkpoint)

    print(f"Extracting expert representations (num_experts={args.num_experts})...")
    expert_vectors, layer_labels = extract_expert_representations(
        state_dict, args.num_experts
    )

    # Check we found experts
    total = sum(len(v) for v in expert_vectors.values())
    if total == 0:
        print("No expert weights found in checkpoint!")
        return

    for eid, vecs in expert_vectors.items():
        print(f"  Expert {eid}: {len(vecs)} weight matrices")

    # Truncate to same dim
    expert_vectors = truncate_to_min_length(expert_vectors)

    # Build data matrix for T-SNE
    all_vectors = []
    all_expert_ids = []
    all_layer_ids = []

    for expert_id, vecs in expert_vectors.items():
        for i, v in enumerate(vecs):
            all_vectors.append(v)
            all_expert_ids.append(expert_id)
            all_layer_ids.append(layer_labels[expert_id][i])

    X = np.array(all_vectors)
    expert_ids = np.array(all_expert_ids)
    layer_ids = np.array(all_layer_ids)

    # PCA first to reduce dimensionality (T-SNE can't handle millions of dims)
    pca_dim = min(50, X.shape[0] - 1, X.shape[1])
    print(f"\nPCA: {X.shape[1]} dims -> {pca_dim} dims...")
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    print(f"Running T-SNE on {X_pca.shape[0]} vectors...")
    perplexity = min(args.perplexity, X_pca.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_pca)

    # Plot
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by expert
    ax = axes[0]
    for eid in range(args.num_experts):
        mask = expert_ids == eid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[eid % len(colors)], label=f"Expert {eid}",
                   alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
    ax.set_title("Expert Weight Representations (T-SNE)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")

    # Right: colored by layer
    ax = axes[1]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=layer_ids, cmap="viridis",
                         alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label="Layer index")
    ax.set_title("Layer Depth (T-SNE)", fontsize=14)
    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
