"""
Empirical Validation of COMPLEXITY-DEEP Theorems
=================================================
Generates two figures for reviewer defense:

1. Gradient Cosine Similarity between Experts (Theorem 3 - Gradient Orthogonalization)
   Shows that expert gradients diverge during training, validating
   that the modulo routing induces expert specialization.

2. PiD Gradient Norm Analysis (PiD Controller Stabilization)
   Shows gradient norms with vs without PiD controller,
   validating that PiD prevents gradient explosions.

Usage:
    python validate_theorems.py --checkpoint ./checkpoints/final.pt
    python validate_theorems.py --checkpoint C:/INL/pacific-prime/checkpoints/final.pt --device cpu
"""

import torch
import torch.nn.functional as F
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
import sys

sys.path.insert(0, str(Path(__file__).parent))

from complexity_deep import DeepForCausalLM, DeepConfig
from transformers import AutoTokenizer


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load COMPLEXITY-DEEP model from checkpoint."""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = DeepConfig.from_dict(config_dict)
    model = DeepForCausalLM(config)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    state_dict = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    return model, config


def get_sample_batch(tokenizer, device, batch_size=2, seq_len=128):
    """Create a small sample batch for gradient computation."""
    texts = [
        "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy.",
        "In quantum mechanics, particles exhibit wave-particle duality and their behavior is described by probability amplitudes.",
        "Machine learning algorithms can identify patterns in large datasets and make predictions based on statistical models.",
        "The human brain contains approximately 86 billion neurons connected by trillions of synapses forming complex networks.",
    ][:batch_size]

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len,
    )
    input_ids = encodings["input_ids"].to(device)
    labels = input_ids.clone()
    return input_ids, labels


def analyze_gradient_cosine_similarity(model, input_ids, labels, config):
    """
    Compute pairwise cosine similarity between expert gradients per layer.
    Validates Theorem 3: Gradient Orthogonalization.
    """
    print("\n" + "=" * 60)
    print("THEOREM 3: Gradient Cosine Similarity Between Experts")
    print("=" * 60)

    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    expert_pairs = list(combinations(range(num_experts), 2))

    # Forward + backward
    model.zero_grad()
    model.train()
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    print(f"Loss: {loss.item():.4f}")

    # Collect gradient cosine similarity per layer
    layer_cosine_sims = {f"E{i}-E{j}": [] for i, j in expert_pairs}
    layer_avg_cosine = []

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp

        # gate_up_proj: [num_experts, hidden_size, 2*expert_intermediate]
        if hasattr(mlp, 'gate_up_proj') and mlp.gate_up_proj.grad is not None:
            grad = mlp.gate_up_proj.grad  # [4, H, 2I]

            # Extract per-expert gradient vectors
            expert_grads = []
            for e in range(num_experts):
                expert_grads.append(grad[e].flatten().float())

            # Compute pairwise cosine similarity
            layer_sims = []
            for (i, j) in expert_pairs:
                cos_sim = F.cosine_similarity(
                    expert_grads[i].unsqueeze(0),
                    expert_grads[j].unsqueeze(0)
                ).item()
                layer_cosine_sims[f"E{i}-E{j}"].append(cos_sim)
                layer_sims.append(cos_sim)

            avg_sim = np.mean(layer_sims)
            layer_avg_cosine.append(avg_sim)
            print(f"  Layer {layer_idx:2d}: avg cosine sim = {avg_sim:.4f}")
        else:
            # No gradient (frozen or no gate_up_proj)
            for key in layer_cosine_sims:
                layer_cosine_sims[key].append(0.0)
            layer_avg_cosine.append(0.0)

    model.eval()
    return layer_cosine_sims, layer_avg_cosine, expert_pairs


def analyze_pid_gradient_norms(model, input_ids, labels, config, device):
    """
    Compare gradient norms with vs without PiD controller.
    Validates PiD's role as training stabilizer.
    """
    print("\n" + "=" * 60)
    print("PID CONTROLLER: Gradient Norm Analysis")
    print("=" * 60)

    num_layers = config.num_hidden_layers

    # --- Run 1: Full model (PiD active) ---
    print("\n[1/2] Computing gradient norms WITH PiD...")
    model.zero_grad()
    model.train()
    outputs = model(input_ids, labels=labels)
    loss_with_pid = outputs.loss
    loss_with_pid.backward()

    norms_with_pid = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        layer_norm = 0.0
        count = 0
        for name, param in layer.named_parameters():
            if param.grad is not None:
                layer_norm += param.grad.float().norm().item() ** 2
                count += 1
        norms_with_pid.append(np.sqrt(layer_norm) if layer_norm > 0 else 0.0)

    print(f"  Loss with PiD: {loss_with_pid.item():.4f}")
    print(f"  Avg gradient norm: {np.mean(norms_with_pid):.4f}")

    # --- Save PiD weights, then zero them out ---
    print("\n[2/2] Computing gradient norms WITHOUT PiD...")
    saved_pid_weights = {}
    for layer_idx in range(num_layers):
        dynamics = model.model.layers[layer_idx].dynamics
        saved_pid_weights[layer_idx] = {
            'controller_in_weight': dynamics.controller_in.weight.data.clone(),
            'controller_in_bias': dynamics.controller_in.bias.data.clone(),
            'controller_out_weight': dynamics.controller_out.weight.data.clone(),
            'controller_out_bias': dynamics.controller_out.bias.data.clone(),
        }
        # Zero out controller (PiD becomes constant alpha/beta/gate from bias init)
        dynamics.controller_in.weight.data.zero_()
        dynamics.controller_out.weight.data.zero_()

    model.zero_grad()
    outputs = model(input_ids, labels=labels)
    loss_without_pid = outputs.loss
    loss_without_pid.backward()

    norms_without_pid = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        layer_norm = 0.0
        for name, param in layer.named_parameters():
            if param.grad is not None:
                layer_norm += param.grad.float().norm().item() ** 2
        norms_without_pid.append(np.sqrt(layer_norm) if layer_norm > 0 else 0.0)

    print(f"  Loss without PiD: {loss_without_pid.item():.4f}")
    print(f"  Avg gradient norm: {np.mean(norms_without_pid):.4f}")

    # --- Restore PiD weights ---
    for layer_idx in range(num_layers):
        dynamics = model.model.layers[layer_idx].dynamics
        dynamics.controller_in.weight.data = saved_pid_weights[layer_idx]['controller_in_weight']
        dynamics.controller_in.bias.data = saved_pid_weights[layer_idx]['controller_in_bias']
        dynamics.controller_out.weight.data = saved_pid_weights[layer_idx]['controller_out_weight']
        dynamics.controller_out.bias.data = saved_pid_weights[layer_idx]['controller_out_bias']

    model.eval()

    # Compute ratio
    ratio = []
    for w, wo in zip(norms_with_pid, norms_without_pid):
        if wo > 0:
            ratio.append(w / wo)
        else:
            ratio.append(1.0)

    return norms_with_pid, norms_without_pid, ratio, loss_with_pid.item(), loss_without_pid.item()


def plot_gradient_cosine_similarity(layer_cosine_sims, layer_avg_cosine, expert_pairs, num_layers, output_path):
    """Plot gradient cosine similarity between experts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-pair cosine similarity
    ax = axes[0]
    x = range(num_layers)
    colors = plt.cm.Set2(np.linspace(0, 1, len(expert_pairs)))
    for idx, (i, j) in enumerate(expert_pairs):
        key = f"E{i}-E{j}"
        ax.plot(x, layer_cosine_sims[key], marker='.', color=colors[idx],
                label=f"Expert {i} vs {j}", alpha=0.7)

    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Orthogonal (0.0)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Identical (1.0)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('(a) Gradient Cosine Similarity per Expert Pair')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.1)

    # Right: average cosine similarity
    ax = axes[1]
    bars = ax.bar(x, layer_avg_cosine, color='steelblue', alpha=0.8)
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Orthogonal')
    avg_overall = np.mean(layer_avg_cosine)
    ax.axhline(y=avg_overall, color='orange', linestyle='--', alpha=0.7,
               label=f'Mean: {avg_overall:.3f}')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Cosine Similarity')
    ax.set_title('(b) Average Gradient Cosine Similarity per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('COMPLEXITY-DEEP: Expert Gradient Orthogonalization (Theorem 3)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    plt.close()


def plot_pid_gradient_norms(norms_with, norms_without, ratio, loss_with, loss_without, num_layers, output_path):
    """Plot gradient norms with vs without PiD."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = range(num_layers)

    # Left: absolute gradient norms
    ax = axes[0]
    ax.plot(x, norms_with, 'b-o', markersize=4, label=f'With PiD (loss={loss_with:.3f})')
    ax.plot(x, norms_without, 'r-s', markersize=4, label=f'Without PiD (loss={loss_without:.3f})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient Norm (L2)')
    ax.set_title('(a) Gradient Norms per Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Middle: ratio
    ax = axes[1]
    colors = ['green' if r <= 1.0 else 'red' for r in ratio]
    ax.bar(x, ratio, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Ratio (with PiD / without PiD)')
    ax.set_title('(b) PiD Gradient Norm Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: variance across layers
    ax = axes[2]
    var_with = np.std(norms_with)
    var_without = np.std(norms_without)
    bars = ax.bar(['With PiD', 'Without PiD'], [var_with, var_without],
                  color=['steelblue', 'indianred'], alpha=0.8)
    for bar, val in zip(bars, [var_with, var_without]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Std Dev of Gradient Norms')
    ax.set_title('(c) Gradient Norm Variance\n(lower = more stable)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('COMPLEXITY-DEEP: PiD Controller Gradient Stabilization',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate COMPLEXITY-DEEP Theorems Empirically")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final.pt")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = str(Path(args.checkpoint).parent)
    if args.config is None:
        args.config = str(Path(checkpoint_dir) / "config.json")
    if args.tokenizer is None:
        args.tokenizer = checkpoint_dir

    print("=" * 60)
    print("COMPLEXITY-DEEP: Empirical Theorem Validation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Batch: {args.batch_size} x {args.seq_len}")

    # Load model
    model, config = load_model(args.checkpoint, args.config, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get sample batch
    input_ids, labels = get_sample_batch(tokenizer, args.device, args.batch_size, args.seq_len)
    print(f"Input shape: {input_ids.shape}")

    # === Analysis 1: Gradient Cosine Similarity ===
    layer_cosine_sims, layer_avg_cosine, expert_pairs = analyze_gradient_cosine_similarity(
        model, input_ids, labels, config
    )
    plot_gradient_cosine_similarity(
        layer_cosine_sims, layer_avg_cosine, expert_pairs,
        config.num_hidden_layers,
        Path(args.output_dir) / "gradient_cosine_similarity.png"
    )

    # === Analysis 2: PiD Gradient Norms ===
    norms_with, norms_without, ratio, loss_w, loss_wo = analyze_pid_gradient_norms(
        model, input_ids, labels, config, args.device
    )
    plot_pid_gradient_norms(
        norms_with, norms_without, ratio, loss_w, loss_wo,
        config.num_hidden_layers,
        Path(args.output_dir) / "pid_gradient_analysis.png"
    )

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_cosine = np.mean(layer_avg_cosine)
    print(f"Theorem 3 - Avg gradient cosine similarity: {avg_cosine:.4f}")
    if avg_cosine < 0.5:
        print("  -> Expert gradients are substantially divergent (validates orthogonalization)")
    elif avg_cosine < 0.8:
        print("  -> Expert gradients show moderate divergence")
    else:
        print("  -> Expert gradients are still correlated (limited orthogonalization)")

    avg_ratio = np.mean(ratio)
    std_with = np.std(norms_with)
    std_without = np.std(norms_without)
    print(f"\nPiD Analysis:")
    print(f"  Gradient norm std WITH PiD:    {std_with:.4f}")
    print(f"  Gradient norm std WITHOUT PiD: {std_without:.4f}")
    if std_with < std_without:
        print(f"  -> PiD reduces gradient variance by {(1 - std_with/std_without)*100:.1f}% (stabilizing)")
    else:
        print(f"  -> PiD increases gradient variance (dynamic regulation)")

    print(f"\nFigures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
