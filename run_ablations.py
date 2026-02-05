"""
COMPLEXITY-DEEP Ablation Study
===============================
Evaluates model performance with each component disabled:
1. Full model (baseline)
2. Without Mu-Guidance (zero mu influence on attention)
3. Without Token-Routing (single expert for all tokens)
4. Without PiD Controller (fixed alpha/beta/gate)

Uses the EXACT same evaluation format as run_benchmarks.py for comparable results.

Usage:
    python run_ablations.py --checkpoint ./checkpoints/final.pt
"""

import argparse
import json
import torch
import logging
import sys
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from run_benchmarks import (
    load_model,
    load_tokenizer,
    get_logprobs,
    run_mmlu,
    run_hellaswag,
    run_arc,
    run_winogrande,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ============================================================================
# Ablation contexts - temporarily disable components
# ============================================================================

@contextmanager
def ablate_mu_guidance(model):
    """Disable Mu-Guidance: zero out mu influence on attention K, Q, V."""
    saved_weights = {}

    for name, module in model.named_modules():
        if hasattr(module, 'mu_to_k'):
            saved_weights[f"{name}.mu_to_k"] = module.mu_to_k.weight.data.clone()
            module.mu_to_k.weight.data.zero_()
        if hasattr(module, 'mu_to_q'):
            saved_weights[f"{name}.mu_to_q"] = module.mu_to_q.weight.data.clone()
            module.mu_to_q.weight.data.zero_()
        if hasattr(module, 'mu_to_v'):
            saved_weights[f"{name}.mu_to_v"] = module.mu_to_v.weight.data.clone()
            module.mu_to_v.weight.data.zero_()
        if hasattr(module, 'mu_router'):
            saved_weights[f"{name}.mu_router"] = module.mu_router.weight.data.clone()
            module.mu_router.weight.data.zero_()

    logging.info(f"  [Ablation] Zeroed {len(saved_weights)} mu-guidance weights")
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if hasattr(module, 'mu_to_k') and f"{name}.mu_to_k" in saved_weights:
                module.mu_to_k.weight.data.copy_(saved_weights[f"{name}.mu_to_k"])
            if hasattr(module, 'mu_to_q') and f"{name}.mu_to_q" in saved_weights:
                module.mu_to_q.weight.data.copy_(saved_weights[f"{name}.mu_to_q"])
            if hasattr(module, 'mu_to_v') and f"{name}.mu_to_v" in saved_weights:
                module.mu_to_v.weight.data.copy_(saved_weights[f"{name}.mu_to_v"])
            if hasattr(module, 'mu_router') and f"{name}.mu_router" in saved_weights:
                module.mu_router.weight.data.copy_(saved_weights[f"{name}.mu_router"])
        logging.info(f"  [Ablation] Restored mu-guidance weights")


@contextmanager
def ablate_token_routing(model):
    """Disable Token-Routing: force all tokens to expert 0."""
    saved_buffers = {}

    for name, module in model.named_modules():
        if hasattr(module, 'token_to_expert'):
            saved_buffers[name] = module.token_to_expert.data.clone()
            module.token_to_expert.data.zero_()

    logging.info(f"  [Ablation] Forced {len(saved_buffers)} routing tables to expert 0")
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if name in saved_buffers and hasattr(module, 'token_to_expert'):
                module.token_to_expert.data.copy_(saved_buffers[name])
        logging.info(f"  [Ablation] Restored token routing")


@contextmanager
def ablate_pid_controller(model):
    """Disable PiD Controller: fix alpha, beta, gate to default values."""
    saved_weights = {}

    for name, module in model.named_modules():
        if hasattr(module, 'controller_in') and hasattr(module, 'controller_out'):
            saved_weights[f"{name}.controller_in.weight"] = module.controller_in.weight.data.clone()
            saved_weights[f"{name}.controller_in.bias"] = module.controller_in.bias.data.clone()
            saved_weights[f"{name}.controller_out.weight"] = module.controller_out.weight.data.clone()
            saved_weights[f"{name}.controller_out.bias"] = module.controller_out.bias.data.clone()
            module.controller_in.weight.data.zero_()
            module.controller_in.bias.data.zero_()

    logging.info(f"  [Ablation] Fixed {len(saved_weights)//4} PiD controllers to constant output")
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if hasattr(module, 'controller_in') and f"{name}.controller_in.weight" in saved_weights:
                module.controller_in.weight.data.copy_(saved_weights[f"{name}.controller_in.weight"])
                module.controller_in.bias.data.copy_(saved_weights[f"{name}.controller_in.bias"])
                module.controller_out.weight.data.copy_(saved_weights[f"{name}.controller_out.weight"])
                module.controller_out.bias.data.copy_(saved_weights[f"{name}.controller_out.bias"])
        logging.info(f"  [Ablation] Restored PiD controllers")


# ============================================================================
# Run benchmarks (reuses run_benchmarks.py functions)
# ============================================================================

def run_all_benchmarks(model, tokenizer, device, max_samples=200):
    """Run all benchmarks using the same functions as run_benchmarks.py."""
    results = {}

    results["MMLU"] = run_mmlu(model, tokenizer, device, max_samples)
    results["HellaSwag"] = run_hellaswag(model, tokenizer, device, max_samples)
    results["ARC-Challenge"] = run_arc(model, tokenizer, device, max_samples, challenge=True)
    results["ARC-Easy"] = run_arc(model, tokenizer, device, max_samples, challenge=False)
    results["Winogrande"] = run_winogrande(model, tokenizer, device, max_samples)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="COMPLEXITY-DEEP Ablation Study")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final.pt")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=200, help="Samples per benchmark")
    parser.add_argument("--output", type=str, default="ablation_results.json")
    args = parser.parse_args()

    # Resolve paths
    checkpoint_dir = Path(args.checkpoint).parent if args.checkpoint.endswith('.pt') else Path(args.checkpoint)
    config_path = args.config or str(checkpoint_dir / "config.json")
    tokenizer_path = args.tokenizer or str(checkpoint_dir)

    print("=" * 60)
    print("COMPLEXITY-DEEP ABLATION STUDY")
    print("=" * 60)

    model = load_model(args.checkpoint, config_path, args.device)
    tokenizer = load_tokenizer(tokenizer_path)

    all_results = {}

    # 1. Full model (baseline)
    print("\n" + "=" * 60)
    print("[1/4] FULL MODEL (baseline)")
    print("=" * 60)
    all_results["full_model"] = run_all_benchmarks(model, tokenizer, args.device, args.max_samples)

    # 2. Without Mu-Guidance
    print("\n" + "=" * 60)
    print("[2/4] WITHOUT MU-GUIDANCE")
    print("=" * 60)
    with ablate_mu_guidance(model):
        all_results["no_mu_guidance"] = run_all_benchmarks(model, tokenizer, args.device, args.max_samples)

    # 3. Without Token-Routing
    print("\n" + "=" * 60)
    print("[3/4] WITHOUT TOKEN-ROUTING (single expert)")
    print("=" * 60)
    with ablate_token_routing(model):
        all_results["no_token_routing"] = run_all_benchmarks(model, tokenizer, args.device, args.max_samples)

    # 4. Without PiD Controller
    print("\n" + "=" * 60)
    print("[4/4] WITHOUT PID CONTROLLER")
    print("=" * 60)
    with ablate_pid_controller(model):
        all_results["no_pid_controller"] = run_all_benchmarks(model, tokenizer, args.device, args.max_samples)

    # ========================================================================
    # Summary table
    # ========================================================================
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)

    benchmarks = ["MMLU", "HellaSwag", "ARC-Challenge", "ARC-Easy", "Winogrande"]
    configs = [
        ("Full Model", "full_model"),
        ("No Mu-Guidance", "no_mu_guidance"),
        ("No Token-Routing", "no_token_routing"),
        ("No PiD Controller", "no_pid_controller"),
    ]

    # Header
    header = f"{'Configuration':<20}"
    for b in benchmarks:
        header += f" {b:>14}"
    header += f" {'Avg':>8}"
    print(header)
    print("-" * len(header))

    # Rows
    for label, key in configs:
        row = f"{label:<20}"
        avg = 0
        for b in benchmarks:
            score = all_results[key][b]
            avg += score
            if key != "full_model":
                delta = score - all_results["full_model"][b]
                row += f" {score:>8.1f}({delta:+.1f})"
            else:
                row += f" {score:>13.1f}%"
        avg /= len(benchmarks)
        row += f" {avg:>7.1f}%"
        print(row)

    # Save results
    all_results["metadata"] = {
        "checkpoint": args.checkpoint,
        "max_samples": args.max_samples,
        "date": datetime.now().isoformat(),
        "device": str(args.device),
    }

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
