"""
COMPLEXITY-DEEP Ablation Study
===============================
Evaluates model performance with each component disabled:
1. Full model (baseline)
2. Without Mu-Guidance (zero mu influence on attention)
3. Without Token-Routing (single expert for all tokens)
4. Without PiD Controller (fixed alpha/beta/gate)

Usage:
    python run_ablations.py --checkpoint ./checkpoints/final.pt
"""

import argparse
import json
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from complexity_deep.models.modeling import DeepForCausalLM
from complexity_deep.models.config import ComplexityConfig
from transformers import PreTrainedTokenizerFast


def load_model(checkpoint_path, device="cuda"):
    """Load model from checkpoint."""
    checkpoint_dir = Path(checkpoint_path).parent if checkpoint_path.endswith('.pt') else Path(checkpoint_path)
    checkpoint_file = checkpoint_path if checkpoint_path.endswith('.pt') else str(Path(checkpoint_path) / "final.pt")

    # Load config
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = ComplexityConfig(**config_dict)

    # Load model
    model = DeepForCausalLM(config)
    state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(checkpoint_dir))

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer, config, device


# ============================================================================
# Ablation contexts - temporarily disable components
# ============================================================================

@contextmanager
def ablate_mu_guidance(model):
    """Disable Mu-Guidance: zero out mu influence on attention K, Q, V."""
    saved_weights = {}

    for name, module in model.named_modules():
        # Zero out mu_to_k, mu_to_q, mu_to_v in attention
        if hasattr(module, 'mu_to_k'):
            saved_weights[f"{name}.mu_to_k"] = module.mu_to_k.weight.data.clone()
            module.mu_to_k.weight.data.zero_()
        if hasattr(module, 'mu_to_q'):
            saved_weights[f"{name}.mu_to_q"] = module.mu_to_q.weight.data.clone()
            module.mu_to_q.weight.data.zero_()
        if hasattr(module, 'mu_to_v'):
            saved_weights[f"{name}.mu_to_v"] = module.mu_to_v.weight.data.clone()
            module.mu_to_v.weight.data.zero_()
        # Also zero mu_router in MLP if present
        if hasattr(module, 'mu_router'):
            saved_weights[f"{name}.mu_router"] = module.mu_router.weight.data.clone()
            module.mu_router.weight.data.zero_()

    print(f"  [Ablation] Zeroed {len(saved_weights)} mu-guidance weights")
    try:
        yield
    finally:
        # Restore weights
        for name, module in model.named_modules():
            if hasattr(module, 'mu_to_k') and f"{name}.mu_to_k" in saved_weights:
                module.mu_to_k.weight.data.copy_(saved_weights[f"{name}.mu_to_k"])
            if hasattr(module, 'mu_to_q') and f"{name}.mu_to_q" in saved_weights:
                module.mu_to_q.weight.data.copy_(saved_weights[f"{name}.mu_to_q"])
            if hasattr(module, 'mu_to_v') and f"{name}.mu_to_v" in saved_weights:
                module.mu_to_v.weight.data.copy_(saved_weights[f"{name}.mu_to_v"])
            if hasattr(module, 'mu_router') and f"{name}.mu_router" in saved_weights:
                module.mu_router.weight.data.copy_(saved_weights[f"{name}.mu_router"])
        print(f"  [Ablation] Restored mu-guidance weights")


@contextmanager
def ablate_token_routing(model):
    """Disable Token-Routing: force all tokens to expert 0."""
    saved_buffers = {}

    for name, module in model.named_modules():
        if hasattr(module, 'token_to_expert'):
            saved_buffers[name] = module.token_to_expert.data.clone()
            # Force all tokens to expert 0
            module.token_to_expert.data.zero_()

    print(f"  [Ablation] Forced {len(saved_buffers)} routing tables to expert 0")
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if name in saved_buffers and hasattr(module, 'token_to_expert'):
                module.token_to_expert.data.copy_(saved_buffers[name])
        print(f"  [Ablation] Restored token routing")


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
            # Zero controller input so output is just bias
            module.controller_in.weight.data.zero_()
            module.controller_in.bias.data.zero_()

    print(f"  [Ablation] Fixed {len(saved_weights)//4} PiD controllers to constant output")
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if hasattr(module, 'controller_in') and f"{name}.controller_in.weight" in saved_weights:
                module.controller_in.weight.data.copy_(saved_weights[f"{name}.controller_in.weight"])
                module.controller_in.bias.data.copy_(saved_weights[f"{name}.controller_in.bias"])
                module.controller_out.weight.data.copy_(saved_weights[f"{name}.controller_out.weight"])
                module.controller_out.bias.data.copy_(saved_weights[f"{name}.controller_out.bias"])
        print(f"  [Ablation] Restored PiD controllers")


# ============================================================================
# Evaluation (reuse logic from run_benchmarks.py)
# ============================================================================

def compute_log_prob(model, tokenizer, prompt, completion, device):
    """Compute log probability of completion given prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    input_ids = prompt_ids + completion_ids
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor, token_ids=input_tensor)
        logits = outputs.logits

    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

    total_log_prob = 0.0
    start_idx = len(prompt_ids) - 1
    for i, token_id in enumerate(completion_ids):
        total_log_prob += log_probs[start_idx + i, token_id].item()

    return total_log_prob / len(completion_ids) if completion_ids else 0.0


def eval_mmlu(model, tokenizer, device, max_samples=200):
    """Evaluate MMLU benchmark."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")

    correct = 0
    total = 0
    choices = ["A", "B", "C", "D"]

    for i, example in enumerate(ds):
        if i >= max_samples:
            break

        question = example["question"]
        options = example["choices"]
        answer_idx = example["answer"]

        prompt = f"Question: {question}\n"
        for j, opt in enumerate(options):
            prompt += f"{choices[j]}. {opt}\n"
        prompt += "Answer:"

        best_score = float('-inf')
        best_choice = -1
        for j, choice in enumerate(choices[:len(options)]):
            score = compute_log_prob(model, tokenizer, prompt, f" {choice}", device)
            if score > best_score:
                best_score = score
                best_choice = j

        if best_choice == answer_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def eval_hellaswag(model, tokenizer, device, max_samples=200):
    """Evaluate HellaSwag benchmark."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        if i >= max_samples:
            break

        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        best_score = float('-inf')
        best_choice = -1
        for j, ending in enumerate(endings):
            score = compute_log_prob(model, tokenizer, ctx, ending, device)
            if score > best_score:
                best_score = score
                best_choice = j

        if best_choice == label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def eval_arc(model, tokenizer, device, subset="ARC-Challenge", max_samples=200):
    """Evaluate ARC benchmark."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split="test")

    correct = 0
    total = 0

    for i, example in enumerate(ds):
        if i >= max_samples:
            break

        question = example["question"]
        choices_text = example["choices"]["text"]
        choices_label = example["choices"]["label"]
        answer_key = example["answerKey"]

        prompt = f"Question: {question}\nAnswer:"

        best_score = float('-inf')
        best_label = None
        for text, label in zip(choices_text, choices_label):
            score = compute_log_prob(model, tokenizer, prompt, f" {text}", device)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label == answer_key:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def run_all_benchmarks(model, tokenizer, device, max_samples=200):
    """Run all benchmarks and return results dict."""
    results = {}

    print("    MMLU...", end=" ", flush=True)
    results["MMLU"] = eval_mmlu(model, tokenizer, device, max_samples)
    print(f"{results['MMLU']*100:.1f}%")

    print("    HellaSwag...", end=" ", flush=True)
    results["HellaSwag"] = eval_hellaswag(model, tokenizer, device, max_samples)
    print(f"{results['HellaSwag']*100:.1f}%")

    print("    ARC-Challenge...", end=" ", flush=True)
    results["ARC-Challenge"] = eval_arc(model, tokenizer, device, "ARC-Challenge", max_samples)
    print(f"{results['ARC-Challenge']*100:.1f}%")

    print("    ARC-Easy...", end=" ", flush=True)
    results["ARC-Easy"] = eval_arc(model, tokenizer, device, "ARC-Easy", max_samples)
    print(f"{results['ARC-Easy']*100:.1f}%")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="COMPLEXITY-DEEP Ablation Study")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=200, help="Samples per benchmark")
    parser.add_argument("--output", type=str, default="ablation_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("COMPLEXITY-DEEP ABLATION STUDY")
    print("=" * 60)

    model, tokenizer, config, device = load_model(args.checkpoint, args.device)

    all_results = {}

    # 1. Full model (baseline)
    print("\n[1/4] FULL MODEL (baseline)")
    all_results["full_model"] = run_all_benchmarks(model, tokenizer, device, args.max_samples)

    # 2. Without Mu-Guidance
    print("\n[2/4] WITHOUT MU-GUIDANCE")
    with ablate_mu_guidance(model):
        all_results["no_mu_guidance"] = run_all_benchmarks(model, tokenizer, device, args.max_samples)

    # 3. Without Token-Routing
    print("\n[3/4] WITHOUT TOKEN-ROUTING (single expert)")
    with ablate_token_routing(model):
        all_results["no_token_routing"] = run_all_benchmarks(model, tokenizer, device, args.max_samples)

    # 4. Without PiD Controller
    print("\n[4/4] WITHOUT PID CONTROLLER")
    with ablate_pid_controller(model):
        all_results["no_pid_controller"] = run_all_benchmarks(model, tokenizer, device, args.max_samples)

    # ========================================================================
    # Summary table
    # ========================================================================
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)

    benchmarks = ["MMLU", "HellaSwag", "ARC-Challenge", "ARC-Easy"]
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
            score = all_results[key][b] * 100
            avg += score
            # Delta vs baseline
            if key != "full_model":
                delta = score - all_results["full_model"][b] * 100
                row += f" {score:>8.1f}%({delta:+.1f})"
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
        "device": str(device),
    }

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
