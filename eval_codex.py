#!/usr/bin/env python3
"""
Eval Codex - Track code generation quality across epochs.

Usage:
    python eval_codex.py --checkpoint checkpoints/pacific-prime-code
    python eval_codex.py --checkpoint checkpoints/pacific-prime-code --output codex_results.csv
"""

import argparse
import csv
import os
from pathlib import Path
from generate import load_model, generate

PROMPTS = [
    # Level 1 - Basic
    "Write a Python function that reverses a string.",
    "Write a Python function that checks if a number is prime.",
    "Write a Python function that returns the factorial of n.",
    # Level 2 - Medium
    "Write a Python function that implements binary search on a sorted list.",
    "Write a Python class for a stack with push, pop, and peek methods.",
    "Write a Python function that finds all duplicates in a list.",
    # Level 3 - Hard
    "Write a Python function that implements merge sort.",
    "Write a Python class for a binary search tree with insert and search.",
    # Level 4 - Complex
    "Write a Python recursive descent parser for arithmetic expressions (+-*/).",
    "Write a Python function that solves the N-Queens problem using backtracking.",
]


def get_epoch_from_checkpoint(ckpt_path):
    """Extract epoch number from checkpoint filename."""
    for f in sorted(Path(ckpt_path).glob("checkpoint_epoch*.pt")):
        name = f.stem
        if "epoch" in name:
            yield int(name.split("epoch")[1]), f
    for f in sorted(Path(ckpt_path).glob("step_*.pt")):
        yield int(f.stem.split("_")[1]), f


def main():
    parser = argparse.ArgumentParser(description="Eval Codex - epoch diff tracker")
    parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint directory")
    parser.add_argument("--output", "-o", default="codex_results.csv", help="Output CSV")
    parser.add_argument("--max_tokens", type=int, default=300, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.3, help="Low temp for reproducibility")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    checkpoints = list(get_epoch_from_checkpoint(ckpt_dir))

    if not checkpoints:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Running {len(PROMPTS)} prompts per checkpoint\n")

    rows = []

    for epoch_num, ckpt_file in checkpoints:
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch_num}")
        print(f"{'='*60}")

        # Temporarily copy checkpoint as the "latest" for load_model
        model, tokenizer, config, device = load_model(str(ckpt_dir))

        for i, prompt in enumerate(PROMPTS):
            print(f"\n--- Prompt {i+1}/{len(PROMPTS)} ---")
            print(f">>> {prompt}\n")

            output = generate(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
                stream=True,
            )

            # Remove prompt from output
            response = output[len(prompt):].strip()

            rows.append({
                "epoch": epoch_num,
                "prompt_id": i + 1,
                "prompt": prompt,
                "response": response,
                "response_len": len(response),
            })

        # Free memory
        del model
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "prompt_id", "prompt", "response", "response_len"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\nResults saved to {args.output}")
    print(f"Total: {len(rows)} generations ({len(checkpoints)} epochs x {len(PROMPTS)} prompts)")


if __name__ == "__main__":
    main()
