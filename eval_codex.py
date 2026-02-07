#!/usr/bin/env python3
"""
Eval Codex - Track code generation quality across epochs.

Usage:
    python eval_codex.py --checkpoint checkpoints/pacific-prime-code
    python eval_codex.py --checkpoint checkpoints/pacific-prime-code --output codex_results.csv
"""

import argparse
import csv
import json
import torch
from pathlib import Path
from safetensors.torch import load_file
from tokenizers import Tokenizer
from complexity_deep import DeepForCausalLM, DeepConfig

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

CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def load_specific_checkpoint(ckpt_dir, ckpt_file, tokenizer_dir=None, device=None):
    """Load model from a specific checkpoint file."""
    ckpt_dir = Path(ckpt_dir)
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else ckpt_dir
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(tokenizer_dir / "config.json", "r") as f:
        cfg = json.load(f)

    config = DeepConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rope_theta=cfg["rope_theta"],
        rms_norm_eps=cfg["rms_norm_eps"],
        attention_dropout=cfg["attention_dropout"],
        hidden_act=cfg["hidden_act"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        use_token_routed_mlp=cfg.get("use_token_routed_mlp", True),
        num_experts=cfg.get("num_experts", 4),
        use_qk_norm=cfg.get("use_qk_norm", True),
        use_sdpa=cfg.get("use_sdpa", True),
    )

    model = DeepForCausalLM(config)

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(unexpected) == len(state_dict):
        stripped = {k.removeprefix("model."): v for k, v in state_dict.items()}
        model.load_state_dict(stripped, strict=False)

    model.eval()
    model = model.to(device)

    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    return model, tokenizer, config, device


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=300, temperature=0.7,
             top_k=50, top_p=0.9, repetition_penalty=1.2, device="cpu"):
    """Generate text from a prompt."""
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long).to(device)
    generated_ids = input_ids.clone()
    generated_set = set(input_ids[0].tolist())

    for _ in range(max_tokens):
        outputs = model(generated_ids)
        next_logits = outputs.logits[0, -1, :].float()

        # Repetition penalty
        for token_id in generated_set:
            next_logits[token_id] /= repetition_penalty

        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-k
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float("-inf")

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
        generated_set.add(next_token.item())

        if next_token.item() == 0:
            break

    return tokenizer.decode(generated_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Eval Codex - epoch diff tracker")
    parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint directory")
    parser.add_argument("--tokenizer", "-t", default=None, help="Tokenizer/config directory (default: same as checkpoint)")
    parser.add_argument("--output", "-o", default="codex_results.csv", help="Output CSV")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--latest", action="store_true", help="Only test the latest checkpoint")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    tokenizer_dir = Path(args.tokenizer) if args.tokenizer else ckpt_dir
    checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"),
                         key=lambda f: int(f.stem.split("epoch")[1]))

    if not checkpoints:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    if args.latest:
        checkpoints = [checkpoints[-1]]

    print(f"Testing {len(checkpoints)} checkpoint(s)")
    print(f"Running {len(PROMPTS)} prompts per checkpoint\n")

    rows = []

    for ckpt_file in checkpoints:
        epoch_num = int(ckpt_file.stem.split("epoch")[1])
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch_num} - {ckpt_file.name}")
        print(f"{'='*60}")

        model, tokenizer, config, device = load_specific_checkpoint(ckpt_dir, ckpt_file, tokenizer_dir=tokenizer_dir)

        for i, prompt in enumerate(PROMPTS):
            formatted = CHAT_TEMPLATE.format(prompt=prompt)
            print(f"\n--- Prompt {i+1}/{len(PROMPTS)} ---")
            print(f">>> {prompt}\n")

            output = generate(
                model, tokenizer, formatted,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
            )

            # Extract assistant response
            response = output.split("<|im_start|>assistant\n")[-1]
            response = response.replace("<|im_end|>", "").strip()
            print(response[:500])

            rows.append({
                "epoch": epoch_num,
                "prompt_id": i + 1,
                "prompt": prompt,
                "response": response,
                "response_len": len(response),
            })

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "prompt_id", "prompt", "response", "response_len"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\nResults saved to {args.output}")
    print(f"Total: {len(rows)} generations ({len(checkpoints)} epochs x {len(PROMPTS)} prompts)")


if __name__ == "__main__":
    main()
