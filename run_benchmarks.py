"""
Benchmark evaluation for Complexity models.

Evaluates on standard NLP benchmarks using log-probability scoring:
- MMLU (Hendrycks et al., 2021)
- HellaSwag (Zellers et al., 2019)
- ARC Challenge (Clark et al., 2018)
- Winogrande (Sakaguchi et al., 2020)

Usage:
    python run_benchmarks.py --checkpoint ./checkpoints/final \\
                             --tokenizer ./tokenizer \\
                             --benchmarks mmlu hellaswag
"""

import sys
import json
import argparse
import logging
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from supplementary_code.models.modeling import ComplexityModel


def load_model(model_dir: str, device: str = "cuda") -> ComplexityModel:
    """Load a Complexity model from a directory."""
    model = ComplexityModel.from_pretrained(model_dir, device=device)
    model.eval()
    return model


@torch.no_grad()
def score_text(model, tokenizer, text: str, device: str) -> float:
    """Sum of log-probabilities for each token given its prefix."""
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    ids = {k: v.to(device) for k, v in ids.items()}
    out = model(ids["input_ids"])
    logits = out.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    tokens = ids["input_ids"][0]
    total = sum(log_probs[0, i - 1, tokens[i]].item() for i in range(1, len(tokens)))
    return total


def run_hellaswag(model, tokenizer, device, max_samples=500):
    """HellaSwag: pick best continuation."""
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    correct = 0
    for sample in tqdm(ds, desc="HellaSwag"):
        scores = [score_text(model, tokenizer, f"{sample['ctx']} {e}", device)
                  for e in sample["endings"]]
        if scores.index(max(scores)) == int(sample["label"]):
            correct += 1
    acc = correct / len(ds) * 100
    logging.info(f"HellaSwag: {acc:.2f}%")
    return acc


def run_arc(model, tokenizer, device, max_samples=500):
    """ARC-Challenge."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    correct = 0
    total = 0
    for s in tqdm(ds, desc="ARC-Challenge"):
        choices = s["choices"]["text"]
        labels = s["choices"]["label"]
        try:
            answer_idx = labels.index(s["answerKey"])
        except ValueError:
            continue
        prompt = f"Question: {s['question']}\nAnswer:"
        scores = [score_text(model, tokenizer, f"{prompt} {c}", device) for c in choices]
        if scores.index(max(scores)) == answer_idx:
            correct += 1
        total += 1
    acc = correct / total * 100
    logging.info(f"ARC-Challenge: {acc:.2f}%")
    return acc


def run_mmlu(model, tokenizer, device, max_samples=500):
    """MMLU (multiple choice)."""
    try:
        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    except Exception:
        ds = load_dataset("lukaemon/mmlu", "all", split="test", trust_remote_code=True)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    correct = 0
    for s in tqdm(ds, desc="MMLU"):
        q = s["question"]
        choices = s["choices"]
        answer = s["answer"]
        if isinstance(answer, str):
            answer = ord(answer.upper()) - ord("A")
        prompt = f"Question: {q}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
        scores = [score_text(model, tokenizer, f"{prompt} {c}", device) for c in ["A", "B", "C", "D"]]
        if scores.index(max(scores)) == answer:
            correct += 1
    acc = correct / len(ds) * 100
    logging.info(f"MMLU: {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on Complexity model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--benchmarks", nargs="+", default=["hellaswag", "arc", "mmlu"])
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    model = load_model(args.checkpoint, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    if "hellaswag" in args.benchmarks:
        results["hellaswag"] = run_hellaswag(model, tokenizer, args.device, args.max_samples)
    if "arc" in args.benchmarks:
        results["arc_challenge"] = run_arc(model, tokenizer, args.device, args.max_samples)
    if "mmlu" in args.benchmarks:
        results["mmlu"] = run_mmlu(model, tokenizer, args.device, args.max_samples)

    logging.info("=" * 40)
    for name, score in results.items():
        logging.info(f"  {name:20s}: {score:.2f}%")

    with open(args.output, "w") as f:
        json.dump({"results": results, "timestamp": datetime.now().isoformat()}, f, indent=2)
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
