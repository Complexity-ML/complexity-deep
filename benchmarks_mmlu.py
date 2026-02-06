"""
MMLU Full Benchmark (100% of test set)
=======================================
Evaluates COMPLEXITY-DEEP on the complete MMLU test set (14,042 questions).
Reports per-subject and overall accuracy.

Usage:
    python benchmarks_mmlu.py --checkpoint ./checkpoints/checkpoint_epoch76.pt
    python benchmarks_mmlu.py --checkpoint ./checkpoints/final.pt --tokenizer ./checkpoints
"""

import torch
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

from run_benchmarks import load_model, load_tokenizer, get_logprobs


@torch.no_grad()
def run_mmlu_full(model, tokenizer, device: str = "cuda"):
    """Run MMLU on 100% of the test set with per-subject breakdown."""

    logging.info("Loading MMLU test set (all subjects)...")
    try:
        dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    except:
        dataset = load_dataset("lukaemon/mmlu", "all", split="test", trust_remote_code=True)

    logging.info(f"Total MMLU test samples: {len(dataset)}")

    correct = 0
    total = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for sample in tqdm(dataset, desc="MMLU Full"):
        question = sample["question"]
        choices = [sample["choices"][i] for i in range(len(sample["choices"]))]
        answer = sample["answer"]
        subject = sample.get("subject", "unknown")

        if isinstance(answer, str):
            answer = ord(answer.upper()) - ord('A')

        user_msg = f"Question: {question}\n\nChoices:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
        prompt = f"User: {user_msg}\n\nAssistant: The answer is"

        choice_letters = ["A", "B", "C", "D"]
        scores = []
        for letter in choice_letters:
            completion = f" {letter}"
            score = get_logprobs(model, tokenizer, prompt, completion, device)
            scores.append(score)

        predicted = scores.index(max(scores))

        if predicted == answer:
            correct += 1
            subject_stats[subject]["correct"] += 1
        total += 1
        subject_stats[subject]["total"] += 1

        # Log progress every 1000 samples
        if total % 1000 == 0:
            logging.info(f"  Progress: {total}/{len(dataset)} - Running accuracy: {correct/total*100:.2f}%")

    overall_accuracy = correct / total * 100

    # Compute per-subject accuracy
    subject_results = {}
    for subject, stats in sorted(subject_stats.items()):
        acc = stats["correct"] / stats["total"] * 100
        subject_results[subject] = {
            "accuracy": round(acc, 2),
            "correct": stats["correct"],
            "total": stats["total"]
        }

    return overall_accuracy, subject_results, correct, total


def main():
    parser = argparse.ArgumentParser(description="MMLU Full Benchmark (100%)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/checkpoint_epoch76.pt")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="mmlu_full_results.json")

    args = parser.parse_args()

    checkpoint_dir = str(Path(args.checkpoint).parent)
    if args.config is None:
        args.config = str(Path(checkpoint_dir) / "config.json")
    if args.tokenizer is None:
        args.tokenizer = checkpoint_dir

    log_file = f"mmlu_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 60)
    logging.info("MMLU FULL BENCHMARK (100%)")
    logging.info("=" * 60)
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Config: {args.config}")
    logging.info(f"Tokenizer: {args.tokenizer}")

    model = load_model(args.checkpoint, args.config, args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    overall_acc, subject_results, correct, total = run_mmlu_full(model, tokenizer, args.device)

    # Summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("MMLU FULL RESULTS")
    logging.info("=" * 60)
    logging.info(f"Overall: {overall_acc:.2f}% ({correct}/{total})")
    logging.info("")
    logging.info("Per-subject breakdown:")
    for subject, stats in sorted(subject_results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        logging.info(f"  {subject:40s}: {stats['accuracy']:6.2f}% ({stats['correct']}/{stats['total']})")
    logging.info("=" * 60)

    # Save
    output_data = {
        "overall_accuracy": round(overall_acc, 2),
        "correct": correct,
        "total": total,
        "per_subject": subject_results,
        "metadata": {
            "checkpoint": args.checkpoint,
            "timestamp": datetime.now().isoformat(),
            "samples": "100% (full test set)"
        }
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
