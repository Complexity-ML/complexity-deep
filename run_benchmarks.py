"""
Benchmark evaluation for COMPLEXITY-DEEP model
Evaluates on: MMLU, HellaSwag, ARC, Winogrande
"""

import torch
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
import sys

# Add parent directory for imports
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

    # Add 'model.' prefix - checkpoint has 'layers.X' but model expects 'model.layers.X'
    state_dict = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in state_dict.items()}

    # Debug: check weight loading
    model_keys = set(dict(model.named_parameters()).keys())
    ckpt_keys = set(state_dict.keys())
    matched = model_keys & ckpt_keys
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    print(f"Checkpoint keys: {len(ckpt_keys)}, Model params: {len(model_keys)}")
    print(f"Matched: {len(matched)}, Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"Missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"Unexpected keys (first 5): {list(unexpected)[:5]}")

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    return model


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer."""
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


@torch.no_grad()
def get_logprobs(model, tokenizer, prompt: str, completion: str, device: str = "cuda"):
    """Get log probabilities for the completion only (not the prompt).

    This is the correct way to evaluate multiple choice: we compute the
    probability of generating the completion given the prompt.

    Important: We concatenate token IDs manually to avoid BPE tokenization
    differences between tokenizing prompt alone vs prompt+completion together.
    """
    # Tokenize prompt with special tokens (BOS)
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=True)["input_ids"]
    prompt_len = prompt_ids.shape[1]

    # Tokenize completion WITHOUT special tokens
    completion_ids = tokenizer(completion, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)["input_ids"]

    # Concatenate manually to preserve exact prompt tokenization
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)

    outputs = model(full_ids)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    # Get log prob of ONLY completion tokens (starting after prompt)
    token_ids = full_ids[0]
    total_logprob = 0.0
    num_completion_tokens = 0

    for i in range(prompt_len, len(token_ids)):
        total_logprob += log_probs[0, i-1, token_ids[i]].item()
        num_completion_tokens += 1

    # Normalize by number of completion tokens to avoid length bias
    if num_completion_tokens > 0:
        total_logprob = total_logprob / num_completion_tokens

    return total_logprob


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, question: str, choices: list, device: str = "cuda"):
    """Evaluate multiple choice question by comparing log probabilities."""
    scores = []
    prompt = f"User: {question}\n\nAssistant:"

    for choice in choices:
        completion = f" {choice}"
        score = get_logprobs(model, tokenizer, prompt, completion, device)
        scores.append(score)

    predicted = scores.index(max(scores))
    return predicted


def run_mmlu(model, tokenizer, device: str = "cuda", max_samples: int = 500):
    """Run MMLU benchmark (subset)."""
    logging.info("")
    logging.info("="*50)
    logging.info("Running MMLU Benchmark")
    logging.info("="*50)

    try:
        dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    except:
        dataset = load_dataset("lukaemon/mmlu", "all", split="test", trust_remote_code=True)

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    correct = 0
    total = 0

    for sample in tqdm(dataset, desc="MMLU"):
        question = sample["question"]
        choices = [sample["choices"][i] for i in range(len(sample["choices"]))]
        answer = sample["answer"]  # 0, 1, 2, or 3

        if isinstance(answer, str):
            answer = ord(answer.upper()) - ord('A')

        # Build prompt with chat template format (matches SFT training)
        user_msg = f"Question: {question}\n\nChoices:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"

        # Compare using chat template format - score just the letter (simpler, more reliable)
        prompt = f"User: {user_msg}\n\nAssistant: The answer is"
        choice_letters = ["A", "B", "C", "D"]
        scores = []
        for i, choice in enumerate(choices):
            completion = f" {choice_letters[i]}"  # Just score the letter
            score = get_logprobs(model, tokenizer, prompt, completion, device)
            scores.append(score)

        predicted = scores.index(max(scores))

        if predicted == answer:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    logging.info(f"MMLU Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def run_hellaswag(model, tokenizer, device: str = "cuda", max_samples: int = 500):
    """Run HellaSwag benchmark."""
    logging.info("")
    logging.info("="*50)
    logging.info("Running HellaSwag Benchmark")
    logging.info("="*50)

    dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    correct = 0
    total = 0

    for sample in tqdm(dataset, desc="HellaSwag"):
        context = sample["ctx"]
        endings = sample["endings"]
        answer = int(sample["label"])

        scores = []
        prompt = f"User: Complete this sentence: {context}\n\nAssistant:"
        for ending in endings:
            # Chat template format - only score the completion
            completion = f" {ending}"
            score = get_logprobs(model, tokenizer, prompt, completion, device)
            scores.append(score)

        predicted = scores.index(max(scores))

        if predicted == answer:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    logging.info(f"HellaSwag Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def run_arc(model, tokenizer, device: str = "cuda", max_samples: int = 500, challenge: bool = True):
    """Run ARC benchmark."""
    subset = "ARC-Challenge" if challenge else "ARC-Easy"
    logging.info("")
    logging.info("="*50)
    logging.info(f"Running ARC ({subset}) Benchmark")
    logging.info("="*50)

    dataset = load_dataset("allenai/ai2_arc", subset, split="test", trust_remote_code=True)

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    correct = 0
    total = 0

    for sample in tqdm(dataset, desc=f"ARC-{subset}"):
        question = sample["question"]
        choices = sample["choices"]["text"]
        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]

        try:
            answer_idx = labels.index(answer_key)
        except ValueError:
            continue

        # Chat template format - only score the completion
        prompt = f"User: Question: {question}\n\nAssistant:"

        scores = []
        for choice in choices:
            completion = f" The answer is {choice}"
            score = get_logprobs(model, tokenizer, prompt, completion, device)
            scores.append(score)

        predicted = scores.index(max(scores))

        if predicted == answer_idx:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    logging.info(f"ARC ({subset}) Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def run_winogrande(model, tokenizer, device: str = "cuda", max_samples: int = 500):
    """Run Winogrande benchmark."""
    logging.info("")
    logging.info("="*50)
    logging.info("Running Winogrande Benchmark")
    logging.info("="*50)

    dataset = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    correct = 0
    total = 0

    for sample in tqdm(dataset, desc="Winogrande"):
        sentence = sample["sentence"]
        option1 = sample["option1"]
        option2 = sample["option2"]
        answer = int(sample["answer"]) - 1  # 1 or 2 -> 0 or 1

        # Replace _ with each option, using chat template format - only score completion
        completed1 = sentence.replace("_", option1)
        completed2 = sentence.replace("_", option2)

        prompt = f"User: Complete the sentence: {sentence}\n\nAssistant:"
        completion1 = f" {completed1}"
        completion2 = f" {completed2}"

        score1 = get_logprobs(model, tokenizer, prompt, completion1, device)
        score2 = get_logprobs(model, tokenizer, prompt, completion2, device)

        predicted = 0 if score1 > score2 else 1

        if predicted == answer:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    logging.info(f"Winogrande Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on COMPLEXITY-DEEP model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config (default: same dir as checkpoint)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer (default: same dir as checkpoint)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max samples per benchmark (for faster testing)")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["mmlu", "hellaswag", "arc", "winogrande"],
                        help="Benchmarks to run")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    # Auto-derive config and tokenizer from checkpoint directory if not provided
    checkpoint_dir = str(Path(args.checkpoint).parent)
    if args.config is None:
        args.config = str(Path(checkpoint_dir) / "config.json")
    if args.tokenizer is None:
        args.tokenizer = checkpoint_dir

    # Setup logging to file and console
    log_file = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger(__name__)

    log.info("="*60)
    log.info("COMPLEXITY-DEEP Benchmark Evaluation")
    log.info("="*60)
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Config: {args.config}")
    log.info(f"Tokenizer: {args.tokenizer}")
    log.info(f"Device: {args.device}")
    log.info(f"Max samples: {args.max_samples}")
    log.info(f"Benchmarks: {args.benchmarks}")
    log.info(f"Log file: {log_file}")
    log.info("="*60)

    # Load model and tokenizer
    model = load_model(args.checkpoint, args.config, args.device)
    tokenizer = load_tokenizer(args.tokenizer)

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # Run benchmarks
    if "mmlu" in args.benchmarks:
        results["mmlu"] = run_mmlu(model, tokenizer, args.device, args.max_samples)

    if "hellaswag" in args.benchmarks:
        results["hellaswag"] = run_hellaswag(model, tokenizer, args.device, args.max_samples)

    if "arc" in args.benchmarks:
        results["arc_challenge"] = run_arc(model, tokenizer, args.device, args.max_samples, challenge=True)
        results["arc_easy"] = run_arc(model, tokenizer, args.device, args.max_samples, challenge=False)

    if "winogrande" in args.benchmarks:
        results["winogrande"] = run_winogrande(model, tokenizer, args.device, args.max_samples)

    # Print summary
    log.info("")
    log.info("="*60)
    log.info("BENCHMARK RESULTS SUMMARY")
    log.info("="*60)
    for benchmark, score in results.items():
        log.info(f"  {benchmark:20s}: {score:.2f}%")
    log.info("="*60)

    # Save results with metadata
    output_data = {
        "results": results,
        "metadata": {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "max_samples": args.max_samples,
            "timestamp": datetime.now().isoformat(),
            "log_file": log_file
        }
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Results saved to {args.output}")
    log.info(f"Full log saved to {log_file}")


if __name__ == "__main__":
    main()
