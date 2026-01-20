"""
Complexity Model Fine-Tuning Script
====================================

Fine-tune a pre-trained Complexity model on instruction datasets.

Supports:
- Instruction fine-tuning (SFT)
- Multiple dataset formats (Alpaca, ShareGPT, FLAN, etc.)
- LoRA (optional, for memory efficiency)
- Mixed pre-training data to prevent catastrophic forgetting

Usage:
    # Basic instruction fine-tuning
    python fine_tuning.py --checkpoint ./model --dataset tatsu-lab/alpaca

    # With custom format
    python fine_tuning.py --checkpoint ./model --dataset Open-Orca/OpenOrca --format orca

    # With LoRA (memory efficient)
    python fine_tuning.py --checkpoint ./model --dataset tatsu-lab/alpaca --lora

    # Mix with pre-training data (recommended)
    python fine_tuning.py --checkpoint ./model --dataset tatsu-lab/alpaca --mix-pretrain 0.2
"""

import os
import math
import time
import argparse
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")

# Mixed precision
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False


# ============================================================================
# DATASET FORMATS
# ============================================================================

DATASET_FORMATS = {
    "alpaca": {
        "instruction": "instruction",
        "input": "input",
        "output": "output",
        "template": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
        "template_no_input": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
    },
    "sharegpt": {
        "conversations": "conversations",
        "template": None,  # Special handling
    },
    "orca": {
        "system": "system_prompt",
        "question": "question",
        "response": "response",
        "template": "{system}\n\nUser: {question}\n\nAssistant: {response}",
    },
    "dolly": {
        "instruction": "instruction",
        "context": "context",
        "response": "response",
        "template": "### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}",
        "template_no_context": "### Instruction:\n{instruction}\n\n### Response:\n{response}",
    },
    "simple": {
        "prompt": "prompt",
        "completion": "completion",
        "template": "{prompt}{completion}",
    },
    "qa": {
        "question": "question",
        "answer": "answer",
        "template": "Question: {question}\n\nAnswer: {answer}",
    },
}


def format_example(example: Dict[str, Any], fmt: Dict[str, Any]) -> str:
    """Format a single example according to dataset format."""

    # ShareGPT special handling
    if "conversations" in fmt:
        convs = example.get(fmt["conversations"], [])
        text = ""
        for conv in convs:
            role = conv.get("from", conv.get("role", ""))
            content = conv.get("value", conv.get("content", ""))
            if role in ("human", "user"):
                text += f"User: {content}\n\n"
            elif role in ("gpt", "assistant"):
                text += f"Assistant: {content}\n\n"
            elif role == "system":
                text += f"{content}\n\n"
        return text.strip()

    # Template-based formats
    template = fmt.get("template", "")

    # Check for optional fields
    if "template_no_input" in fmt:
        input_field = fmt.get("input", "input")
        if not example.get(input_field, "").strip():
            template = fmt["template_no_input"]

    if "template_no_context" in fmt:
        context_field = fmt.get("context", "context")
        if not example.get(context_field, "").strip():
            template = fmt["template_no_context"]

    # Build kwargs for formatting
    kwargs = {}
    for key, field in fmt.items():
        if key not in ("template", "template_no_input", "template_no_context"):
            kwargs[key] = example.get(field, "")

    return template.format(**kwargs)


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerFast,
        format_name: str = "alpaca",
        max_length: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        subset: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format = DATASET_FORMATS.get(format_name, DATASET_FORMATS["simple"])

        # Load dataset
        print(f"Loading dataset: {dataset_name}")
        if subset:
            ds = load_dataset(dataset_name, subset, split=split, token=token)
        else:
            ds = load_dataset(dataset_name, split=split, token=token)

        # Limit samples if specified
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))

        self.examples = list(ds)
        print(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = format_example(example, self.format)

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate/pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}


class MixedDataset(Dataset):
    """Mix instruction data with pre-training data."""

    def __init__(
        self,
        instruction_dataset: Dataset,
        pretrain_texts: List[str],
        tokenizer: PreTrainedTokenizerFast,
        mix_ratio: float = 0.2,
        max_length: int = 512,
    ):
        self.instruction_dataset = instruction_dataset
        self.pretrain_texts = pretrain_texts
        self.tokenizer = tokenizer
        self.mix_ratio = mix_ratio
        self.max_length = max_length

        # Calculate effective length
        n_pretrain = int(len(instruction_dataset) * mix_ratio / (1 - mix_ratio))
        self.n_pretrain = min(n_pretrain, len(pretrain_texts))
        self.total_len = len(instruction_dataset) + self.n_pretrain

        print(f"Mixed dataset: {len(instruction_dataset)} instruction + {self.n_pretrain} pretrain")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < len(self.instruction_dataset):
            return self.instruction_dataset[idx]
        else:
            # Pre-training example
            text_idx = (idx - len(self.instruction_dataset)) % len(self.pretrain_texts)
            text = self.pretrain_texts[text_idx]

            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            input_ids = torch.tensor(tokens, dtype=torch.long)
            labels = input_ids.clone()

            return {"input_ids": input_ids, "labels": labels}


def load_pretrain_texts(dataset_name: str, max_samples: int = 10000, token: Optional[str] = None) -> List[str]:
    """Load pre-training texts for mixing."""
    print(f"Loading pre-training data: {dataset_name} (max {max_samples} samples)")

    try:
        ds = load_dataset(dataset_name, split="train", streaming=True, token=token)
        texts = []
        for i, example in enumerate(ds):
            if i >= max_samples:
                break
            text = example.get("text", example.get("content", ""))
            if text:
                texts.append(text)
        print(f"Loaded {len(texts)} pre-training texts")
        return texts
    except Exception as e:
        print(f"Warning: Could not load pre-training data: {e}")
        return []


def collate_fn(batch):
    """Collate function with padding."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    gradient_accumulation: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
) -> int:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        if use_amp and AMP_AVAILABLE:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if global_step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss", loss.item() * gradient_accumulation, global_step)
                writer.add_scalar("train/lr", lr, global_step)

        total_loss += loss.item() * gradient_accumulation
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    return global_step


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Complexity model")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pre-trained model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer (default: same as checkpoint)")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name (e.g., tatsu-lab/alpaca)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset/config")
    parser.add_argument("--format", type=str, default="alpaca",
                        choices=list(DATASET_FORMATS.keys()),
                        help="Dataset format")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token")

    # Pre-training mix
    parser.add_argument("--mix-pretrain", type=float, default=0.0,
                        help="Mix ratio of pre-training data (0-1, default: 0 = disabled)")
    parser.add_argument("--pretrain-dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                        help="Pre-training dataset for mixing")
    parser.add_argument("--pretrain-samples", type=int, default=10000,
                        help="Max pre-training samples to load")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 mixed precision")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")

    # Output
    parser.add_argument("--output", type=str, default="./checkpoints-sft",
                        help="Output directory")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sft_{Path(args.dataset).name}_{timestamp}"
    writer = SummaryWriter(f"runs/{run_name}")

    print("\n" + "=" * 60)
    print("COMPLEXITY MODEL FINE-TUNING")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset} (format: {args.format})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.lr}")
    if args.mix_pretrain > 0:
        print(f"Pre-training mix: {args.mix_pretrain * 100:.0f}%")
    print("=" * 60 + "\n")

    # Load tokenizer
    tokenizer_path = args.tokenizer or args.checkpoint
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    from complexity_deep import DeepForCausalLM

    # Try multiple checkpoint names
    checkpoint_names = ["model.pt", "final.pt", "step_1000000.pt", "checkpoint.pt", "last.pt"]
    checkpoint_path = None
    for name in checkpoint_names:
        path = f"{args.checkpoint}/{name}"
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {args.checkpoint}. Tried: {checkpoint_names}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load model config from config.json (not from checkpoint which has training params)
    import json
    config_json_path = f"{args.checkpoint}/config.json"
    if os.path.exists(config_json_path):
        with open(config_json_path) as f:
            config = json.load(f)
        print(f"Loaded model config from: {config_json_path}")
    else:
        # Fallback to checkpoint config
        config = checkpoint.get("config", {})
        print("Warning: No config.json found, using checkpoint config")

    # Use from_dict() to filter out non-model config keys
    from complexity_deep import DeepConfig
    model_config = DeepConfig.from_dict(config)
    print(f"Model: hidden_size={model_config.hidden_size}, layers={model_config.num_hidden_layers}, vocab={model_config.vocab_size}")
    model = DeepForCausalLM(model_config)

    # Handle different checkpoint formats (model_state_dict or model)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")

    # Load datasets
    instruction_dataset = InstructionDataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        format_name=args.format,
        max_length=args.max_length,
        max_samples=args.max_samples,
        token=args.token,
        subset=args.subset,
    )

    # Mix with pre-training data if requested
    if args.mix_pretrain > 0:
        pretrain_texts = load_pretrain_texts(
            args.pretrain_dataset,
            max_samples=args.pretrain_samples,
            token=args.token,
        )
        if pretrain_texts:
            train_dataset = MixedDataset(
                instruction_dataset=instruction_dataset,
                pretrain_texts=pretrain_texts,
                tokenizer=tokenizer,
                mix_ratio=args.mix_pretrain,
                max_length=args.max_length,
            )
        else:
            train_dataset = instruction_dataset
    else:
        train_dataset = instruction_dataset

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # Warmup (manual)
    def warmup_lr(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0

    # Mixed precision
    scaler = GradScaler() if args.bf16 and AMP_AVAILABLE else None

    # Training loop
    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 40}")

        global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            gradient_accumulation=args.gradient_accumulation,
            use_amp=args.bf16,
        )

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = f"{args.output}/checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "global_step": global_step,
            }, checkpoint_path)
            print(f"Saved: {checkpoint_path}")

    # Save final model
    final_path = f"{args.output}/model_sft.pt"
    torch.save({
        "model": model.state_dict(),
        "config": config,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    # Also save in HuggingFace format
    hf_path = f"{args.output}/hf"
    os.makedirs(hf_path, exist_ok=True)

    # Save config.json
    import json
    with open(f"{hf_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(hf_path)

    print(f"HuggingFace format saved: {hf_path}")

    writer.close()
    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()
