"""
Complexity Model Training Script
=================================

Reference training recipe matching the paper:
- AdamW with betas=(0.9, 0.95), weight_decay=0.1
- Dynamic warmup: 5% of total steps
- Cosine learning-rate schedule decaying to 1% of peak
- GPT-style residual init: 1/sqrt(2*num_layers) scaling
- BF16 mixed precision
- Gradient clipping at 1.0
- Fused cross-entropy (when available) for memory savings

Usage:
    python train_complexity.py --size 150m --dataset roneneldan/TinyStories
    python train_complexity.py --size 150m --data ./pretokenized_data --bf16
"""

import os
import sys
import math
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# Add grandparent so we can import supplementary_code as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from supplementary_code.models.config import ComplexityConfig
from supplementary_code.models.modeling import ComplexityModel
from supplementary_code.models.utils import create_complexity_model

# Mixed precision
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# Optional fused cross-entropy (saves memory by not materializing logits)
try:
    from torch.nn.functional import cross_entropy as fused_ce
    HAS_FUSED_CE = True
except ImportError:
    HAS_FUSED_CE = False


# ======================================================================
# Dataset helpers
# ======================================================================

class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset from HuggingFace.

    Tokenizes on the fly and packs into fixed-length chunks.
    For production training, pre-tokenize with prepare_data.py instead.
    """

    def __init__(self, dataset_name, tokenizer, max_length=512, split="train", token=None):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.token = token

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True, token=self.token)
        buffer = []
        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]
                ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": ids, "labels": labels}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


# ======================================================================
# Learning rate schedule
# ======================================================================

def get_lr_lambda(warmup_steps: int, total_steps: int):
    """
    Dynamic warmup + cosine decay.

    - Linear warmup for warmup_steps (5% of total)
    - Cosine decay from peak to 1% of peak
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


# ======================================================================
# Training loop
# ======================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    max_steps: int = 100_000,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    log_interval: int = 100,
    save_interval: int = 10_000,
    checkpoint_dir: str = "./checkpoints",
):
    """
    Training loop with mixed precision, gradient accumulation, and logging.
    """
    model.train()
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # GradScaler only needed for FP16 (not BF16)
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler("cuda") if use_scaler else None

    global_step = 0
    running_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        if use_amp:
            with autocast("cuda", dtype=amp_dtype):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / grad_accum_steps
        else:
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * grad_accum_steps

        # Optimizer step every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - t0
                ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                lr = scheduler.get_last_lr()[0]
                mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(
                    f"step {global_step:>7d} | loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                    f"lr {lr:.2e} | mem {mem:.1f}GB | {elapsed:.0f}s"
                )
                running_loss = 0.0

            # Checkpointing
            if global_step % save_interval == 0:
                path = ckpt_dir / f"step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, path)
                print(f"Saved checkpoint: {path}")

    return global_step


# ======================================================================
# Main
# ======================================================================

# Optimal hyperparameters by model size (lr scales ~1/sqrt(params))
SIZE_HYPERPARAMS = {
    "tiny":  {"lr": 5e-4},
    "20m":   {"lr": 5e-4},
    "small": {"lr": 3e-4},
    "150m":  {"lr": 1e-4},
    "350m":  {"lr": 8e-5},
    "1b":    {"lr": 3e-5},
}


def main():
    parser = argparse.ArgumentParser(description="Train Complexity model")

    # Model
    parser.add_argument("--size", type=str, default="150m",
                        choices=list(SIZE_HYPERPARAMS.keys()))

    # Data
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer", type=str, default="gpt2")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on size)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 instead of BF16")

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=10_000)

    # Other
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Learning rate
    lr = args.lr or SIZE_HYPERPARAMS[args.size]["lr"]
    warmup_steps = int(0.05 * args.max_steps)  # 5% warmup

    print("=" * 60)
    print("COMPLEXITY MODEL TRAINING")
    print("=" * 60)
    print(f"Size:      {args.size}")
    print(f"Device:    {device}")
    print(f"LR:        {lr}")
    print(f"Warmup:    {warmup_steps} steps (5%)")
    print(f"Max steps: {args.max_steps}")
    print(f"Precision: {'BF16' if args.bf16 else 'FP16' if args.fp16 else 'FP32'}")
    print("=" * 60)

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} (vocab={len(tokenizer)})")

    # Model
    model = create_complexity_model(args.size, vocab_size=len(tokenizer))
    model = model.to(device)
    n_params = model.num_parameters()
    print(f"Parameters: {n_params:,}")

    # Optimizer: AdamW with betas=(0.9, 0.95), weight_decay=0.1
    # Exclude bias, norms, and mu base parameters from weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ("bias", "norm", "mu_init")) or \
           ("mu_guidance.mu" in name and "mu_proj" not in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )
    print(f"Optimizer: {len(decay_params)} params w/ decay, "
          f"{len(no_decay_params)} w/o (bias/norm/mu)")

    # Scheduler: dynamic warmup (5%) + cosine decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(warmup_steps, args.max_steps)
    )

    # Dataset
    print(f"Dataset: {args.dataset} (streaming)")
    dataset = StreamingTextDataset(args.dataset, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    # AMP config
    use_amp = AMP_AVAILABLE and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Train
    print(f"\nStarting training...")
    final_step = train(
        model, loader, optimizer, scheduler, device,
        max_steps=args.max_steps,
        grad_accum_steps=args.gradient_accumulation,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Save final
    model.save_pretrained(Path(args.checkpoint_dir) / "final")
    print(f"\nTraining complete at step {final_step}.")


if __name__ == "__main__":
    main()
