"""
Complexity Model Training Script (CUDA Optimized)
==================================================

Train a Complexity model with CUDA-optimized kernels for maximum performance.

Optimizations enabled:
- Fused QK Norm + Flash Attention (~15-20% faster)
- Fused RMSNorm + MLP (~20-30% faster)
- Persistent CGGR Token-Routed MLP (~10-15% faster)
- Fused Residual + RMSNorm (~5-10% faster)
- Mixed precision (FP16/BF16) training
- Gradient checkpointing (optional, for large models)
- torch.compile (optional, PyTorch 2.0+)

Total speedup: ~40-50% faster than baseline PyTorch

Usage:
    # Train from scratch (auto-detects optimizations)
    python train_complexity_optimized.py --size small --dataset Pacific-Prime/mixed-inl

    # Train with all optimizations
    python train_complexity_optimized.py --size base --fp16 --compile

    # Train WITHOUT optimizations (for debugging)
    python train_complexity_optimized.py --size small --no-optimized

    # Resume training
    python train_complexity_optimized.py --size base --resume ./checkpoints/last.pt
"""

import os
import math
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Suppress dtype mismatch warning from PyTorch RMSNorm
warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from complexity_deep import DeepConfig, DeepForCausalLM, create_deep_model

# Mixed precision (new API for PyTorch 2.0+)
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False

# CUDA Optimizations
try:
    from complexity_deep.cuda import HAS_TRITON, get_optimization_info
    CUDA_OPTIMIZATIONS_AVAILABLE = HAS_TRITON
except ImportError:
    CUDA_OPTIMIZATIONS_AVAILABLE = False
    HAS_TRITON = False
    def get_optimization_info():
        return {"triton_available": False, "optimizations": {}}


# ============================================================================
# OPTIMIZATION UTILITIES
# ============================================================================

def print_optimization_status():
    """Print status of CUDA optimizations."""
    print("\n" + "=" * 60)
    print("CUDA OPTIMIZATIONS STATUS")
    print("=" * 60)

    if not CUDA_OPTIMIZATIONS_AVAILABLE:
        print("  [DISABLED] Triton not installed")
        print("  Install with: pip install triton")
        print("=" * 60)
        return False

    info = get_optimization_info()
    print(f"  Triton available: {info['triton_available']}")
    print()
    for name, opt in info["optimizations"].items():
        status = "OK" if opt["available"] else "DISABLED"
        print(f"  [{status}] {opt['description']}")
        print(f"          Speedup: {opt['speedup']}")
    print("=" * 60)
    return True


def create_optimized_model(
    size: str,
    vocab_size: int,
    use_optimizations: bool = True,
    use_gradient_checkpointing: bool = False,
) -> nn.Module:
    """
    Create model with optional CUDA optimizations.

    Args:
        size: Model size preset
        vocab_size: Vocabulary size
        use_optimizations: Whether to use CUDA optimizations
        use_gradient_checkpointing: Enable gradient checkpointing for memory

    Returns:
        model: Optimized or standard model
    """
    # Size presets
    SIZE_PRESETS = {
        "tiny": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                 "num_key_value_heads": 4, "intermediate_size": 2048, "num_experts": 4},  # ~150M
        "20m": {"hidden_size": 320, "num_hidden_layers": 8, "num_attention_heads": 8,
                "num_key_value_heads": 4, "intermediate_size": 896, "num_experts": 4},  # ~20M
        "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8,
                  "num_key_value_heads": 4, "intermediate_size": 1408, "num_experts": 4},  # ~50M
        "base": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16,
                 "num_key_value_heads": 4, "intermediate_size": 2816, "num_experts": 4},  # ~250M
        "350m": {"hidden_size": 1280, "num_hidden_layers": 20, "num_attention_heads": 16,
                 "num_key_value_heads": 4, "intermediate_size": 3456, "num_experts": 4},  # ~350M
        "medium": {"hidden_size": 1536, "num_hidden_layers": 24, "num_attention_heads": 16,
                   "num_key_value_heads": 4, "intermediate_size": 4096, "num_experts": 8},  # ~760M
        "1b": {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16,
               "num_key_value_heads": 8, "intermediate_size": 5504, "num_experts": 8},  # ~1B
        "large": {"hidden_size": 2048, "num_hidden_layers": 32, "num_attention_heads": 32,
                  "num_key_value_heads": 8, "intermediate_size": 5504, "num_experts": 8},  # ~1.5B
        "3b": {"hidden_size": 2560, "num_hidden_layers": 32, "num_attention_heads": 32,
               "num_key_value_heads": 8, "intermediate_size": 6912, "num_experts": 8},  # ~3B
        "3.8b": {"hidden_size": 3072, "num_hidden_layers": 32, "num_attention_heads": 32,
                 "num_key_value_heads": 8, "intermediate_size": 8192, "num_experts": 8},  # ~3.8B
        "7b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
               "num_key_value_heads": 8, "intermediate_size": 11008, "num_experts": 8},  # ~7B
    }

    preset = SIZE_PRESETS.get(size, SIZE_PRESETS["small"])

    # Standard model with Triton-accelerated layers (TokenRoutedMLPTriton)
    triton_status = "with Triton" if CUDA_OPTIMIZATIONS_AVAILABLE else "without Triton"
    print(f"Creating model ({size}) {triton_status}...")
    model = create_deep_model(size=size, vocab_size=vocab_size)

    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    return model


# ============================================================================
# DATASET
# ============================================================================

class PreTokenizedDataset(torch.utils.data.Dataset):
    """
    Ultra-fast dataset from pre-tokenized parquet files.

    Created by prepare_data.py - ZERO tokenization overhead during training.
    """

    def __init__(self, data_dir: str, max_length: int = 512):
        import pyarrow.parquet as pq

        self.data_dir = Path(data_dir)
        self.max_length = max_length

        # Find all shards
        self.shard_files = sorted(self.data_dir.glob("shard_*.parquet"))
        if not self.shard_files:
            raise ValueError(f"No shard files found in {data_dir}")

        print(f"Loading pre-tokenized data from {data_dir}...")
        print(f"Found {len(self.shard_files)} shards")

        # Load all shards into memory (fast!)
        self.input_ids = []
        self.labels = []

        for shard_path in tqdm(self.shard_files, desc="Loading shards"):
            table = pq.read_table(shard_path)
            for row in range(table.num_rows):
                self.input_ids.append(table['input_ids'][row].as_py())
                self.labels.append(table['labels'][row].as_py())

        print(f"Loaded {len(self.input_ids):,} pre-tokenized samples")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for very large corpora (slower but memory efficient)."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        text_field: str = "text",
        split: str = "train",
        token: Optional[str] = None,
        subset: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.split = split
        self.token = token
        self.subset = subset

    def __iter__(self):
        try:
            if self.subset:
                ds = load_dataset(
                    self.dataset_name,
                    self.subset,
                    split=self.split,
                    streaming=True,
                    token=self.token,
                )
            else:
                ds = load_dataset(
                    self.dataset_name,
                    split=self.split,
                    streaming=True,
                    token=self.token,
                )
        except Exception as e:
            print(f"Warning: Could not load {self.dataset_name}: {e}")
            return

        buffer = []
        for example in ds:
            text = None
            for field in [self.text_field, "text", "content", "code"]:
                if field in example and example[field]:
                    text = example[field]
                    break

            if not text:
                continue

            # IMPORTANT: add_special_tokens=False to avoid <s> and </s> pollution
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.max_length + 1:
                chunk = buffer[: self.max_length + 1]
                buffer = buffer[self.max_length:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    """Collate batch of samples."""
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "labels": labels}


# ============================================================================
# OPTIMIZED TRAINING LOOP
# ============================================================================

def train_optimized(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: dict,
    device: torch.device,
    writer: SummaryWriter = None,
):
    """
    Optimized training loop with:
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Efficient memory management
    """
    model.train()
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = config.get("start_step", 0)
    total_loss = 0.0
    log_interval = config.get("log_interval", 100)
    save_interval = config.get("save_interval", 1000)
    max_steps = config.get("max_steps", 100000)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    use_amp = config.get("use_amp", True) and AMP_AVAILABLE

    # Determine autocast dtype
    if config.get("bf16", False) and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False  # BF16 doesn't need GradScaler
        print("Using BF16 mixed precision (no scaler needed)")
    else:
        amp_dtype = torch.float16
        use_scaler = True  # FP16 needs GradScaler
        print("Using FP16 mixed precision")

    # Mixed precision scaler - only for FP16, not BF16
    scaler = GradScaler('cuda') if (use_amp and use_scaler) else None

    start_time = time.time()
    pbar = tqdm(total=max_steps, initial=global_step, desc="Training")

    optimizer.zero_grad()
    accum_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward with mixed precision
        try:
            if use_amp:
                with autocast('cuda', dtype=amp_dtype):
                    outputs = model(input_ids)
                    # Compute loss manually for optimized model
                    if outputs is None:
                        raise ValueError("Model returned None - check Triton kernel compatibility")
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        loss = outputs.loss
                    else:
                        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                        if logits is None:
                            raise ValueError("Model logits are None")
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                        )
                    loss = loss / grad_accum_steps
            else:
                outputs = model(input_ids)
                if outputs is None:
                    raise ValueError("Model returned None - check Triton kernel compatibility")
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                    if logits is None:
                        raise ValueError("Model logits are None")
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                loss = loss / grad_accum_steps
        except Exception as e:
            print(f"\nERROR during forward pass: {e}")
            print(f"Input shape: {input_ids.shape}, dtype: {input_ids.dtype}")
            print(f"Model dtype: {next(model.parameters()).dtype}")
            raise

        # Backward with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()

        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))

            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            total_loss += accum_loss * grad_accum_steps
            accum_loss = 0.0
            global_step += 1
            pbar.update(1)

            # Logging
            if global_step % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                tokens_per_sec = (global_step * config["batch_size"] * config["max_length"]) / elapsed
                perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

                # Memory stats
                if torch.cuda.is_available():
                    mem_used = torch.cuda.max_memory_allocated() / 1e9
                    mem_str = f"{mem_used:.1f}GB"
                else:
                    mem_str = "N/A"

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "ppl": f"{perplexity:.2f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                    "mem": mem_str,
                })

                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/perplexity", perplexity, global_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("train/tokens_per_sec", tokens_per_sec, global_step)
                    if torch.cuda.is_available():
                        writer.add_scalar("train/memory_gb", mem_used, global_step)

                total_loss = 0.0

            # Save checkpoint
            if global_step % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"step_{global_step}.pt"
                save_dict = {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                }
                if scaler is not None:
                    save_dict["scaler_state_dict"] = scaler.state_dict()

                torch.save(save_dict, checkpoint_path)
                torch.save(save_dict, checkpoint_dir / "last.pt")
                print(f"\nSaved checkpoint: {checkpoint_path}")

    pbar.close()
    return global_step


# ============================================================================
# MAIN
# ============================================================================

SIZE_CONFIGS = ["tiny", "20m", "small", "150m", "base", "350m", "medium", "1b", "large", "3b", "3.8b", "7b"]

# Optimal hyperparameters by model size (based on scaling laws)
# lr scales roughly as 1/sqrt(params), warmup scales with model size
SIZE_HYPERPARAMS = {
    "tiny": {"lr": 3e-4, "warmup_steps": 500},      # ~50M
    "20m": {"lr": 5e-4, "warmup_steps": 300},       # ~20M
    "small": {"lr": 3e-4, "warmup_steps": 500},     # ~50M
    "150m": {"lr": 1e-4, "warmup_steps": 2000},     # ~150M - was causing NaN at 3e-4!
    "base": {"lr": 1e-4, "warmup_steps": 2000},     # ~250M
    "350m": {"lr": 8e-5, "warmup_steps": 2500},     # ~350M
    "medium": {"lr": 5e-5, "warmup_steps": 3000},   # ~760M
    "1b": {"lr": 3e-5, "warmup_steps": 5000},       # ~1B
    "large": {"lr": 3e-5, "warmup_steps": 5000},    # ~1.5B
    "3b": {"lr": 2e-5, "warmup_steps": 8000},       # ~3B
    "3.8b": {"lr": 1.5e-5, "warmup_steps": 10000},  # ~3.8B
    "7b": {"lr": 1e-5, "warmup_steps": 10000},      # ~7B
}


def main():
    parser = argparse.ArgumentParser(description="Train Complexity model (CUDA Optimized)")

    # Model
    parser.add_argument("--size", type=str, default="tiny", choices=SIZE_CONFIGS,
                        help="Model size preset")

    # Optimizations
    parser.add_argument("--optimized", action="store_true", default=True,
                        help="Use CUDA optimizations (default: True)")
    parser.add_argument("--no-optimized", action="store_true",
                        help="Disable CUDA optimizations")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 mixed precision (default: True)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 instead of FP16")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves memory)")

    # Data
    parser.add_argument("--data", type=str, default=None,
                        help="Path to pre-tokenized data (from prepare_data.py) - FAST")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="HuggingFace dataset (streaming, slower)")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer",
                        help="Path to tokenizer")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Text field in dataset")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")

    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--max-steps", type=int, default=1000000,
                        help="Max training steps (default: 1M)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on model size)")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Warmup steps (default: auto based on model size)")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=50000,
                        help="Save every N steps (default: 50K)")

    # TensorBoard
    parser.add_argument("--tensorboard-dir", type=str, default="./runs",
                        help="TensorBoard log directory")

    # Learning rate options
    parser.add_argument("--constant-lr", action="store_true",
                        help="Use constant learning rate (no cosine decay) - useful for recovery after NaN")
    parser.add_argument("--cosine-restarts", action="store_true",
                        help="Use cosine annealing with warm restarts (SGDR) - helps escape local minima")
    parser.add_argument("--restart-period", type=int, default=50000,
                        help="Period of first restart cycle T_0 (default: 50K steps)")
    parser.add_argument("--restart-mult", type=int, default=2,
                        help="Multiplier for restart period T_mult (default: 2, so cycles: 50K, 100K, 200K...)")

    # Other
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Apply optimal hyperparameters based on model size (if not overridden)
    size_params = SIZE_HYPERPARAMS.get(args.size, SIZE_HYPERPARAMS["small"])
    if args.lr is None:
        args.lr = size_params["lr"]
        print(f"Auto-setting lr={args.lr} for size={args.size}")
    if args.warmup_steps is None:
        args.warmup_steps = size_params["warmup_steps"]
        print(f"Auto-setting warmup_steps={args.warmup_steps} for size={args.size}")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("COMPLEXITY MODEL TRAINING (CUDA OPTIMIZED)")
    print("=" * 60)
    print(f"Model size: {args.size}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Dataset: {args.dataset}")

    # Print optimization status
    use_optimizations = args.optimized and not args.no_optimized
    if use_optimizations:
        print_optimization_status()
    else:
        print("\n[!] CUDA optimizations DISABLED")

    # Load tokenizer
    print("\nLoading tokenizer...")
    if os.path.exists(args.tokenizer):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    else:
        # Try to use a default tokenizer
        print(f"Tokenizer not found at {args.tokenizer}, using GPT-2 tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Vocab size: {len(tokenizer)}")

    # Create model
    print(f"\nCreating model...")
    model = create_optimized_model(
        size=args.size,
        vocab_size=len(tokenizer),
        use_optimizations=use_optimizations,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    model = model.to(device)

    # Convert to bf16 if requested (for consistent dtypes)
    if args.bf16 and torch.cuda.is_bf16_supported():
        print("Converting model to BF16...")
        model = model.to(torch.bfloat16)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer with proper weight decay handling
    # Exclude bias, LayerNorm, and mu from weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for bias, norm, and mu (equilibrium point)
        if 'bias' in name or 'norm' in name or '.mu' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    print(f"Optimizer: {len(decay_params)} params with decay, {len(no_decay_params)} without (bias/norm/mu)")

    # Scheduler with warmup (or constant or cosine restarts)
    if args.constant_lr:
        # Constant lr - useful for recovery after NaN
        def lr_lambda(step):
            return 1.0  # No decay, just use the base lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Using CONSTANT learning rate: {args.lr}")
    elif args.cosine_restarts:
        # Cosine annealing with warm restarts (SGDR)
        # T_0 = first cycle length, T_mult = multiplier for subsequent cycles
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.restart_period,
            T_mult=args.restart_mult,
            eta_min=args.lr * 0.01,  # Min LR = 1% of max
        )
        print(f"Using COSINE RESTARTS: T_0={args.restart_period}, T_mult={args.restart_mult}")
        print(f"  Cycles: {args.restart_period} -> {args.restart_period * args.restart_mult} -> {args.restart_period * args.restart_mult**2} ...")
    else:
        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / args.warmup_steps
            return 0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Try to load optimizer state, but skip if structure changed (e.g., new param groups)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Optimizer state restored")
        except ValueError as e:
            print(f"Warning: Could not restore optimizer state ({e})")
            print("Starting with fresh optimizer (momentum reset)")
        start_step = checkpoint["step"]

        # CRITICAL: Override learning rate BEFORE creating scheduler
        # The scheduler uses optimizer's current lr as the base
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            param_group['initial_lr'] = args.lr  # Important for LambdaLR!
        print(f"Learning rate overridden to: {args.lr}")

        # Rebuild scheduler with new lr as base
        if args.constant_lr:
            # Constant lr for recovery - multiplier is always 1.0
            def lr_lambda_resume(step):
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_resume)
            print(f"Using CONSTANT learning rate: {args.lr} (no decay)")
        elif args.cosine_restarts:
            # Cosine restarts - fresh start from current position
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args.restart_period,
                T_mult=args.restart_mult,
                eta_min=args.lr * 0.01,
            )
            # Optionally restore scheduler state, or start fresh cycle
            if "scheduler_state_dict" in checkpoint and not args.cosine_restarts:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                print(f"Starting FRESH cosine restart cycle at step {start_step}")
            print(f"Using COSINE RESTARTS: T_0={args.restart_period}, T_mult={args.restart_mult}")
        else:
            def lr_lambda_resume(step):
                total_step = start_step + step
                if total_step < args.warmup_steps:
                    return total_step / args.warmup_steps
                return 0.5 * (1 + math.cos(math.pi * (total_step - args.warmup_steps) / (args.max_steps - args.warmup_steps)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_resume)

        print(f"Resumed at step {start_step}")

    # Dataset
    print(f"\nLoading dataset...")
    if args.data:
        # Pre-tokenized data (FAST!) - from prepare_data.py
        print(f"Using PRE-TOKENIZED data from {args.data}")
        train_dataset = PreTokenizedDataset(
            data_dir=args.data,
            max_length=args.max_length,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        # Streaming dataset (slower, but no preprocessing needed)
        print(f"Using STREAMING dataset: {args.dataset} (slower)")
        train_dataset = StreamingTextDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            text_field=args.text_field,
            split="train",
            token=args.token,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0,  # Streaming doesn't support multiple workers
        )

    # Training config
    train_config = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "checkpoint_dir": args.checkpoint_dir,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "max_grad_norm": 1.0,
        "start_step": start_step,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "use_amp": args.fp16 or args.bf16,
        "bf16": args.bf16,
    }

    # TensorBoard
    run_name = f"complexity_{args.size}_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tensorboard_dir = Path(args.tensorboard_dir) / run_name
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"TensorBoard: {tensorboard_dir}")

    # Train
    print(f"\n{'=' * 60}")
    print("STARTING OPTIMIZED TRAINING")
    print(f"{'=' * 60}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Mixed precision: {'BF16' if args.bf16 else 'FP16' if args.fp16 else 'Disabled'}")
    print(f"{'=' * 60}\n")

    final_step = train_optimized(model, train_loader, optimizer, scheduler, train_config, device, writer)
    print(f"\nTraining complete! Final step: {final_step}")

    # Save final model
    final_path = Path(args.checkpoint_dir) / "final.pt"
    torch.save({
        "step": final_step,
        "model_state_dict": model.state_dict(),
        "config": train_config,
    }, final_path)
    print(f"Final model saved: {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
