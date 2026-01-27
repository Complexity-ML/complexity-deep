#!/usr/bin/env python3
"""
Train a specialist model using YAML config.

Usage:
    # Train single specialist
    python train_specialist.py --config configs/sft/node_0_code.yaml

    # Train all specialists sequentially
    python train_specialist.py --all

    # Train specific nodes
    python train_specialist.py --nodes 0,1,2
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_from_config(config_path: str, dry_run: bool = False):
    """Run training from config file."""
    config = load_config(config_path)

    # Build command
    checkpoint_path = config["model"]["checkpoint"]
    # Tokenizer is in the same directory as checkpoint
    tokenizer_path = str(Path(checkpoint_path).parent)

    cmd = [
        "python", "conversational_sft.py",
        "--checkpoint", checkpoint_path,
        "--tokenizer", tokenizer_path,
        "--output", config["model"]["output"],
        "--epochs", str(config["training"]["epochs"]),
        "--batch-size", str(config["training"]["batch_size"]),
        "--gradient-accumulation", str(config["training"]["gradient_accumulation"]),
        "--lr", str(config["training"]["learning_rate"]),
        "--max-length", str(config["training"]["max_length"]),
        "--warmup-ratio", str(config["training"]["warmup_ratio"]),
    ]

    # Add flags
    if config["training"].get("gradient_checkpointing", False):
        cmd.append("--gradient-checkpointing")

    if config["training"].get("bf16", True):
        cmd.append("--bf16")

    # Dataset (use first one for now, or combine)
    datasets = config["data"]["datasets"]
    if datasets:
        cmd.extend(["--dataset", datasets[0]["name"]])
        if "subset" in datasets[0]:
            cmd.extend(["--subset", datasets[0]["subset"]])

    # Max samples
    if config["data"].get("max_samples"):
        cmd.extend(["--max-samples", str(config["data"]["max_samples"])])

    # Format
    if config["data"].get("format"):
        cmd.extend(["--format", config["data"]["format"]])

    print(f"\n{'='*60}")
    print(f"Training: {config['specialization']['name'].upper()}")
    print(f"Config: {config_path}")
    print(f"Output: {config['model']['output']}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return 0

    # Run
    result = subprocess.run(cmd)
    return result.returncode


def find_configs(configs_dir: str = "configs/sft") -> list:
    """Find all config files."""
    config_dir = Path(configs_dir)
    if not config_dir.exists():
        print(f"Config directory not found: {configs_dir}")
        return []

    configs = sorted(config_dir.glob("node_*.yaml"))
    return [str(c) for c in configs]


def main():
    parser = argparse.ArgumentParser(description="Train specialist models")
    parser.add_argument("--config", "-c", type=str, help="Single config file")
    parser.add_argument("--all", action="store_true", help="Train all specialists")
    parser.add_argument("--nodes", type=str, help="Comma-separated node IDs (e.g., 0,1,2)")
    parser.add_argument("--configs-dir", type=str, default="configs/sft", help="Configs directory")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without running")

    args = parser.parse_args()

    if args.config:
        # Single config
        return train_from_config(args.config, args.dry_run)

    # Find configs
    all_configs = find_configs(args.configs_dir)

    if not all_configs:
        print("No configs found!")
        return 1

    print(f"Found {len(all_configs)} configs:")
    for i, c in enumerate(all_configs):
        print(f"  {i}: {c}")
    print()

    # Filter by nodes
    if args.nodes:
        node_ids = [int(n.strip()) for n in args.nodes.split(",")]
        configs = [c for c in all_configs if any(f"node_{n}_" in c for n in node_ids)]
    elif args.all:
        configs = all_configs
    else:
        print("Specify --config, --all, or --nodes")
        return 1

    print(f"Training {len(configs)} specialists...")

    # Train each
    for config in configs:
        returncode = train_from_config(config, args.dry_run)
        if returncode != 0 and not args.dry_run:
            print(f"Training failed for {config}")
            return returncode

    print(f"\n{'='*60}")
    print("All training complete!")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
