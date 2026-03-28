"""
vllm-i64 :: Checkpoint Converter

Converts a Pacific-Prime training checkpoint (.pt) to:
  1. HuggingFace-compatible safetensors + config.json
  2. Optionally quantized INT8 version

Usage:
    python scripts/convert_checkpoint.py \
        --input checkpoints/pacific-prime-chat/checkpoint_epoch360.pt \
        --output models/pacific-prime-chat \
        --quantize int8

INL - 2025
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_training_checkpoint(path: str) -> dict:
    """Load a training checkpoint and extract model weights + config."""
    print(f"Loading checkpoint: {path}")
    start = time.perf_counter()

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    elapsed = time.perf_counter() - start
    print(f"  Loaded in {elapsed:.1f}s")

    # Extract components
    state_dict = ckpt.get("model", ckpt)
    config = ckpt.get("config", {})
    chat_template = ckpt.get("chat_template", None)
    epoch = ckpt.get("epoch", None)
    global_step = ckpt.get("global_step", None)

    # Strip "model." prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean_key = k.replace("model.", "", 1) if k.startswith("model.") else k
            cleaned[clean_key] = v

    total_params = sum(v.numel() for v in cleaned.values())
    print(f"  {len(cleaned)} tensors, {total_params:,} parameters")
    if epoch is not None:
        print(f"  Epoch {epoch}, step {global_step}")

    return {
        "state_dict": cleaned,
        "config": config,
        "chat_template": chat_template,
        "epoch": epoch,
        "global_step": global_step,
    }


def save_safetensors(state_dict: dict, output_dir: str, dtype: torch.dtype = torch.float16):
    """Save state dict as safetensors in FP16."""
    from safetensors.torch import save_file

    # Convert to target dtype
    converted = {}
    for k, v in state_dict.items():
        if v.dtype in (torch.float32, torch.float64):
            converted[k] = v.to(dtype)
        else:
            converted[k] = v

    size_mb = sum(v.numel() * v.element_size() for v in converted.values()) / 1024 / 1024
    print(f"  Saving {len(converted)} tensors as safetensors ({size_mb:.0f} MB, {dtype})")

    path = os.path.join(output_dir, "model.safetensors")
    save_file(converted, path)
    print(f"  Saved: {path}")
    return path


def quantize_int8_state_dict(state_dict: dict) -> dict:
    """Quantize all float weights to INT8 with per-channel scales."""
    from vllm_i64.core.quantization import quantize_int8

    quantized = {}
    num_quantized = 0

    for k, v in state_dict.items():
        if v.dtype in (torch.float16, torch.float32, torch.bfloat16) and v.dim() >= 2:
            # Quantize 2D+ weight matrices
            fp32 = v.float()
            q, scale = quantize_int8(fp32)
            quantized[k] = q
            quantized[k + ".scale"] = scale.to(torch.float16)
            num_quantized += 1
        else:
            # Keep 1D tensors (norms, biases) in FP16
            if v.dtype in (torch.float32, torch.float64):
                quantized[k] = v.to(torch.float16)
            else:
                quantized[k] = v

    size_mb = sum(v.numel() * v.element_size() for v in quantized.values()) / 1024 / 1024
    print(f"  Quantized {num_quantized} weight matrices to INT8 ({size_mb:.0f} MB)")
    return quantized


def save_config(config: dict, output_dir: str, chat_template: str = None, quantization: str = None):
    """Save config.json in HuggingFace-compatible format."""
    # Build HF-compatible config
    hf_config = {
        "model_type": config.get("model_type", "complexity-deep"),
        "architecture": config.get("architecture", "DeepForCausalLM"),
        "architectures": ["DeepForCausalLM"],
        "version": config.get("version", "0.13.0"),

        "vocab_size": config.get("vocab_size", 32000),
        "hidden_size": config.get("hidden_size", 2048),
        "intermediate_size": config.get("intermediate_size", 5632),
        "num_hidden_layers": config.get("num_hidden_layers", 24),
        "num_attention_heads": config.get("num_attention_heads", 16),
        "num_key_value_heads": config.get("num_key_value_heads", 8),

        "max_position_embeddings": config.get("max_position_embeddings", 2048),
        "rope_theta": config.get("rope_theta", 10000.0),

        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
        "attention_dropout": config.get("attention_dropout", 0.0),
        "hidden_act": config.get("hidden_act", "silu"),

        "tie_word_embeddings": config.get("tie_word_embeddings", True),
        "initializer_range": config.get("initializer_range", 0.02),

        "pad_token_id": config.get("pad_token_id", 1),
        "bos_token_id": config.get("bos_token_id", 2),
        "eos_token_id": config.get("eos_token_id", 0),

        "use_token_routed_mlp": config.get("use_token_routed_mlp", True),
        "num_experts": config.get("num_experts", 4),

        "use_qk_norm": config.get("use_qk_norm", True),
        "use_sdpa": config.get("use_sdpa", True),

        "shared_expert": config.get("shared_expert", True),
        "use_mu_guidance": config.get("use_mu_guidance", True),

        "torch_dtype": "float16",
    }

    if quantization:
        hf_config["quantization_config"] = {
            "quant_method": quantization,
            "bits": 8 if quantization == "int8" else 4,
        }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"  Saved: {config_path}")

    # Save chat template
    if chat_template:
        template_path = os.path.join(output_dir, "chat_template.jinja")
        with open(template_path, "w") as f:
            f.write(chat_template)
        print(f"  Saved: {template_path}")


def copy_tokenizer(src_dir: str, dst_dir: str):
    """Copy tokenizer files from the training checkpoint directory."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for fname in tokenizer_files:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            import shutil
            dst = os.path.join(dst_dir, fname)
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Convert Pacific-Prime checkpoint to safetensors")
    parser.add_argument("--input", "-i", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--quantize", "-q", default=None, choices=["int8", "int4", None],
                        help="Quantization method")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"],
                        help="Output dtype (default: float16)")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map[args.dtype]

    # Load
    data = load_training_checkpoint(args.input)

    # Create output dirs
    fp16_dir = os.path.join(args.output, "fp16")
    os.makedirs(fp16_dir, exist_ok=True)

    # Save FP16 safetensors
    print("\n=== Saving FP16 model ===")
    save_safetensors(data["state_dict"], fp16_dir, dtype=target_dtype)
    save_config(data["config"], fp16_dir, data["chat_template"])

    # Copy tokenizer
    src_dir = os.path.dirname(args.input)
    copy_tokenizer(src_dir, fp16_dir)

    # Quantize
    if args.quantize:
        quant_dir = os.path.join(args.output, args.quantize)
        os.makedirs(quant_dir, exist_ok=True)

        print(f"\n=== Quantizing to {args.quantize.upper()} ===")
        if args.quantize == "int8":
            q_state = quantize_int8_state_dict(data["state_dict"])
        else:
            print("  INT4 quantization for safetensors export not yet implemented")
            print("  (INT4 is applied at runtime via loader.py)")
            q_state = None

        if q_state:
            from safetensors.torch import save_file
            quant_path = os.path.join(quant_dir, "model.safetensors")
            save_file(q_state, quant_path)
            print(f"  Saved: {quant_path}")

            save_config(data["config"], quant_dir, data["chat_template"], quantization=args.quantize)
            copy_tokenizer(src_dir, quant_dir)

    # Summary
    print("\n=== Done ===")
    print(f"  FP16:  {fp16_dir}/")
    if args.quantize:
        print(f"  {args.quantize.upper()}: {os.path.join(args.output, args.quantize)}/")

    # Size comparison
    for subdir in os.listdir(args.output):
        full = os.path.join(args.output, subdir)
        if os.path.isdir(full):
            safetensors = os.path.join(full, "model.safetensors")
            if os.path.exists(safetensors):
                size_mb = os.path.getsize(safetensors) / 1024 / 1024
                print(f"  {subdir}: {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
