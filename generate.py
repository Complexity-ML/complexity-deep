#!/usr/bin/env python3
"""
complexity-deep v0.13.0 - Text Generation Script

Usage:
    python generate.py "Your prompt here"
    python generate.py "Your prompt" --max_tokens 100 --temperature 0.8
    python generate.py --interactive
"""

import argparse
import json
import torch
from pathlib import Path
from safetensors.torch import load_file
from tokenizers import Tokenizer
from complexity_deep import DeepForCausalLM, DeepConfig


def load_model(checkpoint_dir: str = "checkpoints", device: str = None):
    """Load model and tokenizer from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    with open(checkpoint_dir / "config.json", "r") as f:
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
        dynamics_alpha=cfg.get("dynamics_alpha", 0.9),
        dynamics_beta=cfg.get("dynamics_beta", 0.1),
        dynamics_gate=cfg.get("dynamics_gate", 0.5),
        dynamics_dt=cfg.get("dynamics_dt", 0.1),
        dynamics_controller_hidden=cfg.get("dynamics_controller_hidden", 64),
    )

    # Find latest checkpoint - support both .safetensors and .pt
    safetensors_ckpts = list(checkpoint_dir.glob("step_*.safetensors"))
    pt_ckpts = list(checkpoint_dir.glob("checkpoint_epoch*.pt")) + list(checkpoint_dir.glob("step_*.pt"))

    latest = None
    is_pt = False

    if safetensors_ckpts:
        latest = max(safetensors_ckpts, key=lambda p: int(p.stem.split("_")[1]))
    elif pt_ckpts:
        # Sort by epoch/step number
        def get_num(p):
            name = p.stem
            if "epoch" in name:
                return int(name.split("epoch")[1])
            return int(name.split("_")[1])
        latest = max(pt_ckpts, key=get_num)
        is_pt = True
    elif (checkpoint_dir / "model.safetensors").exists():
        latest = checkpoint_dir / "model.safetensors"
    elif (checkpoint_dir / "final.pt").exists():
        latest = checkpoint_dir / "final.pt"
        is_pt = True

    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading model from {latest}")
    print(f"Device: {device}")

    # Load model
    model = DeepForCausalLM(config)

    if is_pt or latest.suffix == ".pt":
        # Load .pt checkpoint (training format)
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        # Strip 'model.' prefix if present
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    else:
        # Load safetensors
        state_dict = load_file(latest)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))

    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")

    return model, tokenizer, config, device


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: str = "cpu",
    eos_token_id: int = 0,
    stream: bool = True,
):
    """Generate text from a prompt."""
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long).to(device)
    generated_ids = input_ids.clone()

    # Track generated tokens for repetition penalty
    generated_set = set(input_ids[0].tolist())

    if stream:
        print(prompt, end="", flush=True)

    for _ in range(max_tokens):
        # Forward pass
        outputs = model(generated_ids)
        next_logits = outputs.logits[0, -1, :].float()

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in generated_set:
                next_logits[token_id] /= repetition_penalty

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float("-inf")

        # Sample
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
        generated_set.add(next_token.item())

        # Stream output
        if stream:
            token_text = tokenizer.decode([next_token.item()])
            print(token_text, end="", flush=True)

        # Stop at EOS
        if next_token.item() == eos_token_id:
            break

    if stream:
        print()  # Newline

    return tokenizer.decode(generated_ids[0].tolist())


def interactive_mode(model, tokenizer, config, device):
    """Interactive chat mode."""
    print("\n" + "=" * 60)
    print("complexity-deep v0.13.0 - Interactive Mode")
    print("=" * 60)
    print("Commands: /quit, /clear, /temp <value>, /tokens <value>")
    print("=" * 60 + "\n")

    temperature = 0.8
    max_tokens = 100

    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not prompt:
            continue

        # Commands
        if prompt == "/quit":
            print("Bye!")
            break
        elif prompt == "/clear":
            print("\033[H\033[J")  # Clear terminal
            continue
        elif prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Usage: /temp <value>")
            continue
        elif prompt.startswith("/tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"Max tokens set to {max_tokens}")
            except:
                print("Usage: /tokens <value>")
            continue

        # Generate
        print()
        generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device,
            eos_token_id=config.eos_token_id if hasattr(config, "eos_token_id") else 0,
            stream=True,
        )
        print()


def main():
    parser = argparse.ArgumentParser(description="complexity-deep text generation")
    parser.add_argument("prompt", nargs="?", default=None, help="Input prompt")
    parser.add_argument("--checkpoint", "-c", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--max_tokens", "-m", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", "-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", "-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition_penalty", "-r", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", "-d", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--no_stream", action="store_true", help="Disable streaming output")

    args = parser.parse_args()

    # Load model
    model, tokenizer, config, device = load_model(args.checkpoint, args.device)

    if args.interactive:
        interactive_mode(model, tokenizer, config, device)
    elif args.prompt:
        generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
            eos_token_id=config.eos_token_id if hasattr(config, "eos_token_id") else 0,
            stream=not args.no_stream,
        )
    else:
        # Default: interactive mode
        interactive_mode(model, tokenizer, config, device)


if __name__ == "__main__":
    main()
