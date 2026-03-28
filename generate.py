#!/usr/bin/env python3
"""
Text generation with a trained Complexity model.

Usage:
    python generate.py "Your prompt here"
    python generate.py "Your prompt" --max_tokens 200 --temperature 0.7
    python generate.py --interactive --checkpoint ./checkpoints/final
"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from supplementary_code.models.modeling import ComplexityModel


def load_model(checkpoint_dir: str, device: str = None):
    """Load model and tokenizer from a checkpoint directory."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ComplexityModel.from_pretrained(checkpoint_dir, device=device)
    model.eval()

    # Try loading tokenizer (HF tokenizer or custom)
    try:
        from tokenizers import Tokenizer
        tok_path = Path(checkpoint_dir) / "tokenizer.json"
        if tok_path.exists():
            tokenizer = Tokenizer.from_file(str(tok_path))
            print(f"Loaded tokenizer from {tok_path}")
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except Exception:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Using default GPT-2 tokenizer")

    n = model.num_parameters()
    print(f"Model: {n:,} parameters on {device}")
    return model, tokenizer, device


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
    stream: bool = True,
):
    """Generate text from a prompt."""
    # Encode
    if hasattr(tokenizer, "encode"):
        if hasattr(tokenizer, "ids"):
            # tokenizers.Tokenizer
            ids = tokenizer.encode(prompt).ids
        else:
            # HF tokenizer
            ids = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        ids = tokenizer(prompt)["input_ids"]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    if stream:
        print(prompt, end="", flush=True)

    # Generate using model's built-in generate()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
    )

    # Decode new tokens
    new_ids = output_ids[0, len(ids):].tolist()
    if hasattr(tokenizer, "decode"):
        text = tokenizer.decode(new_ids)
    else:
        text = tokenizer.decode(new_ids)

    if stream:
        print(text)
    return prompt + text


def interactive_mode(model, tokenizer, device):
    """Interactive generation loop."""
    print("\nComplexity Model — Interactive Mode")
    print("Type /quit to exit, /temp <val> to set temperature\n")

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
        if prompt == "/quit":
            break
        if prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature = {temperature}")
            except ValueError:
                print("Usage: /temp <float>")
            continue
        if prompt.startswith("/tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"Max tokens = {max_tokens}")
            except ValueError:
                print("Usage: /tokens <int>")
            continue

        print()
        generate_text(model, tokenizer, prompt,
                      max_tokens=max_tokens, temperature=temperature,
                      device=device, stream=True)
        print()


def main():
    parser = argparse.ArgumentParser(description="Complexity text generation")
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--checkpoint", "-c", default="./checkpoints/final")
    parser.add_argument("--max_tokens", "-m", type=int, default=100)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--top_k", "-k", type=int, default=50)
    parser.add_argument("--top_p", "-p", type=float, default=0.9)
    parser.add_argument("--device", "-d", default=None)
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.checkpoint, args.device)

    if args.interactive or args.prompt is None:
        interactive_mode(model, tokenizer, device)
    else:
        generate_text(model, tokenizer, args.prompt,
                      max_tokens=args.max_tokens, temperature=args.temperature,
                      top_k=args.top_k, top_p=args.top_p,
                      device=device, stream=True)


if __name__ == "__main__":
    main()
