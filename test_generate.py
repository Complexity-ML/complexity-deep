"""
Test generation from checkpoint.

Usage:
    python test_generate.py checkpoints/step_5000.pt
    python test_generate.py checkpoints/step_500000.pt
"""

import argparse
import torch
from transformers import AutoTokenizer
from complexity_deep.models import ComplexityForCausalLM, ComplexityConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("tokenizer/")
    print(f"Vocab size: {len(tokenizer)}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print(f"Step: {ckpt.get('step', 'unknown')}")

    # Create model with 350m config
    config = ComplexityConfig.complexity_350m()
    config.vocab_size = 32000

    model = ComplexityForCausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.cuda()
    # Note: triton sigmoid only supports fp32/fp64, so we stay in fp32 for inference
    # Training uses autocast which handles this properly

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Generate
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").cuda()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated:\n{generated}")

    # Check for spaces
    print(f"\n{'='*60}")
    has_spaces = " " in generated[len(args.prompt):]
    print(f"Has spaces in generation: {has_spaces}")
    if not has_spaces:
        print("WARNING: No spaces detected - model may have been trained with special token pollution!")


if __name__ == "__main__":
    main()
