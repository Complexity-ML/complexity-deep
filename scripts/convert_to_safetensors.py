"""Convert PyTorch checkpoint to safetensors format."""

import torch
from safetensors.torch import save_model, save_file
from pathlib import Path
import sys


def convert_checkpoint(checkpoint_path: str, output_path: str = None):
    """Convert .pt checkpoint to .safetensors format."""
    checkpoint_path = Path(checkpoint_path)

    if output_path is None:
        output_path = checkpoint_path.with_suffix('.safetensors')
    else:
        output_path = Path(output_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Step: {checkpoint.get('step', 'unknown')}")
        print(f"  Loss: {checkpoint.get('loss', 'unknown')}")
    elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume it's already a state dict
        state_dict = checkpoint

    # Convert all tensors to contiguous format (required by safetensors)
    clean_state_dict = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            clean_state_dict[key] = tensor.contiguous()

    print(f"  Parameters: {len(clean_state_dict)}")
    print(f"Saving to: {output_path}")

    # Handle shared tensors (tied embeddings) by making separate copies
    if 'lm_head.weight' in clean_state_dict and 'model.embed_tokens.weight' in clean_state_dict:
        # Check if they share memory
        if clean_state_dict['lm_head.weight'].data_ptr() == clean_state_dict['model.embed_tokens.weight'].data_ptr():
            print("  Handling tied embeddings (making separate copy for lm_head)...")
            clean_state_dict['lm_head.weight'] = clean_state_dict['model.embed_tokens.weight'].clone()

    save_file(clean_state_dict, output_path)

    # Verify
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done! Size: {size_mb:.1f} MB")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_safetensors.py <checkpoint.pt> [output.safetensors]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_checkpoint(input_path, output_path)
