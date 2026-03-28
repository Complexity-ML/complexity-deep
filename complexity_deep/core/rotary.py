"""
Rotary Position Embedding (RoPE) for the Complexity architecture.

Encodes position information directly into Q/K vectors via rotation,
providing relative position awareness without learned position embeddings.

Reference:
    Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes cos/sin tables and applies rotation to Q and K tensors.
    Supports dynamic extension beyond the initial max_seq_len.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Inverse frequencies: theta^{-2i/d} for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos/sin cache for positions [0, seq_len)."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)         # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables of shape [seq_len, dim]."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to Q and K tensors.

    Args:
        q: [batch, heads, seq, head_dim]
        k: [batch, heads, seq, head_dim]
        cos: [seq, head_dim]
        sin: [seq, head_dim]

    Returns:
        Rotated (q, k) tensors with the same shape.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
