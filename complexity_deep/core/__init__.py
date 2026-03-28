"""
Complexity Core Components
==========================

Building blocks for the Complexity architecture.

Each TransformerBlock contains:
    1. GQA Attention with Mu-Guidance (cross-layer top-down signal)
    2. Token-Routed MLP with Shared Lexical Expert (sparse dispatch)
    3. MuGuidance module (produces mu AFTER the MLP)
"""

from .normalization import RMSNorm
from .rotary import RotaryEmbedding, apply_rotary_pos_emb
from .attention import ComplexityAttention
from .mlp import SwiGLU
from .token_routed_mlp import TokenRoutedMLP
from .layer import TransformerBlock, MuGuidance

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "ComplexityAttention",
    "SwiGLU",
    "TokenRoutedMLP",
    "TransformerBlock",
    "MuGuidance",
]
