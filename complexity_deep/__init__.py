"""
Complexity Framework — Supplementary Code for TMLR Submission
=============================================================

Reference implementation of the Complexity architecture:
- Token-Routed MLP with Zipf-balanced routing and Shared Lexical Expert
- Mu-Guidance: cross-layer contextual signal biasing K, Q, V in attention
- GPT-style initialization with residual scaling

Usage:
    from complexity import ComplexityConfig, ComplexityModel, create_complexity_model

    # Quick start
    model = create_complexity_model("150m")

    # Custom config
    config = ComplexityConfig(hidden_size=768, num_hidden_layers=12)
    model = ComplexityModel(config)
"""

from .core import (
    RMSNorm,
    RotaryEmbedding,
    ComplexityAttention,
    SwiGLU,
    TokenRoutedMLP,
    TransformerBlock,
    MuGuidance,
)

from .models import (
    ComplexityConfig,
    ComplexityModel,
    create_complexity_model,
    count_parameters,
)

__version__ = "1.0.0"

__all__ = [
    # Core
    "RMSNorm",
    "RotaryEmbedding",
    "ComplexityAttention",
    "SwiGLU",
    "TokenRoutedMLP",
    "TransformerBlock",
    "MuGuidance",
    # Models
    "ComplexityConfig",
    "ComplexityModel",
    "create_complexity_model",
    "count_parameters",
]
