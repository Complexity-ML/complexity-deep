"""
Model configuration for the Complexity architecture.

All hyperparameters are collected in a single dataclass.  Preset
configurations provide reproducible model sizes from the paper.
"""

import inspect
from dataclasses import dataclass
from typing import Optional


@dataclass
class ComplexityConfig:
    """
    Configuration for Complexity models.

    The defaults match the 166M-parameter model used in the paper's
    iso-parameter ablations.
    """

    # --- Vocabulary and embeddings ---
    vocab_size: int = 100_000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4       # GQA groups
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    # --- Special tokens ---
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 0

    # --- Token-Routed MLP ---
    use_token_routed_mlp: bool = True
    num_experts: int = 4
    shared_expert: bool = True         # Shared Lexical Expert

    # --- Mu-Guidance ---
    use_mu_guidance: bool = True       # Cross-layer contextual mu

    # --- Attention ---
    use_qk_norm: bool = True           # QK normalization

    # ================================================================
    # Presets
    # ================================================================

    @classmethod
    def complexity_tiny(cls) -> "ComplexityConfig":
        """~15M params (debugging)."""
        return cls(hidden_size=256, intermediate_size=704,
                   num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2)

    @classmethod
    def complexity_20m(cls) -> "ComplexityConfig":
        """~20M params (quick experiments)."""
        return cls(hidden_size=320, intermediate_size=896,
                   num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4)

    @classmethod
    def complexity_small(cls) -> "ComplexityConfig":
        """~50M params."""
        return cls(hidden_size=512, intermediate_size=1408,
                   num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4)

    @classmethod
    def complexity_150m(cls) -> "ComplexityConfig":
        """~166M params (paper's main ablation size)."""
        return cls(hidden_size=768, intermediate_size=2048,
                   num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4)

    @classmethod
    def complexity_350m(cls) -> "ComplexityConfig":
        """~350M params."""
        return cls(hidden_size=1280, intermediate_size=3456,
                   num_hidden_layers=20, num_attention_heads=16, num_key_value_heads=4)

    @classmethod
    def complexity_1b(cls) -> "ComplexityConfig":
        """~1B params."""
        return cls(hidden_size=2048, intermediate_size=5632,
                   num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=8,
                   num_experts=8)

    # ================================================================
    # Serialization
    # ================================================================

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ComplexityConfig":
        """Create from dictionary, ignoring unknown keys."""
        valid = set(inspect.signature(cls).parameters.keys())
        return cls(**{k: v for k, v in d.items() if k in valid})
