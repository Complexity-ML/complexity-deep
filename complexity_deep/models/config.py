"""
Configuration for Complexity models.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComplexityConfig:
    """
    Configuration class for Complexity models.

    Attributes:
        vocab_size: Size of vocabulary (default: 100K for Complexity-4 tokenizer)
        hidden_size: Dimension of hidden states
        intermediate_size: Dimension of MLP intermediate layer
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads (for GQA)
        max_position_embeddings: Maximum sequence length
        rms_norm_eps: Epsilon for RMSNorm
        rope_theta: Base frequency for RoPE
        tie_word_embeddings: Whether to tie input/output embeddings
        attention_dropout: Dropout rate for attention
        hidden_act: Activation function for MLP
    """
    # Model architecture
    vocab_size: int = 100000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4

    # Position embeddings
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0

    # Normalization
    rms_norm_eps: float = 1e-6

    # Regularization
    attention_dropout: float = 0.0

    # Activation
    hidden_act: str = "silu"

    # Embedding tying
    tie_word_embeddings: bool = True

    # Special tokens
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 0

    # Initializer
    initializer_range: float = 0.02

    # Token-Routed MLP (Complexity innovation)
    use_token_routed_mlp: bool = True
    num_experts: int = 4

    # 2024 Innovations
    use_qk_norm: bool = True       # QK Normalization - stabilizes training
    use_sdpa: bool = True          # Flash Attention via SDPA
    sliding_window: int = None     # Sliding Window Attention (None = full)

    # INL Dynamics (Full robotics control)
    dynamics_alpha: float = 0.9           # Inertia (momentum, smooth movements)
    dynamics_beta: float = 0.1            # Correction strength (feedback)
    dynamics_gate: float = 0.5            # Amplitude control
    dynamics_dt: float = 0.1              # Integration timestep
    dynamics_controller_hidden: int = 64  # Controller MLP hidden size

    # ========================================================================
    # PRESET CONFIGURATIONS
    # ========================================================================

    @classmethod
    def complexity_tiny(cls) -> "ComplexityConfig":
        """~15M params - for debugging."""
        return cls(
            hidden_size=256,
            intermediate_size=704,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

    @classmethod
    def complexity_150m(cls) -> "ComplexityConfig":
        """~150M params - good for testing."""
        return cls(
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
        )

    @classmethod
    def complexity_20m(cls) -> "ComplexityConfig":
        """~20M params - for quick experiments."""
        return cls(
            hidden_size=320,
            intermediate_size=896,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

    @classmethod
    def complexity_small(cls) -> "ComplexityConfig":
        """~50M params - for testing."""
        return cls(
            hidden_size=512,
            intermediate_size=1408,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
        )

    @classmethod
    def complexity_base(cls) -> "ComplexityConfig":
        """~125M params - baseline."""
        return cls(
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
        )

    @classmethod
    def complexity_medium(cls) -> "ComplexityConfig":
        """~350M params."""
        return cls(
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
        )

    @classmethod
    def complexity_large(cls) -> "ComplexityConfig":
        """~760M params."""
        return cls(
            hidden_size=1536,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
        )

    @classmethod
    def complexity_1b(cls) -> "ComplexityConfig":
        """~1B params."""
        return cls(
            hidden_size=2048,
            intermediate_size=5632,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=8,
        )

    @classmethod
    def complexity_3b(cls) -> "ComplexityConfig":
        """~3B params."""
        return cls(
            hidden_size=2560,
            intermediate_size=6912,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=8,
        )

    @classmethod
    def complexity_3_8b(cls) -> "ComplexityConfig":
        """~3.8B params."""
        return cls(
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=8,
        )

    @classmethod
    def complexity_7b(cls) -> "ComplexityConfig":
        """~7B params."""
        return cls(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=8,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_dropout": self.attention_dropout,
            "hidden_act": self.hidden_act,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "initializer_range": self.initializer_range,
            "use_token_routed_mlp": self.use_token_routed_mlp,
            "num_experts": self.num_experts,
            "use_qk_norm": self.use_qk_norm,
            "use_sdpa": self.use_sdpa,
            "sliding_window": self.sliding_window,
            "dynamics_alpha": self.dynamics_alpha,
            "dynamics_beta": self.dynamics_beta,
            "dynamics_gate": self.dynamics_gate,
            "dynamics_dt": self.dynamics_dt,
            "dynamics_controller_hidden": self.dynamics_controller_hidden,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ComplexityConfig":
        """Create config from dictionary, filtering invalid keys."""
        import inspect
        valid_params = set(inspect.signature(cls.__init__).parameters.keys())
        valid_params.discard('self')
        filtered = {k: v for k, v in config_dict.items() if k in valid_params}
        return cls(**filtered)
