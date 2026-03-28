"""
Utility functions for creating and inspecting Complexity models.
"""

from .config import ComplexityConfig
from .modeling import ComplexityModel


SIZE_PRESETS = {
    "tiny": ComplexityConfig.complexity_tiny,
    "20m": ComplexityConfig.complexity_20m,
    "small": ComplexityConfig.complexity_small,
    "150m": ComplexityConfig.complexity_150m,
    "350m": ComplexityConfig.complexity_350m,
    "1b": ComplexityConfig.complexity_1b,
}


def create_complexity_model(
    size: str = "150m",
    vocab_size: int = 100_000,
    **overrides,
) -> ComplexityModel:
    """
    Create a Complexity model by size name.

    Args:
        size:       one of "tiny", "20m", "small", "150m", "350m", "1b"
        vocab_size: vocabulary size
        **overrides: additional config overrides (e.g., use_mu_guidance=False)

    Returns:
        ComplexityModel ready for training or inference
    """
    if size not in SIZE_PRESETS:
        raise ValueError(f"Unknown size '{size}'. Choose from {list(SIZE_PRESETS.keys())}")

    config = SIZE_PRESETS[size]()
    config.vocab_size = vocab_size
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    return ComplexityModel(config)


def count_parameters(config: ComplexityConfig) -> dict:
    """
    Estimate parameter counts for a given configuration.

    Returns dictionary with component-level and total counts.
    """
    H = config.hidden_size
    I = config.intermediate_size
    V = config.vocab_size
    L = config.num_hidden_layers
    Nq = config.num_attention_heads
    Nkv = config.num_key_value_heads
    D = H // Nq  # head_dim
    E = config.num_experts if config.use_token_routed_mlp else 1
    I_e = I // E  # expert intermediate size

    # Embeddings (tied => counted once)
    embed = V * H

    # Per-layer attention: Q + K + V + O + mu_to_{Q,K,V}
    attn_base = (Nq * D * H) + 2 * (Nkv * D * H) + (H * H)
    attn_mu = (Nq * D * H) + 2 * (Nkv * D * H) if config.use_mu_guidance else 0
    attn_qk_norm = 2 * D if config.use_qk_norm else 0
    attn = attn_base + attn_mu + attn_qk_norm

    # Per-layer MLP
    if config.use_token_routed_mlp:
        routed = E * (H * I_e + H * I_e + I_e * H)  # gate + up + down per expert
        shared = (H * I_e + H * I_e + I_e * H) if config.shared_expert else 0
        mlp = routed + shared
    else:
        mlp = 3 * H * I

    # Norms: 2 per layer + 1 final
    norms = 2 * H

    # Mu-Guidance per layer
    mu = H + H * H if config.use_mu_guidance else 0  # mu param + mu_proj

    layer_total = attn + mlp + norms + mu
    total = embed + L * layer_total + H  # +H for final norm

    return {
        "total": total,
        "embedding": embed,
        "per_layer": layer_total,
        "attention_per_layer": attn,
        "mlp_per_layer": mlp,
        "mu_per_layer": mu,
        "num_layers": L,
        "memory_bf16_gb": total * 2 / (1024**3),
    }
