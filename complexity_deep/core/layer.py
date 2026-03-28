"""
TransformerBlock — single decoder layer for the Complexity architecture.

Architecture (Pre-Norm, three phases per layer):

    1. Attention:   attn_out = Attention(LN(x), mu_prev)
                    x = x + attn_out                         (residual)
    2. MLP:         mlp_out = TokenRoutedMLP(LN(x), token_ids)
                    x = x + mlp_out                          (residual)
    3. Mu:          mu = MuGuidance(x)                       (contextual mu)

mu from layer L is passed as mu_prev to layer L+1's attention, creating a
cross-layer top-down signal (Section 3.3 of the paper).

Note: there is NO PiD controller, NO velocity state, NO INLDynamics.
Mu-Guidance is a simple learnable projection, not a dynamical system.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .normalization import RMSNorm
from .attention import ComplexityAttention
from .mlp import SwiGLU
from .token_routed_mlp import TokenRoutedMLP


class MuGuidance(nn.Module):
    """
    Mu-Guidance — produces a contextual latent vector that flows between layers.

    Equation (Section 3.3):
        mu_contextual = clamp(mu_param) + mu_proj(hidden_states)

    mu_param: learnable per-dimension base value, clamped to [mu_min, mu_max]
    mu_proj:  linear projection (zero-initialized so mu starts neutral)

    This module is placed AFTER the MLP so that mu captures expert-specific
    information (which expert processed each token).
    """

    def __init__(self, hidden_size: int, mu_min: float = 0.0, mu_max: float = 2.0):
        super().__init__()
        self.mu_min = mu_min
        self.mu_max = mu_max

        # Learnable base equilibrium, initialized to midpoint of clamp range
        self.mu = nn.Parameter(torch.full((hidden_size,), (mu_min + mu_max) / 2))

        # Context-dependent projection (zero-init = start neutral)
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S, H] — post-MLP hidden states
        Returns:
            mu_contextual: [B, S, H]
        """
        return torch.clamp(self.mu, self.mu_min, self.mu_max) + self.mu_proj(hidden_states)


class TransformerBlock(nn.Module):
    """
    Single decoder block: Attention -> MLP -> Mu-Guidance.

    Parameters:
        hidden_size:             model dimension
        intermediate_size:       total MLP intermediate width
        num_attention_heads:     Q heads
        num_key_value_heads:     K/V heads (GQA)
        max_position_embeddings: RoPE context length
        rms_norm_eps:            RMSNorm epsilon
        rope_theta:              RoPE base frequency
        attention_dropout:       attention dropout (training only)
        use_token_routed_mlp:    True = Token-Routed MLP, False = dense SwiGLU
        num_experts:             number of routed experts (if token-routed)
        vocab_size:              vocabulary size (for routing table)
        shared_expert:           include shared lexical expert in Token-Routed MLP
        token_frequencies:       corpus frequencies for Zipf-balanced routing
        use_qk_norm:             QK normalization in attention
        use_mu_guidance:         produce and consume mu across layers
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        use_token_routed_mlp: bool = True,
        num_experts: int = 4,
        vocab_size: int = 100_000,
        shared_expert: bool = True,
        token_frequencies: Optional[torch.Tensor] = None,
        use_qk_norm: bool = True,
        use_mu_guidance: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_token_routed_mlp = use_token_routed_mlp

        # --- 1. Pre-norm + Attention ---
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = ComplexityAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            use_qk_norm=use_qk_norm,
        )

        # --- 2. Pre-norm + MLP ---
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        if use_token_routed_mlp:
            self.mlp = TokenRoutedMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                vocab_size=vocab_size,
                shared_expert=shared_expert,
                token_frequencies=token_frequencies,
            )
        else:
            self.mlp = SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)

        # --- 3. Mu-Guidance (AFTER MLP) ---
        self.use_mu_guidance = use_mu_guidance
        if use_mu_guidance:
            self.mu_guidance = MuGuidance(hidden_size=hidden_size)
        else:
            self.mu_guidance = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, S, H]
            attention_mask: optional additive mask
            past_key_value: cached (K, V) for generation
            use_cache:      return updated cache
            token_ids:      [B, S] input token IDs (for routing)
            mu_prev:        [B, S, H] mu from previous layer

        Returns:
            hidden_states:  [B, S, H]
            new_kv:         optional cached (K, V)
            mu_contextual:  [B, S, H] or None
        """
        # === 1. Attention (with mu-guided K, Q, V) ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            mu_prev=mu_prev,
        )
        hidden_states = residual + hidden_states

        # === 2. MLP (Token-Routed or dense SwiGLU) ===
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_token_routed_mlp:
            hidden_states = self.mlp(hidden_states, token_ids=token_ids)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # === 3. Mu-Guidance (AFTER MLP) ===
        mu_contextual = None
        if self.mu_guidance is not None:
            mu_contextual = self.mu_guidance(hidden_states)

        return hidden_states, new_kv, mu_contextual
