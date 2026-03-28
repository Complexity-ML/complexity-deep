"""
Grouped Query Attention (GQA) with Mu-Guidance for the Complexity architecture.

Standard GQA (Ainslie et al., 2023) with two Complexity innovations:
1. Mu-Guidance: a contextual signal mu from the previous layer biases K, Q,
   and V projections, creating a top-down information flow across layers.
       K = x W_K + mu_prev W_muK
       Q = x W_Q + mu_prev W_muQ
       V = x W_V + mu_prev W_muV
2. QK-Normalization (Dehghani et al., 2023): RMSNorm on Q and K before the
   dot product, stabilizing training at scale.

Also integrates:
- Rotary Position Embeddings (RoPE)
- Flash Attention via PyTorch SDPA (PyTorch >= 2.0)
- KV-cache for autoregressive generation

Reference: Section 3.3 of the paper (Mu-Guidance equations).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rotary import RotaryEmbedding, apply_rotary_pos_emb

# Check for SDPA (PyTorch 2.0+)
HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


class ComplexityAttention(nn.Module):
    """
    GQA attention with Mu-Guidance.

    Parameters:
        hidden_size:             model dimension (H)
        num_attention_heads:     number of Q heads
        num_key_value_heads:     number of K/V heads (GQA groups)
        max_position_embeddings: max sequence length for RoPE cache
        rope_theta:              RoPE base frequency
        attention_dropout:       dropout on attention weights (training only)
        use_qk_norm:             apply RMSNorm to Q and K
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        assert self.head_dim * num_attention_heads == hidden_size
        assert num_attention_heads % num_key_value_heads == 0

        # --- Standard QKV projections (no bias) ---
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # --- Mu-Guidance projections (Eq. 5 in paper) ---
        # mu from the previous layer biases K, Q, V:
        #   K = x W_K + mu_prev W_muK
        self.mu_to_q = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.mu_to_k = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.mu_to_v = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)

        # --- QK normalization ---
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        # --- RoPE ---
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_position_embeddings, theta=rope_theta)

        self.attention_dropout = attention_dropout

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        mu_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [B, S, H]
            attention_mask: optional additive mask
            past_key_value: cached (K, V) for generation
            use_cache:      return updated KV cache
            mu_prev:        [B, S, H] mu from previous layer (Mu-Guidance)

        Returns:
            output:         [B, S, H]
            new_kv_cache:   optional (K, V) tuple
        """
        B, S, _ = hidden_states.shape

        # --- Compute Q, K, V with optional mu bias ---
        # Fused path: cat([x, mu]) @ cat([W, W_mu]) = x W + mu W_mu
        # This halves the number of matmuls (3 instead of 6).
        if mu_prev is not None:
            x_mu = torch.cat([hidden_states, mu_prev], dim=-1)       # [B, S, 2H]
            wq = torch.cat([self.q_proj.weight, self.mu_to_q.weight], dim=1)
            wk = torch.cat([self.k_proj.weight, self.mu_to_k.weight], dim=1)
            wv = torch.cat([self.v_proj.weight, self.mu_to_v.weight], dim=1)
            q = F.linear(x_mu, wq)
            k = F.linear(x_mu, wk)
            v = F.linear(x_mu, wv)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # Reshape to multi-head: [B, heads, S, head_dim]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK normalization
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # KV cache handling
        kv_seq_len = S
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        # Apply RoPE
        cos, sin = self.rotary_emb(kv_seq_len)
        cos = cos.to(q.device, dtype=q.dtype)
        sin = sin.to(q.device, dtype=q.dtype)
        if past_key_value is not None:
            cos = cos[kv_seq_len - S:]
            sin = sin[kv_seq_len - S:]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Append to cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        # GQA: repeat KV heads to match Q heads
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention (prefer SDPA / Flash Attention)
        if HAS_SDPA:
            dropout_p = self.attention_dropout if self.training else 0.0
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=(attention_mask is None),
            )
        else:
            attn_output = self._manual_attention(q, k, v, attention_mask, S, kv_seq_len)

        # Merge heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_output), new_kv

    # ------------------------------------------------------------------
    # Fallback attention (PyTorch < 2.0)
    # ------------------------------------------------------------------

    def _manual_attention(self, q, k, v, mask, seq_len, kv_seq_len):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len + 1,
        )
        scores = scores.masked_fill(causal, float("-inf"))
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.attention_dropout > 0:
            weights = F.dropout(weights, p=self.attention_dropout)
        return torch.matmul(weights, v)
