"""
Attention mechanism for Complexity architecture.

Innovations 2023-2024:
- SDPA (Flash Attention via PyTorch 2.0+)
- QK Normalization (stabilizes training)
- Sliding Window Attention (optional, for long context)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from complexity_deep.core.rotary import RotaryEmbedding, apply_rotary_pos_emb

# Try to import Triton-accelerated fused Mu-QKV
try:
    from complexity_deep.cuda.triton_mu_qkv import fused_mu_qkv_projection, HAS_TRITON
    HAS_FUSED_MU_QKV = HAS_TRITON
except ImportError:
    HAS_FUSED_MU_QKV = False

# Check if SDPA is available (PyTorch 2.0+)
HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


class ComplexityAttention(nn.Module):
    """
    Multi-Head Attention with modern optimizations + Mu-Guided Attention.

    Features (2023-2024):
    - Grouped Query Attention (GQA) - Llama 2
    - Rotary Position Embeddings (RoPE)
    - Flash Attention via SDPA (PyTorch 2.0+)
    - QK Normalization (optional, stabilizes training)
    - Sliding Window Attention (optional, for efficiency)

    INL Innovation (2025):
    - Mu-Guided Attention: mu from previous layer biases Q and K
    - This creates top-down guidance: dynamics inform attention
    - Bidirectional flow: attention -> dynamics -> attention (next layer)
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        # 2024 innovations
        use_qk_norm: bool = True,
        sliding_window: Optional[int] = None,  # None = full attention
        use_sdpa: bool = True,  # Use Flash Attention via SDPA
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        assert self.head_dim * num_attention_heads == hidden_size, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_attention_heads})"
        assert num_attention_heads % num_key_value_heads == 0, \
            f"num_heads ({num_attention_heads}) must be divisible by num_kv_heads ({num_key_value_heads})"

        # Projections (no bias for efficiency)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # Mu-to-QKV projections (INL 2025 - mu guides attention EVERYWHERE)
        # mu from previous layer biases Q, K, AND V - full top-down guidance
        self.mu_to_q = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.mu_to_k = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.mu_to_v = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)

        # QK Normalization (2024 innovation - stabilizes training)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=max_position_embeddings,
            theta=rope_theta,
        )

        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.use_sdpa = use_sdpa and HAS_SDPA

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        mu_prev: Optional[torch.Tensor] = None,  # INL: mu from previous layer
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Flash Attention (SDPA) + Mu-Guided Q/K.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Optional cached KV for generation
            use_cache: Whether to return updated KV cache
            mu_prev: [batch, seq_len, hidden_size] - mu from previous layer (INL)

        Returns:
            output: [batch, seq_len, hidden_size]
            past_key_value: Optional updated KV cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # INL 2025: Fused Mu-QKV projection (Triton accelerated ~2x speedup)
        # Computes: q = x @ Wq + mu @ Wmu_q (same for k, v)
        if HAS_FUSED_MU_QKV and hidden_states.is_cuda and mu_prev is not None:
            q, k, v = fused_mu_qkv_projection(
                hidden_states, mu_prev,
                self.q_proj.weight.t().contiguous(),
                self.k_proj.weight.t().contiguous(),
                self.v_proj.weight.t().contiguous(),
                self.mu_to_q.weight.t().contiguous(),
                self.mu_to_k.weight.t().contiguous(),
                self.mu_to_v.weight.t().contiguous(),
            )
        else:
            # Fallback: Standard PyTorch path
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            # INL: Mu-guided attention - mu from previous layer biases Q, K, AND V
            if mu_prev is not None:
                q = q + self.mu_to_q(mu_prev)
                k = k + self.mu_to_k(mu_prev)
                v = v + self.mu_to_v(mu_prev)

        # Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK Normalization (2024 innovation)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Handle KV cache for generation
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(kv_seq_len)
        cos = cos.to(q.device, dtype=q.dtype)
        sin = sin.to(q.device, dtype=q.dtype)

        # For cached generation, only rotate the new positions
        if past_key_value is not None:
            cos = cos[kv_seq_len - seq_len:]
            sin = sin[kv_seq_len - seq_len:]

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_past_key_value = (k, v) if use_cache else None

        # GQA: Repeat KV heads to match Q heads
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Use SDPA (Flash Attention) if available
        if self.use_sdpa:
            attn_output = self._sdpa_attention(q, k, v, attention_mask, seq_len, kv_seq_len)
        else:
            attn_output = self._standard_attention(q, k, v, attention_mask, seq_len, kv_seq_len)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_past_key_value

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        kv_seq_len: int,
    ) -> torch.Tensor:
        """
        Flash Attention via PyTorch SDPA.

        Benefits:
        - 2-4x faster than standard attention
        - Memory efficient (O(n) vs O(n²))
        - Automatically uses best backend (Flash, Memory Efficient, or Math)
        """
        # Build attention mask for SDPA
        if self.sliding_window is not None and seq_len > self.sliding_window:
            # Sliding window attention (Mistral-style)
            attn_mask = self._make_sliding_window_mask(seq_len, kv_seq_len, q.device, q.dtype)
        else:
            # Standard causal mask - SDPA handles this with is_causal=True
            attn_mask = None

        # Dropout only during training
        dropout_p = self.attention_dropout if self.training else 0.0

        # SDPA automatically selects best implementation:
        # - Flash Attention (if available)
        # - Memory Efficient Attention (fallback)
        # - Standard Math Attention (fallback)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=(attn_mask is None),  # Use built-in causal mask if no custom mask
        )

        return attn_output

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        kv_seq_len: int,
    ) -> torch.Tensor:
        """Standard attention fallback (for PyTorch < 2.0)."""
        # Attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        if self.sliding_window is not None:
            mask = self._make_sliding_window_mask(seq_len, kv_seq_len, q.device, q.dtype)
            attn_weights = attn_weights + mask
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
                diagonal=kv_seq_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Additional attention mask (e.g., padding)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        return torch.matmul(attn_weights, v)

    def _make_sliding_window_mask(
        self,
        seq_len: int,
        kv_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create sliding window attention mask (Mistral-style)."""
        # Create causal mask
        mask = torch.full((seq_len, kv_seq_len), float("-inf"), device=device, dtype=dtype)

        for i in range(seq_len):
            # Can attend to [i - window + 1, i] in kv_seq
            start = max(0, kv_seq_len - seq_len + i - self.sliding_window + 1)
            end = kv_seq_len - seq_len + i + 1
            mask[i, start:end] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, kv_seq]
