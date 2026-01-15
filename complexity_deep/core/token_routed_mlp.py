"""
Token-Routed MLP for Complexity architecture.

Innovation: Route tokens to specialized experts based on token ID.
Deterministic routing = no router to learn, stable, 100% parallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP (Deterministic MoE).

    Each token is routed to a specific expert based on its token ID.

    Benefits:
    - 1/num_experts compute per token (faster)
    - Specialized experts per token frequency range
    - Deterministic = stable training, no load balancing loss
    - 100% parallel (no routing decisions at runtime)

    Token routing strategy:
        Expert 0: Token IDs 0 to vocab_size/4       (most frequent)
        Expert 1: Token IDs vocab_size/4 to /2
        Expert 2: Token IDs vocab_size/2 to 3/4
        Expert 3: Token IDs 3*vocab_size/4 to end   (most rare)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        vocab_size: int = 100000,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        # Expert size = intermediate_size / num_experts (same total params)
        self.expert_intermediate_size = intermediate_size // num_experts

        # Create experts (SwiGLU style)
        self.experts = nn.ModuleList([
            Expert(hidden_size, self.expert_intermediate_size, hidden_act)
            for _ in range(num_experts)
        ])

        # Precompute token -> expert mapping
        self.register_buffer(
            "token_to_expert",
            self._create_token_mapping(vocab_size, num_experts),
        )

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        """
        Create deterministic mapping from token ID to expert ID.

        Strategy: Modulo routing for uniform distribution.
        token_id % num_experts ensures each expert gets ~25% of tokens
        regardless of token frequency in actual text.

        This prevents expert collapse where frequent tokens (low IDs)
        would all go to expert 0 with range-based routing.
        """
        return torch.arange(vocab_size, dtype=torch.long) % num_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with token-based routing.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len] - original token IDs for routing

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        if token_ids is None:
            # Fallback: use all experts equally (for inference without token_ids)
            return self._forward_all_experts(hidden_states)

        # Get expert assignment for each token
        # Clamp to valid range
        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        expert_ids = self.token_to_expert[token_ids_clamped]  # [batch, seq_len]

        # Process each expert's tokens
        output = torch.zeros_like(hidden_states)

        for expert_id in range(self.num_experts):
            # Mask for tokens routed to this expert
            mask = (expert_ids == expert_id)  # [batch, seq_len]

            if not mask.any():
                continue

            # Get tokens for this expert
            # Flatten for efficient processing
            expert_input = hidden_states[mask]  # [num_tokens, hidden_size]

            # Process through expert
            expert_output = self.experts[expert_id](expert_input)

            # Put back in output
            output[mask] = expert_output

        return output

    def _forward_all_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback: average all experts (for inference without token_ids)."""
        outputs = [expert(hidden_states) for expert in self.experts]
        return torch.stack(outputs, dim=0).mean(dim=0)


class Expert(nn.Module):
    """Single expert MLP (SwiGLU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.act_fn = F.silu if hidden_act == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TokenRoutedMLPParallel(nn.Module):
    """
    Optimized Token-Routed MLP using batched operations.

    v0.12.0: Fused Gate+Up projection (2 bmm -> 1 bmm)
    - Concat gate_proj and up_proj weights
    - Single matmul, split output
    - ~1.3x speedup on SwiGLU

    INL Innovation (2025):
    - Mu-guided expert routing: mu can shift the expert selection
    - Creates soft routing influenced by dynamics context
    - mu_router projects mu to expert logits, adds to base routing
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        vocab_size: int = 100000,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        self.expert_intermediate_size = intermediate_size // num_experts

        # v0.12.0: Fused gate+up projection [num_experts, hidden, 2*intermediate]
        # Instead of separate gate_proj and up_proj, we fuse them
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size * 2) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, self.expert_intermediate_size, hidden_size) * 0.02
        )

        # Backward compatibility: expose gate_proj and up_proj as views
        # This allows loading old checkpoints
        self._gate_proj = None
        self._up_proj = None

        self.act_fn = F.silu if hidden_act == "silu" else F.gelu

        # Token mapping
        self.register_buffer(
            "token_to_expert",
            self._create_token_mapping(vocab_size, num_experts),
        )

        # INL 2025: Mu-guided expert routing
        # mu_router projects mu to expert preference logits
        # Initialized to zero so routing starts as pure token-based
        self.mu_router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.zeros_(self.mu_router.weight)  # Start neutral

    @property
    def gate_proj(self):
        """Backward compatibility: return gate portion of fused weights."""
        return self.gate_up_proj[..., :self.expert_intermediate_size]

    @property
    def up_proj(self):
        """Backward compatibility: return up portion of fused weights."""
        return self.gate_up_proj[..., self.expert_intermediate_size:]

    def _create_token_mapping(self, vocab_size: int, num_experts: int) -> torch.Tensor:
        """Modulo routing for uniform expert distribution."""
        return torch.arange(vocab_size, dtype=torch.long) % num_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,  # INL: mu guides expert selection
    ) -> torch.Tensor:
        """
        Batched forward pass with fused gate+up and mu-guided routing.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            token_ids: [batch, seq_len]
            mu: [batch, seq_len, hidden_size] - mu from dynamics (INL)

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        if token_ids is None:
            # Fallback: use expert 0
            expert_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
        else:
            token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
            base_expert_ids = self.token_to_expert[token_ids_clamped]  # [batch, seq]

            # INL 2025: Mu-guided expert routing
            # mu can override or shift the expert selection
            if mu is not None:
                # Get mu preference for each expert
                mu_logits = self.mu_router(mu)  # [batch, seq, num_experts]

                # Create one-hot for base expert
                base_one_hot = F.one_hot(base_expert_ids, self.num_experts).float()  # [B, S, E]

                # Combine: base routing + mu influence
                # mu_logits adds a soft bias toward different experts
                combined_logits = base_one_hot * 10.0 + mu_logits  # base is strong (10.0)

                # Hard selection: argmax (still deterministic, but mu-influenced)
                expert_ids = combined_logits.argmax(dim=-1)  # [batch, seq]
            else:
                expert_ids = base_expert_ids

        # Flatten
        flat_hidden = hidden_states.view(-1, self.hidden_size)  # [B*S, H]
        flat_expert_ids = expert_ids.view(-1)  # [B*S]

        # v0.12.0: Fused gate+up matmul (1 bmm instead of 2)
        # Gather fused weights: [num_experts, H, 2I] -> [B*S, H, 2I]
        gate_up_weights = self.gate_up_proj[flat_expert_ids]  # [B*S, H, 2I]
        down_weights = self.down_proj[flat_expert_ids]  # [B*S, I, H]

        # Single fused matmul for gate and up
        # [B*S, 1, H] @ [B*S, H, 2I] -> [B*S, 1, 2I] -> [B*S, 2I]
        gate_up_out = torch.bmm(flat_hidden.unsqueeze(1), gate_up_weights).squeeze(1)

        # Split and apply SwiGLU
        gate_out = gate_up_out[..., :self.expert_intermediate_size]  # [B*S, I]
        up_out = gate_up_out[..., self.expert_intermediate_size:]    # [B*S, I]
        intermediate = self.act_fn(gate_out) * up_out  # [B*S, I]

        # Down projection
        # [B*S, 1, I] @ [B*S, I, H] -> [B*S, 1, H]
        output = torch.bmm(intermediate.unsqueeze(1), down_weights).squeeze(1)  # [B*S, H]

        return output.view(batch_size, seq_len, self.hidden_size)
