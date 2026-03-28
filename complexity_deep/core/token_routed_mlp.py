"""
Token-Routed MLP — Deterministic Mixture-of-Experts for the Complexity architecture.

Innovation: each token is routed to exactly one expert based on its token ID.
Routing is deterministic (no learned router, no load-balancing loss) and fully
parallel across experts.

Key design choices:
1. Zipf-balanced greedy bin-packing: tokens sorted by corpus frequency are
   assigned one-by-one to the expert with the lowest accumulated load,
   so each expert sees ~1/E of the total token mass (not just 1/E of the
   vocabulary).
2. Shared Lexical Expert: a dense SwiGLU MLP that ALL tokens pass through,
   capturing common patterns.  The final output is:
       out(x) = SharedMLP(x) + Expert_e(x)
3. Sparse dispatch: only the tokens assigned to expert e are sent through
   that expert's weights (no masked dense computation).

Reference: Section 3.2 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP with Shared Lexical Expert.

    Parameters:
        hidden_size:      model dimension (H)
        intermediate_size: total MLP width (I), split across experts
        num_experts:      number of routed experts (E, default 4)
        vocab_size:       tokenizer vocabulary size (V)
        shared_expert:    whether to include the shared dense MLP (default True)
        token_frequencies: optional [V] tensor of corpus token frequencies
                          for Zipf-balanced routing.  Without it, falls back
                          to simple modulo routing (token_id % E).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        vocab_size: int = 100_000,
        shared_expert: bool = True,
        token_frequencies: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        # Each expert gets 1/E of the total intermediate width
        self.expert_intermediate_size = intermediate_size // num_experts

        # ----- Routed expert weights [E, H, I_e] / [E, I_e, H] -----
        # Stored as 3-D Parameters so FSDP can shard them naturally.
        self.gate_proj_w = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.up_proj_w = nn.Parameter(
            torch.randn(num_experts, hidden_size, self.expert_intermediate_size) * 0.02
        )
        self.down_proj_w = nn.Parameter(
            torch.randn(num_experts, self.expert_intermediate_size, hidden_size) * 0.02
        )

        # ----- Shared Lexical Expert (dense SwiGLU, same width as one expert) -----
        self.use_shared_expert = shared_expert
        if shared_expert:
            self.shared_gate = nn.Linear(hidden_size, self.expert_intermediate_size, bias=False)
            self.shared_up = nn.Linear(hidden_size, self.expert_intermediate_size, bias=False)
            self.shared_down = nn.Linear(self.expert_intermediate_size, hidden_size, bias=False)

        # ----- Token-to-expert mapping (precomputed, non-trainable) -----
        mapping = self._build_token_mapping(vocab_size, num_experts, token_frequencies)
        self.register_buffer("token_to_expert", mapping)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _build_token_mapping(
        vocab_size: int,
        num_experts: int,
        token_frequencies: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build the deterministic token -> expert mapping.

        With token_frequencies: greedy bin-packing.
            1. Sort tokens by descending frequency.
            2. Assign each token to the expert with the lowest accumulated load.
            This ensures each expert handles ~equal total corpus frequency,
            preventing load imbalance caused by Zipf's law.

        Without token_frequencies: simple modulo fallback.
            token_id % num_experts
        """
        if token_frequencies is not None:
            assert token_frequencies.shape[0] == vocab_size
            sorted_indices = token_frequencies.argsort(descending=True)
            mapping = torch.empty(vocab_size, dtype=torch.long)
            expert_loads = [0.0] * num_experts
            for rank in range(vocab_size):
                tid = sorted_indices[rank].item()
                # Assign to least-loaded expert
                best_expert = min(range(num_experts), key=lambda e: expert_loads[e])
                mapping[tid] = best_expert
                expert_loads[best_expert] += token_frequencies[tid].item()
            return mapping

        # Fallback: modulo routing
        return torch.arange(vocab_size, dtype=torch.long) % num_experts

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with sparse dispatch.

        Args:
            hidden_states: [batch, seq_len, H]
            token_ids:     [batch, seq_len] original input token IDs

        Returns:
            output: [batch, seq_len, H]
                    = SharedMLP(x) + Expert_e(x)   if shared_expert
                    = Expert_e(x)                   otherwise
        """
        B, S, H = hidden_states.shape

        if token_ids is None:
            # Inference without token_ids: average all experts
            return self._forward_all_experts(hidden_states)

        # Look up expert assignment per token
        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        expert_ids = self.token_to_expert[token_ids_clamped]  # [B, S]

        # Flatten for dispatch
        flat_x = hidden_states.view(-1, H)           # [N, H]  where N = B*S
        flat_expert_ids = expert_ids.view(-1)          # [N]

        # --- Shared expert (dense, all tokens) ---
        if self.use_shared_expert:
            shared_out = self.shared_down(
                F.silu(self.shared_gate(flat_x)) * self.shared_up(flat_x)
            ).to(flat_x.dtype)
        else:
            shared_out = 0

        # --- Routed experts (sparse dispatch) ---
        routed_out = torch.zeros_like(flat_x)
        for e in range(self.num_experts):
            mask = (flat_expert_ids == e)              # [N] bool
            if not mask.any():
                continue
            x_e = flat_x[mask]                         # [N_e, H]
            gate_e = x_e @ self.gate_proj_w[e]         # [N_e, I_e]
            up_e = x_e @ self.up_proj_w[e]             # [N_e, I_e]
            inter_e = F.silu(gate_e) * up_e            # [N_e, I_e]  (SwiGLU)
            routed_out[mask] = (inter_e @ self.down_proj_w[e]).to(routed_out.dtype)

        # Combine: shared (common patterns) + routed (specialized patterns)
        out = routed_out + shared_out
        return out.view(B, S, H)

    def _forward_all_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fallback: average all experts (inference without token_ids)."""
        flat = hidden_states.view(-1, self.hidden_size)
        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            gate_e = flat @ self.gate_proj_w[e]
            up_e = flat @ self.up_proj_w[e]
            out = out + (F.silu(gate_e) * up_e) @ self.down_proj_w[e]
        out = out / self.num_experts
        if self.use_shared_expert:
            shared = self.shared_down(
                F.silu(self.shared_gate(flat)) * self.shared_up(flat)
            )
            out = out + shared
        return out.view_as(hidden_states)
