"""
Transformer decoder layer for Complexity Deep architecture.

With Full INL Dynamics - robotics-grade control with velocity tracking.

Architecture per layer:
    1. KQV Attention (perception)
    2. INL Dynamics (control/stabilization)
    3. Token-Routed MLP (transformation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from complexity_deep.core.normalization import RMSNorm
from complexity_deep.core.attention import ComplexityAttention
from complexity_deep.core.mlp import ComplexityMLP
from complexity_deep.core.token_routed_mlp import TokenRoutedMLP, TokenRoutedMLPParallel

# Try to import Triton-accelerated version (5-6x faster)
try:
    from complexity_deep.cuda.triton_token_routed import TokenRoutedMLPTriton, HAS_TRITON
except ImportError:
    TokenRoutedMLPTriton = None
    HAS_TRITON = False


class INLDynamics(nn.Module):
    """
    Full INL Dynamics - Robotics-grade control with velocity tracking.

    Equations (like a physical system):
        error = h - mu(h)                   # deviation from contextual equilibrium
        v_next = alpha * v - beta * error   # velocity update (momentum + correction)
        h_next = h + dt * gate * v_next     # position update (integration)

    This creates smooth, stable trajectories like a robot controller:
        - alpha: inertia (momentum, smooth movements)
        - beta: correction strength (feedback, error correction)
        - gate: amplitude control (safety, precision)
        - mu: CONTEXTUAL target equilibrium (adapts per token)
        - dt: timestep (integration speed)

    Benefits for robotics:
        - Smooth trajectories (no jerky movements)
        - Stable convergence (PID-like control)
        - Learnable dynamics per dimension
        - Real-time capable

    Contextual mu:
        mu = mu_base + mu_proj(h)
        - mu_base: global equilibrium (like before)
        - mu_proj: context-dependent adjustment (NEW)
        This allows different tokens to have different attractors.
    """

    def __init__(
        self,
        hidden_size: int,
        init_alpha: float = 0.9,      # high inertia = smooth
        init_beta: float = 0.1,       # low correction = stable
        init_gate: float = 0.5,       # medium amplitude
        dt: float = 0.1,              # integration timestep
        controller_hidden: int = 64,  # controller MLP size
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        # Learnable equilibrium (target position) - base component
        # Renamed from 'mu' to 'mu_base' for contextual mu
        # But we keep 'mu' as alias for checkpoint compatibility
        self.mu = nn.Parameter(torch.zeros(hidden_size))

        # Contextual mu projection - adapts equilibrium per token
        # Initialized to zero so mu_contextual = mu_base at start (checkpoint compatible)
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)  # Start as identity to mu_base

        # Controller MLP - computes alpha, beta, gate from context
        # Input: [h, v] concatenated
        self.controller = nn.Sequential(
            nn.Linear(hidden_size * 2, controller_hidden),
            nn.SiLU(),
            nn.Linear(controller_hidden, hidden_size * 3),  # outputs: alpha, beta, gate
        )

        # Initialize controller biases for desired initial values
        with torch.no_grad():
            bias = self.controller[-1].bias
            # alpha in [0,1] via sigmoid, init to ~0.9
            bias[:hidden_size].fill_(2.2)  # sigmoid(2.2) ≈ 0.9
            # beta in [0,inf) via softplus, init to ~0.1
            bias[hidden_size:hidden_size*2].fill_(-2.2)  # softplus(-2.2) ≈ 0.1
            # gate in [0,1] via sigmoid, init to ~0.5
            bias[hidden_size*2:].fill_(0.0)  # sigmoid(0) = 0.5

            # Small weights for stable start
            self.controller[-1].weight.normal_(0, 0.01)

    def forward(
        self,
        h: torch.Tensor,
        v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply dynamics update.

        Args:
            h: Hidden states [batch, seq_len, hidden_size]
            v: Velocity states [batch, seq_len, hidden_size] (None = init to zero)

        Returns:
            h_next: Updated hidden states
            v_next: Updated velocity states
            mu_contextual: Contextual equilibrium (passed to next layer's attention)
        """
        # Initialize velocity if not provided
        if v is None:
            v = torch.zeros_like(h)

        # Controller computes adaptive parameters from [h, v]
        controller_input = torch.cat([h, v], dim=-1)
        controller_out = self.controller(controller_input)

        # Split and apply activations
        alpha_raw, beta_raw, gate_raw = torch.split(
            controller_out, self.hidden_size, dim=-1
        )
        alpha = torch.sigmoid(alpha_raw)      # [0, 1] - inertia
        # CRITICAL FIX: Clamp beta to prevent explosion!
        # softplus can go to infinity, causing NaN after long training
        # Max beta=2.0 keeps dynamics stable (like a real PID controller)
        beta = torch.clamp(F.softplus(beta_raw), max=2.0)  # [0, 2] - correction
        gate = torch.sigmoid(gate_raw)        # [0, 1] - amplitude

        # Contextual mu: base equilibrium + context-dependent adjustment
        mu_contextual = self.mu + self.mu_proj(h)     # [batch, seq, hidden]

        # Dynamics equations
        error = h - mu_contextual                     # deviation from CONTEXTUAL equilibrium
        v_next = alpha * v - beta * error             # velocity update

        # STABILITY: Clamp velocity to prevent runaway accumulation
        # Like velocity limits in real robotics systems
        v_next = torch.clamp(v_next, min=-10.0, max=10.0)

        h_next = h + self.dt * gate * v_next          # position update

        # Return mu_contextual for next layer's attention (mu-guided QK)
        return h_next, v_next, mu_contextual

    def init_velocity(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Initialize velocity to zero."""
        return torch.zeros(batch_size, seq_len, self.hidden_size, device=device)


class DeepDecoderLayer(nn.Module):
    """
    Complexity Deep decoder layer - multicouche robotics architecture.

    Architecture (3 components per layer):
        1. Mu-Guided Attention - perception guided by previous layer's mu
        2. INL Dynamics  - control (stabilize, smooth trajectories)
        3. Token-Routed MLP - transformation (compute features)

    Like a robot:
        - Attention = eyes (perception) - guided by mu (top-down)
        - Dynamics = cerebellum (motor control, balance)
        - MLP = muscles (action)

    INL Innovation (2025):
        - mu from layer N guides attention in layer N+1
        - Creates bidirectional flow: attention -> dynamics -> attention
        - This is the key to making mu cooperate with QKV
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
        hidden_act: str = "silu",
        # Token-Routed MLP params
        use_token_routed_mlp: bool = True,
        num_experts: int = 4,
        vocab_size: int = 100000,
        # 2024 innovations
        use_qk_norm: bool = True,
        sliding_window: int = None,
        use_sdpa: bool = True,
        # INL Dynamics params
        dynamics_alpha: float = 0.9,
        dynamics_beta: float = 0.1,
        dynamics_gate: float = 0.5,
        dynamics_dt: float = 0.1,
        dynamics_controller_hidden: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_token_routed_mlp = use_token_routed_mlp

        # 1. Attention (perception)
        self.self_attn = ComplexityAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            use_qk_norm=use_qk_norm,
            sliding_window=sliding_window,
            use_sdpa=use_sdpa,
        )

        # 2. INL Dynamics (control)
        self.dynamics = INLDynamics(
            hidden_size=hidden_size,
            init_alpha=dynamics_alpha,
            init_beta=dynamics_beta,
            init_gate=dynamics_gate,
            dt=dynamics_dt,
            controller_hidden=dynamics_controller_hidden,
        )

        # 3. MLP (transformation)
        if use_token_routed_mlp:
            if HAS_TRITON and TokenRoutedMLPTriton is not None:
                # Triton-accelerated (5-6x faster with CGGR)
                self.mlp = TokenRoutedMLPTriton(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    vocab_size=vocab_size,
                    hidden_act=hidden_act,
                    use_cggr=True,
                )
            else:
                # Parallel version (same weight format as Triton, uses bmm)
                self.mlp = TokenRoutedMLPParallel(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    vocab_size=vocab_size,
                    hidden_act=hidden_act,
                )
        else:
            self.mlp = ComplexityMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
            )

        # Layer norms (Pre-LN architecture)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        velocity_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,  # INL: mu from previous layer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the 3-component layer with mu-guided attention.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            velocity_states: [batch, seq_len, hidden_size] - dynamics velocity
            attention_mask: Optional attention mask
            past_key_value: Optional cached KV
            use_cache: Whether to return KV cache
            token_ids: [batch, seq_len] - for Token-Routed MLP routing
            mu_prev: [batch, seq_len, hidden_size] - mu from previous layer (guides attention)

        Returns:
            hidden_states: Updated hidden states
            velocity_states: Updated velocity states
            mu_current: Contextual mu (passed to next layer)
            past_key_value: Optional updated KV cache
        """
        # === 1. MU-GUIDED ATTENTION (perception) ===
        # mu_prev from previous layer guides Q and K (top-down influence)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            mu_prev=mu_prev,  # INL: pass mu to guide attention
        )

        # === 2. DYNAMICS (control) ===
        # Returns mu_current which will guide next layer's attention
        hidden_states, velocity_states, mu_current = self.dynamics(hidden_states, velocity_states)

        hidden_states = residual + hidden_states

        # === 3. MLP (transformation) ===
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_token_routed_mlp:
            hidden_states = self.mlp(hidden_states, token_ids=token_ids)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, velocity_states, mu_current, new_past_key_value


# Alias for backward compatibility
ComplexityDecoderLayer = DeepDecoderLayer
INLDynamicsLite = INLDynamics
