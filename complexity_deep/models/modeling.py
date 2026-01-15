"""
Complexity Deep Model Implementation.

Multicouche robotics architecture:
- KQV Attention (perception)
- INL Dynamics with velocity (control)
- Token-Routed MLP (transformation)

Full velocity tracking for smooth, stable trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

from complexity_deep.core.normalization import RMSNorm
from complexity_deep.core.layer import DeepDecoderLayer
from complexity_deep.models.config import ComplexityConfig

# Try to import Triton-accelerated fused mu residual
try:
    from complexity_deep.cuda.triton_mu_qkv import fused_mu_residual_highway, HAS_TRITON
    HAS_FUSED_MU_RESIDUAL = HAS_TRITON
except ImportError:
    HAS_FUSED_MU_RESIDUAL = False


@dataclass
class DeepOutput:
    """Output from DeepModel."""
    last_hidden_state: torch.Tensor
    last_velocity_state: torch.Tensor  # NEW: velocity for robotics
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


@dataclass
class CausalLMOutput:
    """Output from DeepForCausalLM."""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    velocity_state: Optional[torch.Tensor] = None  # NEW: for robotics


class DeepModel(nn.Module):
    """
    Complexity Deep transformer model (decoder-only).

    Multicouche architecture per layer:
        1. Mu-Guided Attention (perception with top-down guidance)
        2. INL Dynamics (control with velocity)
        3. Token-Routed MLP (transformation)

    INL Innovation (2025):
        - mu from layer N guides attention in layer N+1
        - Creates bidirectional flow: attention -> dynamics -> attention
        - Velocity is tracked across all layers for smooth trajectories
    """

    def __init__(self, config: ComplexityConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers with full dynamics
        self.layers = nn.ModuleList([
            DeepDecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                attention_dropout=config.attention_dropout,
                hidden_act=config.hidden_act,
                # Token-Routed MLP params
                use_token_routed_mlp=config.use_token_routed_mlp,
                num_experts=config.num_experts,
                vocab_size=config.vocab_size,
                # 2024 innovations
                use_qk_norm=config.use_qk_norm,
                sliding_window=config.sliding_window,
                use_sdpa=config.use_sdpa,
                # INL Dynamics params
                dynamics_alpha=getattr(config, 'dynamics_alpha', 0.9),
                dynamics_beta=getattr(config, 'dynamics_beta', 0.1),
                dynamics_gate=getattr(config, 'dynamics_gate', 0.5),
                dynamics_dt=getattr(config, 'dynamics_dt', 0.1),
                dynamics_controller_hidden=getattr(config, 'dynamics_controller_hidden', 64),
            )
            for _ in range(config.num_hidden_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        velocity_state: Optional[torch.Tensor] = None,  # NEW: initial velocity
        use_cache: bool = False,
    ) -> DeepOutput:
        """
        Forward pass with velocity tracking.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            past_key_values: Optional cached KV for generation
            velocity_state: Optional initial velocity [batch, seq_len, hidden_size]
            use_cache: Whether to return KV cache

        Returns:
            DeepOutput with hidden states, velocity states, and optional KV cache
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Initialize velocity if not provided
        if velocity_state is None:
            velocity_state = torch.zeros_like(hidden_states)

        # Process through layers with mu propagation
        # mu from layer N guides attention in layer N+1 (top-down flow)
        # INL 2025: Mu residual - accumulate mu across layers (mu highway)
        new_past_key_values = [] if use_cache else None
        mu_prev = None  # First layer has no mu guidance
        mu_residual = None  # Accumulated mu across all layers

        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None

            hidden_states, velocity_state, mu_current, new_past_kv = layer(
                hidden_states,
                velocity_states=velocity_state,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                token_ids=input_ids,
                mu_prev=mu_prev,  # INL: pass mu from previous layer
            )

            # INL 2025: Mu residual highway (Triton accelerated)
            # Accumulate mu across layers for richer context propagation
            if mu_residual is None:
                mu_residual = mu_current.clone()
                mu_prev = mu_current + 0.1 * mu_residual
            elif HAS_FUSED_MU_RESIDUAL and mu_current.is_cuda:
                # Fused Triton kernel: ~1.5x speedup
                mu_prev, mu_residual = fused_mu_residual_highway(mu_current, mu_residual, 0.1)
            else:
                # PyTorch fallback
                mu_residual = mu_residual + mu_current
                mu_prev = mu_current + 0.1 * mu_residual

            if use_cache:
                new_past_key_values.append(new_past_kv)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return DeepOutput(
            last_hidden_state=hidden_states,
            last_velocity_state=velocity_state,
            past_key_values=new_past_key_values,
        )


class DeepForCausalLM(nn.Module):
    """
    Complexity Deep model with language modeling head.

    For causal language modeling with robotics-grade control:
    - Smooth token generation (velocity tracking)
    - Stable convergence (adaptive alpha/beta/gate)
    - Real-time capable
    """

    def __init__(self, config: ComplexityConfig):
        super().__init__()
        self.config = config

        # Base model
        self.model = DeepModel(config)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        velocity_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            past_key_values: Optional cached KV
            velocity_state: Optional velocity for continuous generation
            use_cache: Whether to return KV cache

        Returns:
            CausalLMOutput with loss, logits, KV cache, and velocity
        """
        # Forward through base model
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            velocity_state=velocity_state,
            use_cache=use_cache,
        )

        # Compute logits
        logits = self.lm_head(outputs.last_hidden_state)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            velocity_state=outputs.last_velocity_state,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        # INL Dynamics for inference (PID with mu) - generation only, doesn't affect checkpoint
        use_dynamics: bool = True,
        dynamics_strength: float = 0.1,
        dynamics_alpha: float = 0.9,   # Inertia (momentum) - ignored, uses model's velocity
        dynamics_beta: float = 0.1,    # Correction - ignored, uses model's velocity
    ) -> torch.Tensor:
        """
        Generate tokens with velocity tracking for smooth trajectories.

        Note: dynamics_alpha/beta are accepted for API compatibility with pyllm-server
        but complexity-deep uses its own trained velocity state from the model layers.
        The velocity_state from forward() already incorporates the learned dynamics.

        Args:
            use_dynamics: If True, use velocity state to influence token sampling.
            dynamics_strength: Strength of velocity influence on logits (0-1).
            dynamics_alpha: Ignored (for API compat) - model uses trained dynamics.
            dynamics_beta: Ignored (for API compat) - model uses trained dynamics.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        past_key_values = None
        velocity_state = None  # Track velocity across generation

        for _ in range(max_new_tokens):
            if past_key_values is not None:
                curr_input_ids = input_ids[:, -1:]
                # For cached generation, only use last velocity
                if velocity_state is not None:
                    velocity_state = velocity_state[:, -1:, :]
            else:
                curr_input_ids = input_ids

            outputs = self.forward(
                curr_input_ids,
                past_key_values=past_key_values,
                velocity_state=velocity_state,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            velocity_state = outputs.velocity_state

            # Save raw logits BEFORE any modifications for fallback
            raw_logits = outputs.logits[:, -1, :].clone()
            logits = raw_logits / temperature

            # NEW: Velocity-aware sampling (INL Dynamics for inference)
            # Use velocity to bias towards tokens "in the direction of motion"
            if use_dynamics and velocity_state is not None and dynamics_strength > 0:
                # Project velocity through lm_head to get "velocity in token space"
                # This tells us which tokens the model is "moving towards"
                velocity_logits = self.lm_head(velocity_state[:, -1, :])
                # Normalize and apply as soft bias
                velocity_bias = torch.tanh(velocity_logits) * dynamics_strength
                logits = logits + velocity_bias

            # Apply top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy with NaN/Inf protection
            if do_sample:
                # Check for invalid logits
                valid_logits = logits[logits != float('-inf')]
                if valid_logits.numel() == 0 or torch.isnan(logits).any() or torch.isinf(logits).all():
                    # Fallback to greedy on RAW logits (not filtered)
                    next_token = torch.argmax(raw_logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits, dim=-1)
                    # Clamp to avoid numerical issues
                    probs = torch.clamp(probs, min=1e-8)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(raw_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if (next_token == eos_token_id).all():
                break

        return input_ids

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_path: str):
        """Save model and config."""
        import json
        from pathlib import Path

        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cpu") -> "DeepForCausalLM":
        """Load model from checkpoint."""
        import json
        from pathlib import Path

        path = Path(load_path)

        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = ComplexityConfig.from_dict(config_dict)

        model = cls(config)

        # Try safetensors first, then .pt
        safetensors_path = path / "model.safetensors"
        pt_path = path / "model.pt"

        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path), device=device)
        elif pt_path.exists():
            state_dict = torch.load(pt_path, map_location=device)
        else:
            raise FileNotFoundError(f"No model weights found in {path}. Expected model.safetensors or model.pt")

        model.load_state_dict(state_dict)
        model = model.to(device)

        return model


# Backward compatibility aliases
ComplexityModel = DeepModel
ComplexityForCausalLM = DeepForCausalLM
ComplexityOutput = DeepOutput
