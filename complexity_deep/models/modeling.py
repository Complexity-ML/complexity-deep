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
        1. KQV Attention (perception)
        2. INL Dynamics (control with velocity)
        3. Token-Routed MLP (transformation)

    Velocity is tracked across all layers for smooth trajectories.
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

        # Process through layers
        new_past_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None

            hidden_states, velocity_state, new_past_kv = layer(
                hidden_states,
                velocity_states=velocity_state,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                token_ids=input_ids,
            )

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
    ) -> torch.Tensor:
        """
        Generate tokens with velocity tracking for smooth trajectories.
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
            logits = outputs.logits[:, -1, :] / temperature

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
                    # Fallback to greedy
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits, dim=-1)
                    # Clamp to avoid numerical issues
                    probs = torch.clamp(probs, min=1e-8)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

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
        state_dict = torch.load(path / "model.pt", map_location=device)
        model.load_state_dict(state_dict)

        return model


# Backward compatibility aliases
ComplexityModel = DeepModel
ComplexityForCausalLM = DeepForCausalLM
ComplexityOutput = DeepOutput
