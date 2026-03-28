"""
Complexity Model — complete decoder-only Transformer.

Architecture overview:
    embed_tokens -> [TransformerBlock x N] -> norm -> lm_head

Each TransformerBlock:
    Attention(x, mu_prev) -> residual -> MLP(x, token_ids) -> residual -> mu

Key innovations:
- mu_init: a learnable parameter that gives layer 0 a mu_prev (so even the
  first layer benefits from Mu-Guidance).
- GPT-style residual init: residual output projections (o_proj, down_proj)
  are initialized with std = 0.02 / sqrt(2 * num_layers) to prevent the
  residual stream from growing with depth.
- Mu flows layer-to-layer with clamping to [-2, 2].

Reference: Section 3 of the paper.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from ..core.normalization import RMSNorm
from ..core.layer import TransformerBlock
from .config import ComplexityConfig


# ======================================================================
# Output containers
# ======================================================================

@dataclass
class ModelOutput:
    """Output from ComplexityModel."""
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


@dataclass
class CausalLMOutput:
    """Output with optional loss."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


# ======================================================================
# Model
# ======================================================================

class ComplexityModel(nn.Module):
    """
    Complexity Transformer (decoder-only).

    Mu-Guidance flow:
        mu_prev = mu_init.expand(B, S, H)     # learnable starting mu
        for layer in layers:
            x, kv, mu = layer(x, ..., mu_prev=mu_prev)
            mu_prev = clamp(mu, -2, 2)         # prevent feedback explosion
    """

    def __init__(self, config: ComplexityConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                attention_dropout=config.attention_dropout,
                use_token_routed_mlp=config.use_token_routed_mlp,
                num_experts=config.num_experts,
                vocab_size=config.vocab_size,
                shared_expert=config.shared_expert,
                use_qk_norm=config.use_qk_norm,
                use_mu_guidance=config.use_mu_guidance,
            )
            for _ in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Learnable mu_init: provides mu_prev for layer 0
        if config.use_mu_guidance:
            self.mu_init = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Initialize weights
        self.apply(self._init_weights)
        self._init_residual_scaling()

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module):
        """Standard init: normal_(std=0.02) for Linear and Embedding."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _init_residual_scaling(self):
        """
        GPT-style residual init: scale output projections by 1/sqrt(2N).

        Targets: o_proj (attention output) and down_proj* (MLP output).
        Prevents the residual stream from growing with depth.
        Ref: Radford et al. (2019), "Language Models are Unsupervised Multitask Learners"
        """
        n = self.config.num_hidden_layers
        residual_std = self.config.initializer_range / (2 * n) ** 0.5
        for layer in self.layers:
            # Attention output
            if hasattr(layer.self_attn, 'o_proj'):
                nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=residual_std)
            # MLP down projection (nn.Parameter for TokenRoutedMLP)
            mlp = layer.mlp
            if hasattr(mlp, 'down_proj_w') and isinstance(mlp.down_proj_w, nn.Parameter):
                nn.init.normal_(mlp.down_proj_w, mean=0.0, std=residual_std)
            elif hasattr(mlp, 'down_proj') and isinstance(mlp.down_proj, nn.Linear):
                nn.init.normal_(mlp.down_proj.weight, mean=0.0, std=residual_std)
            # Shared expert down projection
            if hasattr(mlp, 'shared_down') and isinstance(mlp.shared_down, nn.Linear):
                nn.init.normal_(mlp.shared_down.weight, mean=0.0, std=residual_std)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        """
        Forward pass.

        Args:
            input_ids:       [B, S] token IDs
            attention_mask:  optional mask
            labels:          [B, S] target IDs for CE loss (shifted internally)
            past_key_values: KV caches for generation
            use_cache:       return updated caches

        Returns:
            CausalLMOutput with loss (if labels), logits, and optional caches
        """
        B, S = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Mu-Guidance: learnable starting mu for layer 0
        mu_prev = None
        if self.config.use_mu_guidance:
            mu_prev = self.mu_init.expand(B, S, -1)

        new_kvs = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, new_kv, mu_contextual = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                token_ids=input_ids,
                mu_prev=mu_prev,
            )

            # Propagate mu to next layer (clamped to prevent explosion)
            if mu_contextual is not None:
                mu_prev = torch.clamp(mu_contextual, -2.0, 2.0)

            if use_cache:
                new_kvs.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits, past_key_values=new_kvs)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

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
        Autoregressive text generation with KV-cache.

        Args:
            input_ids:      [B, S] prompt token IDs
            max_new_tokens: maximum tokens to generate
            temperature:    sampling temperature
            top_k:          top-k filtering (0 = off)
            top_p:          nucleus sampling threshold
            do_sample:      sample vs. greedy
            eos_token_id:   stop token

        Returns:
            [B, S + generated] token IDs
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        past_key_values = None

        for _ in range(max_new_tokens):
            # Use only the last token when we have a cache
            curr = input_ids[:, -1:] if past_key_values is not None else input_ids

            out = self.forward(curr, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :] / temperature

            # Top-k
            if top_k > 0:
                cutoff = torch.topk(logits, top_k)[0][..., -1, None]
                logits[logits < cutoff] = float("-inf")

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if (next_token == eos_token_id).all():
                break

        return input_ids

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str):
        """Save model weights and config to a directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        torch.save(self.state_dict(), p / "model.pt")

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "ComplexityModel":
        """Load model from a directory containing config.json and model.pt."""
        p = Path(path)
        with open(p / "config.json") as f:
            config = ComplexityConfig.from_dict(json.load(f))
        model = cls(config)
        weights = torch.load(p / "model.pt", map_location=device)
        model.load_state_dict(weights, strict=False)
        return model.to(device)
