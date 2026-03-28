"""
SwiGLU MLP — standard feed-forward network for the Complexity architecture.

Used as the dense baseline and as the Shared Lexical Expert inside the
Token-Routed MLP.

Architecture:
    out = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

Reference:
    Shazeer (2020), "GLU Variants Improve Transformer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU MLP (Gated Linear Unit with SiLU activation).

    Three linear projections, no bias, matching Llama/Mistral convention.
    Parameter count: 3 * hidden_size * intermediate_size.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
