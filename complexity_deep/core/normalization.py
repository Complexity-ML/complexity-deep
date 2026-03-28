"""
RMSNorm — Root Mean Square Layer Normalization.

Standard component used in Llama, Mistral, and the Complexity architecture.
More efficient than LayerNorm: no mean subtraction, no bias.

Reference:
    Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes by the RMS of activations (no mean centering):
        y = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
