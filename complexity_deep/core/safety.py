"""
Safety Module — Representation Engineering for inference-time safety.

Projects activations onto a learned harm direction and clamps the component
that exceeds a threshold.  This is a post-hoc intervention that does not
affect training; it is installed on a trained model before deployment.

This module is independent of the Mu-Guidance mechanism and works with any
decoder-only transformer.  It is included for completeness but is not central
to the paper's contribution.

Reference:
    Zou et al. (2023), "Representation Engineering: A Top-Down Approach to
    AI Transparency"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SafetyClamp(nn.Module):
    """
    Clamps activations along a harm direction.

    Given a unit-norm harm direction d:
        projection = activation . d
        if projection > threshold:
            activation -= (projection - threshold) * d
    """

    def __init__(self, hidden_size: int, threshold: float = 2.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.register_buffer("harm_direction", torch.zeros(hidden_size))
        self.enabled = False

    def set_harm_direction(self, direction: torch.Tensor):
        """Set the harm direction (will be L2-normalized)."""
        self.harm_direction.copy_(F.normalize(direction.float(), dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.harm_direction.norm() < 1e-6:
            return x
        shape = x.shape
        flat = x.view(-1, shape[-1])
        proj = flat @ self.harm_direction
        correction = F.relu(proj - self.threshold)
        clamped = flat - correction.unsqueeze(-1) * self.harm_direction
        return clamped.view(shape)
