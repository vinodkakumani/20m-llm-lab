"""Normalization layers used by the baseline Transformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final hidden dimension.

    Input and output both have shape ``[..., D]``. Unlike LayerNorm, RMSNorm
    rescales by the root mean square and does not subtract the mean.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize in float32 for stable low-precision training, then restore dtype."""
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        mean_square = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(mean_square + self.eps)
        return normalized.to(input_dtype) * self.weight.to(input_dtype)

