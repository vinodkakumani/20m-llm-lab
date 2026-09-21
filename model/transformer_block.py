"""Pre-normalized decoder-only Transformer block."""

from __future__ import annotations

from torch import Tensor, nn

from .attention import CausalSelfAttention
from .config import ModelConfig
from .mlp import SwiGLU
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Update a residual stream ``[B, T, D]`` through attention then SwiGLU."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply pre-norm residual attention and pre-norm residual MLP."""
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))

