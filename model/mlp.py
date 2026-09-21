"""SwiGLU feed-forward network used in each Transformer block."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


class SwiGLU(nn.Module):
    """Apply SwiGLU: ``down(silu(gate(x)) * up(x))`` to ``[B, T, D]``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_dim, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.d_model, config.ffn_dim, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.ffn_dim, config.d_model, bias=config.mlp_bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return transformed hidden states with the original ``[B, T, D]`` shape."""
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

