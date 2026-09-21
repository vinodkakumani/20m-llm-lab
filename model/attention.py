"""Causal multi-head self-attention with RoPE."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Transform hidden states ``[B, T, D]`` with causal self-attention.

    Q, K, and V are reshaped to ``[B, H, T, Dh]``. The causal attention kernel
    ensures token position *t* attends only to positions at or before *t*.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.context_length = config.context_length
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.rope = RotaryEmbedding(config.head_dim, config.context_length, config.rope_theta)

    def _split_heads(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        return hidden_states.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return causal-attended hidden states with shape ``[B, T, D]``."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, D].")
        batch_size, sequence_length, _ = hidden_states.shape
        if sequence_length > self.context_length:
            raise ValueError("Sequence length exceeds the configured context length.")
        query = self.rope(self._split_heads(self.q_proj(hidden_states)))
        key = self.rope(self._split_heads(self.k_proj(hidden_states)))
        value = self._split_heads(self.v_proj(hidden_states))
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.out_proj(attended)

