"""Rotary positional embeddings (RoPE) for attention queries and keys."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Apply RoPE to tensors shaped ``[B, H, T, Dh]``.

    Cosine and sine caches are non-trainable buffers sized to the configured
    context length. Consecutive pairs in each attention head are rotated by the
    token position, encoding relative position in Q/K dot products.
    """

    def __init__(self, head_dim: int, context_length: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        inverse_frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(context_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequencies)
        # Duplicate each frequency so cos/sin align with even/odd hidden pairs.
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(hidden_states: Tensor) -> Tensor:
        even = hidden_states[..., ::2]
        odd = hidden_states[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(start_dim=-2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Rotate Q or K, preserving its ``[B, H, T, Dh]`` shape."""
        if hidden_states.ndim != 4:
            raise ValueError("RoPE input must have shape [B, H, T, Dh].")
        sequence_length = hidden_states.shape[-2]
        if sequence_length > self.cos_cached.shape[0]:
            raise ValueError("Sequence length exceeds the configured RoPE cache.")
        cosine = self.cos_cached[:sequence_length].to(dtype=hidden_states.dtype)[None, None, :, :]
        sine = self.sin_cached[:sequence_length].to(dtype=hidden_states.dtype)[None, None, :, :]
        return hidden_states * cosine + self._rotate_half(hidden_states) * sine

