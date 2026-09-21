"""Token embeddings for the decoder-only language model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Map token IDs ``[B, T]`` to hidden states ``[B, T, D]``."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Embed integer token IDs with shape ``[batch, time]``."""
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have shape [B, T], got {tuple(token_ids.shape)}.")
        return self.embedding(token_ids)

