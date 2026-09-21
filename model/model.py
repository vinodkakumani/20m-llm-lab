"""The complete baseline decoder-only Transformer language model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .embeddings import TokenEmbedding
from .normalization import RMSNorm
from .transformer_block import TransformerBlock


class BaselineTransformer(nn.Module):
    """Decoder-only language model from token IDs to vocabulary logits.

    Flow: ``[B, T]`` token IDs -> ``[B, T, D]`` residual stream -> six causal
    blocks -> ``[B, T, V]`` logits. The output-head weight is tied to the input
    token embedding, reducing parameters and sharing lexical representations.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return unnormalized next-token logits with shape ``[B, T, V]``."""
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("Sequence length exceeds the configured context length.")
        hidden_states = self.token_embedding(token_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.lm_head(self.final_norm(hidden_states))

    @staticmethod
    def next_token_loss(logits: Tensor, token_ids: Tensor) -> Tensor:
        """Compute mean cross-entropy for next-token prediction.

        Logits at ``t`` predict the input token at ``t + 1``. Inputs are
        ``logits [B, T, V]`` and ``token_ids [B, T]``; at least two tokens are
        required per sequence.
        """
        if logits.ndim != 3 or token_ids.ndim != 2:
            raise ValueError("Expected logits [B, T, V] and token_ids [B, T].")
        if logits.shape[:2] != token_ids.shape:
            raise ValueError("Logits and token IDs must agree on batch and time dimensions.")
        if token_ids.shape[1] < 2:
            raise ValueError("At least two tokens are required for next-token loss.")
        return F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), token_ids[:, 1:].reshape(-1))

