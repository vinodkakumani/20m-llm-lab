"""Deterministic CPU tests for Phase 2 Transformer components."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from model.attention import CausalSelfAttention
from model.config import ModelConfig
from model.model import BaselineTransformer
from model.normalization import RMSNorm
from model.rope import RotaryEmbedding


def small_config() -> ModelConfig:
    """Return a fast, valid model configuration for component tests."""
    return ModelConfig(
        vocab_size=32,
        d_model=8,
        n_layers=1,
        n_heads=2,
        head_dim=4,
        ffn_dim=16,
        context_length=8,
    )


class TransformerComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.config = small_config()

    def test_rms_norm_preserves_shape_and_rms(self) -> None:
        layer = RMSNorm(d_model=4, eps=0.0)
        values = torch.tensor([[[3.0, 4.0, 0.0, 0.0]]])

        output = layer(values)

        self.assertEqual(output.shape, values.shape)
        self.assertTrue(torch.allclose(output.pow(2).mean(dim=-1), torch.ones(1, 1)))

    def test_rope_preserves_shape_and_position_zero(self) -> None:
        rope = RotaryEmbedding(head_dim=4, context_length=4)
        hidden_states = torch.randn(2, 2, 4, 4)

        output = rope(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertTrue(torch.allclose(output[:, :, 0], hidden_states[:, :, 0]))

    def test_attention_output_shape(self) -> None:
        attention = CausalSelfAttention(self.config)
        hidden_states = torch.randn(3, 5, self.config.d_model)

        output = attention(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)

    def test_causal_attention_prevents_future_leakage(self) -> None:
        """Changing future states must not affect outputs at earlier positions."""
        attention = CausalSelfAttention(self.config).eval()
        original = torch.randn(1, 6, self.config.d_model)
        changed_future = original.clone()
        changed_future[:, 3:] = torch.randn_like(changed_future[:, 3:])

        with torch.no_grad():
            original_output = attention(original)
            changed_output = attention(changed_future)

        self.assertTrue(torch.allclose(original_output[:, :3], changed_output[:, :3], atol=1e-6, rtol=0.0))
        self.assertFalse(torch.allclose(original_output[:, 3:], changed_output[:, 3:]))

    def test_model_output_shape_and_tied_weights(self) -> None:
        model = BaselineTransformer(self.config)
        token_ids = torch.randint(0, self.config.vocab_size, (2, 6))

        logits = model(token_ids)

        self.assertEqual(logits.shape, (2, 6, self.config.vocab_size))
        self.assertEqual(model.lm_head.weight.data_ptr(), model.token_embedding.embedding.weight.data_ptr())

    def test_model_prevents_future_token_leakage(self) -> None:
        """The full model must preserve causal isolation, not only its attention layer."""
        model = BaselineTransformer(self.config).eval()
        original = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed_future = torch.tensor([[1, 2, 3, 9, 10, 11]])

        with torch.no_grad():
            original_logits = model(original)
            changed_logits = model(changed_future)

        self.assertTrue(torch.allclose(original_logits[:, :3], changed_logits[:, :3], atol=1e-6, rtol=0.0))
        self.assertFalse(torch.allclose(original_logits[:, 3:], changed_logits[:, 3:]))

    def test_next_token_loss_uses_shifted_targets(self) -> None:
        logits = torch.tensor(
            [[[0.0, 0.0, 9.0], [0.0, 9.0, 0.0], [9.0, 0.0, 0.0]]], requires_grad=True
        )
        token_ids = torch.tensor([[0, 2, 1]])

        loss = BaselineTransformer.next_token_loss(logits, token_ids)
        expected = F.cross_entropy(logits[:, :-1, :].reshape(-1, 3), token_ids[:, 1:].reshape(-1))

        self.assertTrue(torch.allclose(loss, expected))
        self.assertLess(loss.item(), 0.01)

    def test_backward_pass_and_optimizer_step(self) -> None:
        model = BaselineTransformer(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        token_ids = torch.randint(0, self.config.vocab_size, (2, 6))
        before = model.blocks[0].attention.q_proj.weight.detach().clone()

        loss = model.next_token_loss(model(token_ids), token_ids)
        loss.backward()
        self.assertIsNotNone(model.blocks[0].attention.q_proj.weight.grad)
        optimizer.step()

        after = model.blocks[0].attention.q_proj.weight.detach()
        self.assertFalse(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
