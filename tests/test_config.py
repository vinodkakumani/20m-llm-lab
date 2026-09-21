"""Phase 1 tests for model configuration and deterministic parameter counting."""

from __future__ import annotations

from pathlib import Path
import unittest

from model.config import (
    ConfigurationError,
    ModelConfig,
    ParameterCountPolicy,
    calculate_parameter_breakdown,
    load_baseline_config,
)
from model.model import BaselineTransformer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPOSITORY_ROOT / "configs" / "baseline_20m.yaml"


class ModelConfigurationTests(unittest.TestCase):
    def test_baseline_configuration_loads(self) -> None:
        config = load_baseline_config(BASELINE_PATH)

        self.assertEqual(config.model.vocab_size, 32_000)
        self.assertEqual(config.model.d_model, 320)
        self.assertTrue(config.model.tie_embeddings)
        self.assertEqual(config.model.dropout, 0.0)

    def test_baseline_has_documented_exact_parameter_count(self) -> None:
        config = load_baseline_config(BASELINE_PATH)
        breakdown = calculate_parameter_breakdown(config.model)
        model = BaselineTransformer(config.model)

        self.assertEqual(breakdown.total, 20_074_560)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), breakdown.total)
        self.assertEqual(breakdown.lm_head, 0)
        self.assertTrue(config.parameter_count.accepts(breakdown.total))

    def test_parameter_count_policy_rejects_material_drift(self) -> None:
        policy = ParameterCountPolicy(target=20_000_000, tolerance=500_000)

        self.assertFalse(policy.accepts(20_500_001))
        self.assertFalse(policy.accepts(19_499_999))

    def test_incompatible_attention_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "d_model must equal"):
            ModelConfig(
                vocab_size=32_000,
                d_model=320,
                n_layers=6,
                n_heads=5,
                head_dim=32,
                ffn_dim=1280,
                context_length=512,
            )

    def test_dropout_is_locked_to_zero_for_the_baseline(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "dropout"):
            ModelConfig(
                vocab_size=32_000,
                d_model=320,
                n_layers=6,
                n_heads=5,
                head_dim=64,
                ffn_dim=1280,
                context_length=512,
                dropout=0.1,
            )


if __name__ == "__main__":
    unittest.main()
