"""Validated baseline-model configuration and deterministic parameter accounting.

The calculation mirrors the planned architecture, not an instantiated model. Phase 2
will test the resulting ``nn.Module`` parameter total against this same calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration cannot describe the planned baseline model."""


@dataclass(frozen=True)
class ModelConfig:
    """Architecture values for a decoder-only Transformer.

    Tensor flow (implemented in Phase 2): token IDs ``[B, T]`` -> hidden states
    ``[B, T, D]`` -> attention Q/K/V ``[B, H, T, Dh]`` -> logits ``[B, T, V]``.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    head_dim: int
    ffn_dim: int
    context_length: int
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "head_dim",
            "ffn_dim",
            "context_length",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer; got {value!r}.")
        if self.d_model != self.n_heads * self.head_dim:
            raise ConfigurationError(
                "d_model must equal n_heads * head_dim "
                f"({self.d_model} != {self.n_heads} * {self.head_dim})."
            )
        if self.head_dim % 2 != 0:
            raise ConfigurationError("head_dim must be even for pairwise RoPE rotation.")
        if self.dropout != 0.0:
            raise ConfigurationError("The baseline requires dropout to be exactly 0.0.")
        if self.rope_theta <= 0.0:
            raise ConfigurationError("rope_theta must be positive.")
        if self.rms_norm_eps <= 0.0:
            raise ConfigurationError("rms_norm_eps must be positive.")


@dataclass(frozen=True)
class ParameterCountPolicy:
    """Acceptance range for the approximately 20M-parameter baseline."""

    target: int
    tolerance: int

    def __post_init__(self) -> None:
        if self.target <= 0 or self.tolerance < 0:
            raise ConfigurationError("Parameter target must be positive and tolerance non-negative.")

    def accepts(self, parameter_count: int) -> bool:
        """Return whether ``parameter_count`` is within the inclusive tolerance."""
        return abs(parameter_count - self.target) <= self.tolerance


@dataclass(frozen=True)
class ParameterBreakdown:
    """Exact count for each trainable parameter group in the planned model."""

    token_embedding: int
    attention_per_layer: int
    mlp_per_layer: int
    rms_norm_per_layer: int
    final_rms_norm: int
    lm_head: int
    n_layers: int

    @property
    def total(self) -> int:
        """Return the exact trainable-parameter total."""
        return (
            self.token_embedding
            + self.n_layers
            * (self.attention_per_layer + self.mlp_per_layer + self.rms_norm_per_layer)
            + self.final_rms_norm
            + self.lm_head
        )


@dataclass(frozen=True)
class BaselineConfig:
    """The Phase 1 YAML schema: model values plus count acceptance policy."""

    model: ModelConfig
    parameter_count: ParameterCountPolicy


def calculate_parameter_breakdown(config: ModelConfig) -> ParameterBreakdown:
    """Calculate the planned model's exact trainable parameter count.

    The baseline has bias-free Q/K/V/output projections and bias-free SwiGLU
    projections. Each block has two RMSNorm scales. The LM head contributes zero
    additional parameters because it is tied to the token embedding.
    """
    attention_biases = 4 * config.d_model if config.attention_bias else 0
    mlp_biases = 2 * config.ffn_dim + config.d_model if config.mlp_bias else 0
    return ParameterBreakdown(
        token_embedding=config.vocab_size * config.d_model,
        attention_per_layer=4 * config.d_model * config.d_model + attention_biases,
        mlp_per_layer=3 * config.d_model * config.ffn_dim + mlp_biases,
        rms_norm_per_layer=2 * config.d_model,
        final_rms_norm=config.d_model,
        lm_head=0 if config.tie_embeddings else config.vocab_size * config.d_model,
        n_layers=config.n_layers,
    )


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_name} must be a mapping.")
    return value


def _construct_dataclass(cls: type[Any], values: Mapping[str, Any], field_name: str) -> Any:
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown {field_name} fields: {', '.join(sorted(unknown))}.")
    try:
        return cls(**values)
    except TypeError as error:
        raise ConfigurationError(f"Invalid {field_name}: {error}") from error


def load_baseline_config(path: str | Path) -> BaselineConfig:
    """Load the restricted Phase 1 YAML schema using ``yaml.safe_load``."""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as error:
        raise ConfigurationError(f"Could not read configuration {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {error}") from error

    root = _require_mapping(loaded, "configuration")
    allowed = {"model", "parameter_count"}
    unknown = set(root) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown top-level fields: {', '.join(sorted(unknown))}.")
    if set(root) != allowed:
        raise ConfigurationError("Configuration must contain exactly 'model' and 'parameter_count'.")
    return BaselineConfig(
        model=_construct_dataclass(ModelConfig, _require_mapping(root["model"], "model"), "model"),
        parameter_count=_construct_dataclass(
            ParameterCountPolicy,
            _require_mapping(root["parameter_count"], "parameter_count"),
            "parameter_count",
        ),
    )
