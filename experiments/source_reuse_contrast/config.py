"""Configuration for causal source-reuse predictability learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SourceReuseConfig:
    """Model, candidate matching, validation, and optimization settings."""

    hidden_dim: int = 64
    layer_embedding_dim: int = 12
    head_embedding_dim: int = 12
    relation_embedding_dim: int = 8
    source_bin_embedding_dim: int = 8
    usage_embedding_dim: int = 6
    prompt_position_bins: int = 16
    response_lag_bins: int = 16
    usage_bins: int = 6

    memory_mode: str = "dynamic"  # current | birth | dynamic
    temperature: float = 0.2
    negative_count: int = 4
    negative_pool_size: int = 32
    prompt_position_tolerance: float = 0.12
    response_lag_tolerance: float = 0.25

    dropout: float = 0.1
    bptt_steps: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    validation_fraction: float = 0.1
    early_stopping_patience: int = 3
    score_rounds: int = 4
    block_rows: int = 8192
    random_seed: int = 20260820
    show_progress: bool = True

    def validate(self) -> None:
        positive_int = (
            self.hidden_dim,
            self.layer_embedding_dim,
            self.head_embedding_dim,
            self.relation_embedding_dim,
            self.source_bin_embedding_dim,
            self.usage_embedding_dim,
            self.prompt_position_bins,
            self.response_lag_bins,
            self.usage_bins,
            self.negative_count,
            self.negative_pool_size,
            self.bptt_steps,
            self.epochs,
            self.early_stopping_patience,
            self.score_rounds,
            self.block_rows,
        )
        if min(positive_int) < 1:
            raise ValueError("integer configuration values must be positive")
        if self.memory_mode not in {"current", "birth", "dynamic"}:
            raise ValueError("memory_mode must be current, birth, or dynamic")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if not 0.0 < self.prompt_position_tolerance <= 1.0:
            raise ValueError("prompt_position_tolerance must be in (0, 1]")
        if not 0.0 < self.response_lag_tolerance <= 1.0:
            raise ValueError("response_lag_tolerance must be in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
