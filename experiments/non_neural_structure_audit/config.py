"""Explicit numerical choices for the non-neural structure audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AuditConfig:
    block_rows: int = 8192
    causal_position_bins: int = 10
    recent_tokens: int = 4
    reference_capacity: int = 2048
    reference_minimum_scale: float = 1e-3
    maximum_standardized_value: float = 10.0
    null_replicates: int = 50
    layer_shuffle_replicates: int = 50
    swap_attempts_per_edge: int = 10
    response_lag_bins: int = 8
    random_seed: int = 20260823
    show_progress: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def reference_settings(self) -> dict[str, object]:
        return {
            "causal_position_bins": self.causal_position_bins,
            "recent_tokens": self.recent_tokens,
            "reference_minimum_scale": self.reference_minimum_scale,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_replicates: int = 2000
    permutation_replicates: int = 499
    onset_window: int = 4
    grouped_cv_folds: int = 5
    minimum_confirmation_samples: int = 100
    minimum_positive_responses: int = 50
    endpoint_minimum_changed_fraction: float = 0.7
    random_seed: int = 20260824
