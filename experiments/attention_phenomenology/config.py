"""Numerical settings for the attention phenomenology audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PhenomenologyConfig:
    """All runtime choices used by feature extraction and controls."""

    null_prompt_position_bins: int = 4
    null_response_lag_bins: int = 8
    recent_response_tokens: int = 4
    causal_position_bins: int = 10
    reference_minimum_scale: float = 1e-3
    maximum_standardized_value: float = 10.0
    block_rows: int = 8192
    random_seed: int = 20260819
    epsilon: float = 1e-8

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
