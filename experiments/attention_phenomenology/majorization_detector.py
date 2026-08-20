"""Label-free detector built from causal prompt-route concentration traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .majorization_dynamics import (
    CausalRouteTrace,
    CausalStateFilter,
)


@dataclass(frozen=True)
class MajorizationDetectorConfig:
    history_decay: float = 0.9
    majorization_tolerance: float = 1e-6
    fit_tokens_per_sample: int = 128
    minimum_scale: float = 0.01
    maximum_standardized_value: float = 10.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.history_decay < 1.0:
            raise ValueError("history_decay must be in [0, 1)")
        if self.majorization_tolerance < 0.0:
            raise ValueError("majorization_tolerance cannot be negative")
        if self.fit_tokens_per_sample < 1:
            raise ValueError("fit_tokens_per_sample must be positive")
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive")
        if self.maximum_standardized_value <= 0.0:
            raise ValueError("maximum_standardized_value must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MajorizationTokenScores:
    majorization_evidence: torch.Tensor
    concentration_level: torch.Tensor
    hill_shape: torch.Tensor
    source_affinity: torch.Tensor
    valid_channel_fraction: torch.Tensor
    standardized_observation: torch.Tensor
    state_probability: torch.Tensor
    entry_probability: torch.Tensor
    basin_probability: torch.Tensor
    current_probability: torch.Tensor
    forecast_probability: torch.Tensor
    valid: torch.Tensor


class CausalMajorizationDetector:
    """Robustly normalize route traces and apply the causal state filter."""

    labels_read = False

    def __init__(
        self,
        *,
        config: MajorizationDetectorConfig,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):
        if center.shape != (2,) or scale.shape != (2,):
            raise ValueError("center and scale must each contain two values")
        if not bool(torch.isfinite(center).all() and torch.isfinite(scale).all()):
            raise ValueError("center and scale must be finite")
        if bool((scale <= 0).any()):
            raise ValueError("scale must be positive")
        self.config = config
        self.center = center.detach().float().cpu()
        self.scale = scale.detach().float().cpu()

    @classmethod
    def fit(
        cls,
        traces: list[CausalRouteTrace] | tuple[CausalRouteTrace, ...],
        *,
        config: MajorizationDetectorConfig | None = None,
    ) -> CausalMajorizationDetector:
        """Fit robust scales with an equal token cap for every response."""

        config = MajorizationDetectorConfig() if config is None else config
        sampled = []
        for trace in traces:
            values = torch.stack(
                (trace.majorization_evidence, trace.concentration_level), dim=-1
            )
            values = values[trace.valid]
            if not len(values):
                continue
            if len(values) > config.fit_tokens_per_sample:
                index = torch.linspace(
                    0,
                    len(values) - 1,
                    config.fit_tokens_per_sample,
                    device=values.device,
                ).round().long()
                values = values[index]
            sampled.append(values.float().cpu())
        if not sampled:
            raise ValueError("fit needs at least one valid causal route observation")

        matrix = torch.cat(sampled)
        center = matrix.median(dim=0).values
        mad = 1.4826 * (matrix - center).abs().median(dim=0).values
        standard_deviation = matrix.std(dim=0, unbiased=False)
        scale = torch.where(
            mad > config.minimum_scale,
            mad,
            standard_deviation,
        ).clamp_min(config.minimum_scale)
        return cls(config=config, center=center, scale=scale)

    def score(self, trace: CausalRouteTrace) -> MajorizationTokenScores:
        raw = torch.stack(
            (trace.majorization_evidence, trace.concentration_level), dim=-1
        )
        center = self.center.to(device=raw.device, dtype=raw.dtype)
        scale = self.scale.to(device=raw.device, dtype=raw.dtype)
        standardized = ((raw - center) / scale).clamp(
            -self.config.maximum_standardized_value,
            self.config.maximum_standardized_value,
        )
        observations = torch.column_stack(
            (standardized, trace.source_affinity.clamp(0.0, 1.0))
        )
        states = CausalStateFilter().run(observations, valid=trace.valid)
        return MajorizationTokenScores(
            majorization_evidence=trace.majorization_evidence,
            concentration_level=trace.concentration_level,
            hill_shape=trace.hill_shape,
            source_affinity=trace.source_affinity,
            valid_channel_fraction=trace.valid_channel_fraction,
            standardized_observation=standardized,
            state_probability=states.state_probability,
            entry_probability=states.entry_probability,
            basin_probability=states.basin_probability,
            current_probability=states.current_probability,
            forecast_probability=states.forecast_probability,
            valid=trace.valid,
        )
