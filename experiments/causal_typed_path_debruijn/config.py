"""Validated numerical configuration for causal typed-path routing.

The method is deliberately parameter-free: these values select graph and
counting resolutions, not trainable model dimensions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class GraphConfig:
    """Sparse graph decoding and RR-relation resolution."""

    block_rows: int = 4096
    recent_lag: int = 4

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _positive_int(self.block_rows, "block_rows")
        _positive_int(self.recent_lag, "recent_lag")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PathConfig:
    """Maximum number of response-history transitions in the path DP."""

    max_hops: int = 3

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _positive_int(self.max_hops, "max_hops")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeBruijnConfig:
    """Label-free soft higher-order transition counting."""

    order: int = 2
    soft_top_k: int = 2
    alpha: float = 0.5

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if isinstance(self.order, bool) or self.order not in (1, 2):
            raise ValueError("order must be exactly 1 or 2")
        _positive_int(self.soft_top_k, "soft_top_k")
        if _finite(self.alpha, "alpha") <= 0.0:
            raise ValueError("alpha must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChangeConfig:
    """Robust rupture and persistent feedback-lock-in settings."""

    cusum_slack: float = 0.5
    # Prompt-lineage loss is not directionally stable in the existing onset
    # audit.  Keep it as a recorded diagnostic unless an experiment explicitly
    # preregisters a non-zero weight before labels are opened.
    prompt_lineage_drop_weight: float = 0.0
    rupture_decay: float = 0.95
    feedback_ema_decay: float = 0.9
    scale_floor: float = 1e-3
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if _finite(self.cusum_slack, "cusum_slack") < 0.0:
            raise ValueError("cusum_slack must be non-negative")
        if (
            _finite(
                self.prompt_lineage_drop_weight,
                "prompt_lineage_drop_weight",
            )
            < 0.0
        ):
            raise ValueError("prompt_lineage_drop_weight must be non-negative")
        for name in ("rupture_decay", "feedback_ema_decay"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0,1)")
        if _finite(self.scale_floor, "scale_floor") <= 0.0:
            raise ValueError("scale_floor must be positive")
        if _finite(self.eps, "eps") <= 0.0:
            raise ValueError("eps must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationConfig:
    """Three disjoint source streams for fit, channel, and fusion calibration."""

    channel_fraction: float = 0.2
    fusion_fraction: float = 0.2
    reference_size: int = 12000
    top_channels: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        channel = _finite(self.channel_fraction, "channel_fraction")
        fusion = _finite(self.fusion_fraction, "fusion_fraction")
        if channel <= 0.0 or fusion <= 0.0:
            raise ValueError("calibration fractions must be positive")
        if channel + fusion >= 1.0:
            raise ValueError("channel_fraction + fusion_fraction must be less than 1")
        if _positive_int(self.reference_size, "reference_size") < 2:
            raise ValueError("reference_size must be at least 2")
        _positive_int(self.top_channels, "top_channels")
        _non_negative_int(self.seed, "seed")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
