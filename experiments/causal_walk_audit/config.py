"""Frozen configuration for the typed route-grammar detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class GraphConfig:
    block_rows: int = 4096
    recent_lag: int = 4
    numerical_tolerance: float = 1e-3


@dataclass(frozen=True)
class GrammarConfig:
    alpha: float = 0.5
    backoff_tau: float = 32.0


@dataclass(frozen=True)
class PhaseConfig:
    cusum_slack: float = 0.5
    rupture_decay: float = 0.95
    closure_decay: float = 0.9
    scale_floor: float = 1e-3
    epsilon: float = 1e-8


@dataclass(frozen=True)
class CalibrationConfig:
    channel_fraction: float = 0.2
    fusion_fraction: float = 0.2
    reservoir_rows: int = 20_000
    topology_min_changed_fraction: float = 0.5
    seed: int = 20260825


@dataclass(frozen=True)
class AuditConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    grammar: GrammarConfig = field(default_factory=GrammarConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
