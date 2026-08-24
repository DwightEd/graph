"""Configuration for the attention holonomy mechanism audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


PRIMARY_FEATURES = (
    "depth_transport_error",
    "relay_transport_error",
    "relay_path_dispersion",
    "depth_relay_disagreement",
    "query_set_error",
    "diamond_holonomy",
)

CONTROL_FEATURES = (
    "depth_transport_gain",
    "relay_transport_gain",
    "query_set_gain",
    "relay_rewire_gain",
    "diamond_target_error",
)


@dataclass(frozen=True)
class GraphConfig:
    block_rows: int = 4096
    censored_fill_ratio: float = 0.5
    numerical_tolerance: float = 4e-3
    max_relay_predecessors: int = 12
    max_query_events: int = 32
    minimum_event_mass: float = 1e-8


@dataclass(frozen=True)
class TransportConfig:
    ridge_alpha: float = 1e-2
    minimum_pairs: int = 32


@dataclass(frozen=True)
class ReferenceConfig:
    calibration_fraction: float = 0.3
    reservoir_rows: int = 50_000
    nuisance_ridge_alpha: float = 1e-2
    position_degree: int = 3
    seed: int = 20260825


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_replicates: int = 500
    matched_position_weight: float = 1.0
    matched_degree_weight: float = 0.25
    seed: int = 20260825


@dataclass(frozen=True)
class AuditConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
