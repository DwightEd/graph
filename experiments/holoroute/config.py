"""Configuration for prompt-provenance cuts."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class GraphConfig:
    block_rows: int = 4096
    minimum_edge_weight: float = 0.0
    numerical_tolerance: float = 4e-3


@dataclass(frozen=True)
class PCutConfig:
    identity_dim: int = 16
    head_projection_dim: int = 4
    tail_layers: int = 8
    epsilon: float = 1e-8
    seed: int = 20260826


@dataclass(frozen=True)
class DetectionConfig:
    reservoir_rows: int = 50_000
    ridge_alpha: float = 1e-3
    scale_floor: float = 1e-3


@dataclass(frozen=True)
class MethodConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    pcut: PCutConfig = field(default_factory=PCutConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
