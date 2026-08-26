"""Configuration for deterministic attention-graph node features."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class GraphConfig:
    block_rows: int = 4096
    minimum_edge_weight: float = 0.0
    numerical_tolerance: float = 4e-3


@dataclass(frozen=True)
class FeatureConfig:
    source_basis_dim: int = 6
    head_projection_dim: int = 8
    projection_seed: int = 20260826


@dataclass(frozen=True)
class DetectionConfig:
    reservoir_rows: int = 12_000
    calibration_fraction: float = 0.20
    pca_components: int = 128
    ridge_alpha: float = 1e-3
    scale_floor: float = 1e-3
    standardized_clip: float = 10.0
    seed: int = 20260826


@dataclass(frozen=True)
class MethodConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
