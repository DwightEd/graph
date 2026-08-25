"""Configuration for the HoloRoute event-graph model."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class GraphConfig:
    block_rows: int = 4096
    max_relay_predecessors: int = 12
    max_query_events: int = 32
    minimum_event_mass: float = 1e-8
    numerical_tolerance: float = 4e-3


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    head_layers: int = 2
    head_attention_heads: int = 4
    transport_rank: int = 8
    message_layers: int = 2
    dropout: float = 0.1
    lag_buckets: int = 12


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.20
    mask_fraction: float = 0.20
    relay_drop_fraction: float = 0.15
    minimum_masked_events: int = 1
    validation_masks: int = 3
    seed: int = 20260825


@dataclass(frozen=True)
class LossConfig:
    event: float = 1.0
    depth: float = 0.75
    query: float = 0.50
    relay: float = 0.15
    holonomy: float = 0.05
    variance: float = 0.05
    support: float = 0.25
    censored: float = 0.10


@dataclass(frozen=True)
class DetectionConfig:
    score_folds: int = 8
    reservoir_rows: int = 20_000
    ridge_alpha: float = 1e-3
    covariance_shrinkage: float = 0.20
    scale_floor: float = 1e-3


@dataclass(frozen=True)
class HoloRouteConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
