"""Configuration for GroundedRoute."""

from dataclasses import asdict, dataclass, field

GRAPH_VARIANTS = ("real", "no_message", "weight_shuffle", "endpoint_rewire")
MESSAGE_MODES = ("neighbor", "row_local")
LEARNING_OBJECTIVES = ("pairwise", "row_distribution")


@dataclass(frozen=True)
class GraphConfig:
    block_rows: int = 4096
    numerical_tolerance: float = 4e-3


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    edge_hidden_dim: int = 128
    lag_buckets: int = 12
    dropout: float = 0.1
    head_transition_identity_bias: float = 2.0
    message_mode: str = "neighbor"


@dataclass(frozen=True)
class LearningConfig:
    objective: str = "row_distribution"
    route_rows_per_graph: int = 256
    positive_edges_per_graph: int = 16_384
    negative_count: int = 4
    negative_attempt_factor: int = 4
    variance_weight: float = 0.05


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.15
    detector_fraction: float = 0.20
    seed: int = 20260825


@dataclass(frozen=True)
class InterventionConfig:
    variant: str = "real"
    minimum_changed_fraction: float = 0.01
    endpoint_rewire_passes: int = 4


@dataclass(frozen=True)
class GroundedRouteConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
