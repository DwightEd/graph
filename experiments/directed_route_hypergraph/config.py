"""Configuration for the explicit directed row-hypergraph encoder."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    slot_count: int = 4
    slots_per_role: int = 2
    slot_dim: int = 16
    edge_hidden_dim: int = 64
    lag_buckets: int = 12
    dropout: float = 0.1
    residual_weight: float = 1.0

    @property
    def hidden_dim(self) -> int:
        return self.slot_count * self.slot_dim


@dataclass(frozen=True)
class LearningConfig:
    rows_per_graph: int = 256
    layout_rows_per_batch: int = 64
    layout_min_mass: float = 1e-4
    layout_max_elements: int = 8_000_000
    layout_max_work_elements: int = 250_000_000
    layout_order: str = "ordered"
    incidence_dropout: float = 0.15
    head_dropout: float = 0.05
    flow_weight: float = 0.5
    layout_weight: float = 0.25
    variance_weight: float = 0.05


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.15
    detector_fraction: float = 0.20
    seed: int = 20260827
