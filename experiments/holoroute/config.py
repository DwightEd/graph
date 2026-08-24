"""Frozen configuration for the HoloRoute attention-event graph model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    head_encoder_layers: int = 2
    head_encoder_heads: int = 4
    transport_rank: int = 8
    query_inducing_points: int = 4
    message_blocks: int = 2
    dropout: float = 0.1
    lag_buckets: int = 12
    use_depth: bool = True
    use_relay: bool = True
    use_query: bool = True
    use_transport: bool = True
    use_holonomy: bool = True


@dataclass(frozen=True)
class MaskConfig:
    event_fraction: float = 0.2
    relay_fraction: float = 0.15
    minimum_events: int = 1
    score_rounds: int = 6


@dataclass(frozen=True)
class LossConfig:
    event_weight: float = 1.0
    path_weight: float = 0.5
    depth_weight: float = 0.5
    query_weight: float = 0.5
    holonomy_weight: float = 0.25
    variance_weight: float = 0.05


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.2
    seed: int = 20260825


@dataclass(frozen=True)
class DensityConfig:
    ridge_alpha: float = 1e-3
    covariance_shrinkage: float = 0.2
    scale_floor: float = 1e-3
    reservoir_rows: int = 20_000


@dataclass(frozen=True)
class HoloRouteConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskConfig = field(default_factory=MaskConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    density: DensityConfig = field(default_factory=DensityConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
