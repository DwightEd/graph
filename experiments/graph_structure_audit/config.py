"""Configuration for label-free multiplex graph recovery."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RecoveryConfig:
    representation: str = "full"
    hidden_dim: int = 96
    role_dim: int = 8
    position_dim: int = 8
    lag_bins: int = 16
    channel_mask_rate: float = 0.25
    pair_layer_mask_rate: float = 0.15
    diagonal_mask_rate: float = 0.25
    dropout: float = 0.1
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.1
    patience: int = 3
    score_rounds: int = 4
    block_rows: int = 8192
    random_seed: int = 20260822
    show_progress: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
