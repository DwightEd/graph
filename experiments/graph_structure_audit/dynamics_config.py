"""Configuration for cross-origin routing dynamics audits."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DynamicsConfig:
    hidden_dim: int = 96
    role_dim: int = 8
    position_dim: int = 8
    lag_bins: int = 16
    dropout: float = 0.1
    input_dropout: float = 0.1
    edge_loss_weight: float = 1.0
    diagonal_loss_weight: float = 0.5
    support_loss_weight: float = 0.1
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.1
    patience: int = 3
    score_rounds: int = 3
    block_rows: int = 8192
    random_seed: int = 20260823
    show_progress: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
