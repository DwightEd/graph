"""Configuration for the original-code DBGNN reference."""

from dataclasses import asdict, dataclass


HIGHER_ORDER_MODES = ("causal", "no_transition")


@dataclass(frozen=True)
class DBGNNConfig:
    encoder: str = "dbgnn"
    hidden_dim: int = 64
    embedding_dim: int = 64
    dropout: float = 0.1
    delta_layers: int = 1
    higher_order_mode: str = "causal"
    edge_drop_fraction: float = 0.15
    positives_per_graph: int = 4096
    variance_weight: float = 0.05
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.15
    detector_fraction: float = 0.20
    seed: int = 20260826

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
