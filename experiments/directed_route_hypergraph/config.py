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
    latent_mode: str = "deterministic"
    vae_export: str = "mean_logvar"
    posterior_logvar_min: float = -8.0
    posterior_logvar_max: float = 4.0

    @property
    def hidden_dim(self) -> int:
        return self.slot_count * self.slot_dim


@dataclass(frozen=True)
class LearningConfig:
    positive_edges_per_graph: int = 4096
    holdout_fraction: float = 0.15
    negative_count: int = 1
    negative_attempt_factor: int = 8
    layout_rows_per_graph: int = 32
    layout_rows_per_batch: int = 64
    layout_min_mass: float = 1e-4
    layout_max_elements: int = 8_000_000
    layout_max_work_elements: int = 250_000_000
    layout_order: str = "ordered"
    incidence_dropout: float = 0.0
    head_dropout: float = 0.0
    flow_weight: float = 0.0
    layout_weight: float = 0.0
    variance_weight: float = 0.05
    kl_weight: float = 1e-3
    kl_free_bits: float = 1e-2
    kl_warmup_epochs: int = 4


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.15
    detector_fraction: float = 0.20
    seed: int = 20260827
