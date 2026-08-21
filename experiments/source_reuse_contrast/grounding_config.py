"""Configuration for grounding-sensitive attention graph learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GroundingGraphConfig:
    """Architecture, pretext objectives, interventions, and optimization."""

    hidden_dim: int = 96
    layer_embedding_dim: int = 16
    head_embedding_dim: int = 16
    relation_embedding_dim: int = 8
    lag_embedding_dim: int = 8
    response_lag_bins: int = 16
    received_topk: int = 5

    edge_mask_rate: float = 0.25
    perturbation_scale: float = 0.08
    gate_keep_target: float = 0.65
    gate_regularization: float = 0.02
    raw_loss_weight: float = 0.25
    reuse_loss_weight: float = 1.0
    grounding_loss_weight: float = 1.0
    provenance_loss_weight: float = 0.5
    use_reuse_memory: bool = True

    dropout: float = 0.1
    bptt_steps: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    validation_fraction: float = 0.1
    early_stopping_patience: int = 3
    score_rounds: int = 4
    block_rows: int = 8192
    random_seed: int = 20260821
    show_progress: bool = True

    def validate(self) -> None:
        positive_int = (
            self.hidden_dim,
            self.layer_embedding_dim,
            self.head_embedding_dim,
            self.relation_embedding_dim,
            self.lag_embedding_dim,
            self.response_lag_bins,
            self.received_topk,
            self.bptt_steps,
            self.epochs,
            self.early_stopping_patience,
            self.score_rounds,
            self.block_rows,
        )
        if min(positive_int) < 1:
            raise ValueError("integer configuration values must be positive")
        unit_interval = (
            self.edge_mask_rate,
            self.gate_keep_target,
            self.validation_fraction,
        )
        if not all(0.0 < value < 1.0 for value in unit_interval):
            raise ValueError(
                "mask, gate target, and validation fraction must be in (0, 1)"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.perturbation_scale <= 0.0:
            raise ValueError("perturbation_scale must be positive")
        if self.gate_regularization < 0.0 or self.raw_loss_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if min(
            self.reuse_loss_weight,
            self.grounding_loss_weight,
            self.provenance_loss_weight,
        ) <= 0.0:
            raise ValueError("pretext loss weights must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
