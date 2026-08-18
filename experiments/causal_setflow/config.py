"""Configuration objects for the learnable causal attention Set-Flow model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSetConfig:
    """Sparse RR source-set extraction controls.

    ``materialize_query_chunk_size`` only changes how the exact received-support
    tensor is evaluated.  It does not truncate response tokens or source
    candidates beyond the explicitly modelled route/memory source-set bounds.
    """

    max_route_sources: int = 32
    max_memory_sources: int = 16
    route_mass_coverage: float = 0.98
    block_rows: int = 8192
    materialize_query_chunk_size: int = 64
    epsilon: float = 1e-8

    def validate(self) -> None:
        integers = (
            self.max_route_sources,
            self.max_memory_sources,
            self.block_rows,
            self.materialize_query_chunk_size,
        )
        if min(integers) < 1:
            raise ValueError("source-set integer controls must be positive")
        if not 0.5 <= float(self.route_mass_coverage) <= 1.0:
            raise ValueError("route_mass_coverage must be in [0.5, 1]")
        if not 0.0 < float(self.epsilon) < 1.0:
            raise ValueError("epsilon must be in (0, 1)")


@dataclass(frozen=True)
class SetFlowModelConfig:
    """Hierarchical model and self-supervised objective controls.

    Chunking and activation checkpointing preserve the mathematical model. They
    change only the execution schedule and the set of forward activations kept
    for backward.
    """

    hidden_dim: int = 64
    scalar_fourier_dim: int = 16
    set_heads: int = 4
    induced_points: int = 8
    set_blocks: int = 2
    head_mixer_layers: int = 2
    depth_mixer_layers: int = 2
    set_row_chunk_size: int = 4096
    mixer_token_chunk_size: int = 512
    activation_checkpointing: bool = True
    dropout: float = 0.10
    element_mask_probability: float = 0.20
    head_mask_probability: float = 0.20
    layer_mask_probability: float = 0.15
    temporal_loss_weight: float = 0.25
    element_loss_weight: float = 1.0
    head_loss_weight: float = 1.0
    layer_loss_weight: float = 0.5
    variance_loss_weight: float = 0.02
    epsilon: float = 1e-8

    def validate(self) -> None:
        integers = (
            self.hidden_dim,
            self.scalar_fourier_dim,
            self.set_heads,
            self.induced_points,
            self.set_blocks,
            self.head_mixer_layers,
            self.depth_mixer_layers,
            self.set_row_chunk_size,
            self.mixer_token_chunk_size,
        )
        if min(integers) < 1:
            raise ValueError("model integer controls must be positive")
        if self.hidden_dim % self.set_heads:
            raise ValueError("hidden_dim must be divisible by set_heads")
        probabilities = (
            self.dropout,
            self.element_mask_probability,
            self.head_mask_probability,
            self.layer_mask_probability,
        )
        if any(not 0.0 <= float(value) < 1.0 for value in probabilities):
            raise ValueError("dropout and mask probabilities must be in [0,1)")
        weights = (
            self.temporal_loss_weight,
            self.element_loss_weight,
            self.head_loss_weight,
            self.layer_loss_weight,
            self.variance_loss_weight,
        )
        if any(float(value) < 0.0 for value in weights):
            raise ValueError("loss weights must be non-negative")
        if not 0.0 < float(self.epsilon) < 1.0:
            raise ValueError("epsilon must be in (0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    """Label-free optimization and calibration controls."""

    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_accumulation: int = 4
    gradient_clip_norm: float = 1.0
    calibration_fraction: float = 0.25
    reference_per_sample: int = 8
    latent_trim_fraction: float = 0.90
    deterministic_masks: int = 4
    precision: str = "auto"
    profile_cuda_memory: bool = True
    seed: int = 20260818

    def validate(self) -> None:
        integers = (
            self.epochs,
            self.gradient_accumulation,
            self.reference_per_sample,
            self.deterministic_masks,
        )
        if min(integers) < 1:
            raise ValueError("training integer controls must be positive")
        if not 0.0 < float(self.learning_rate):
            raise ValueError("learning_rate must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 < float(self.gradient_clip_norm):
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 < float(self.calibration_fraction) < 1.0:
            raise ValueError("calibration_fraction must be in (0,1)")
        if not 0.5 <= float(self.latent_trim_fraction) <= 1.0:
            raise ValueError("latent_trim_fraction must be in [0.5,1]")
        if str(self.precision) not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("precision must be auto, bf16, fp16, or fp32")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")