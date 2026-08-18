"""Configuration objects for Mechanism-Guided Causal Attention Set-Flow."""

from __future__ import annotations

from dataclasses import dataclass


CORRUPTION_NAMES = (
    "collapse",
    "localize",
    "freeze",
    "homogenize",
    "self_reinforce",
)


@dataclass(frozen=True)
class SourceSetConfig:
    """Exact sparse RR source-set extraction controls.

    `materialize_query_chunk_size` changes only the execution schedule.  The
    returned route and received-memory sets are identical to the dense
    definition up to floating-point summation roundoff.
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
        if min(map(int, integers)) < 1:
            raise ValueError("source-set integer controls must be positive")
        if not 0.5 <= float(self.route_mass_coverage) <= 1.0:
            raise ValueError("route_mass_coverage must be in [0.5,1]")
        if not 0.0 < float(self.epsilon) < 1.0:
            raise ValueError("epsilon must be in (0,1)")


@dataclass(frozen=True)
class SetFlowModelConfig:
    """Set-Flow encoder and learned energy-head controls."""

    hidden_dim: int = 64
    scalar_fourier_dim: int = 16
    set_heads: int = 4
    induced_points: int = 8
    set_blocks: int = 2
    head_mixer_layers: int = 2
    depth_mixer_layers: int = 2
    energy_hidden_multiplier: int = 2
    projector_hidden_multiplier: int = 2
    set_row_chunk_size: int = 4096
    mixer_token_chunk_size: int = 512
    activation_checkpointing: bool = True
    dropout: float = 0.10
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
            self.energy_hidden_multiplier,
            self.projector_hidden_multiplier,
            self.set_row_chunk_size,
            self.mixer_token_chunk_size,
        )
        if min(map(int, integers)) < 1:
            raise ValueError("model integer controls must be positive")
        if int(self.hidden_dim) % int(self.set_heads):
            raise ValueError("hidden_dim must be divisible by set_heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not 0.0 < float(self.epsilon) < 1.0:
            raise ValueError("epsilon must be in (0,1)")


@dataclass(frozen=True)
class CorruptionConfig:
    """Label-free mechanism-guided corruption controls."""

    token_span_min: int = 4
    token_span_max: int = 24
    layer_span_min: int = 4
    layer_span_max: int = 12
    selected_head_fraction: float = 0.50
    collapse_power: float = 4.0
    self_reinforce_power: float = 2.0
    locality_window: int = 4
    margin: float = 1.0
    clean_keep_fraction: float = 0.90
    epsilon: float = 1e-8

    def validate(self) -> None:
        integers = (
            self.token_span_min,
            self.token_span_max,
            self.layer_span_min,
            self.layer_span_max,
            self.locality_window,
        )
        if min(map(int, integers)) < 1:
            raise ValueError("corruption integer controls must be positive")
        if int(self.token_span_min) > int(self.token_span_max):
            raise ValueError("token span minimum exceeds maximum")
        if int(self.layer_span_min) > int(self.layer_span_max):
            raise ValueError("layer span minimum exceeds maximum")
        if not 0.0 < float(self.selected_head_fraction) <= 1.0:
            raise ValueError("selected_head_fraction must be in (0,1]")
        if float(self.collapse_power) <= 1.0:
            raise ValueError("collapse_power must be greater than one")
        if float(self.self_reinforce_power) <= 0.0:
            raise ValueError("self_reinforce_power must be positive")
        if float(self.margin) <= 0.0:
            raise ValueError("corruption margin must be positive")
        if not 0.5 <= float(self.clean_keep_fraction) <= 1.0:
            raise ValueError("clean_keep_fraction must be in [0.5,1]")
        if not 0.0 < float(self.epsilon) < 1.0:
            raise ValueError("epsilon must be in (0,1)")


@dataclass(frozen=True)
class TrainingConfig:
    """Label-free optimization and EMA controls."""

    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_accumulation: int = 2
    gradient_clip_norm: float = 1.0
    calibration_fraction: float = 0.25
    ema_momentum: float = 0.996
    clean_energy_weight: float = 1.0
    corrupt_energy_weight: float = 1.0
    ranking_weight: float = 1.0
    type_weight: float = 0.50
    clean_recovery_weight: float = 1.0
    context_recovery_weight: float = 0.50
    variance_weight: float = 1.0
    covariance_weight: float = 0.04
    precision: str = "auto"
    profile_cuda_memory: bool = True
    seed: int = 20260818

    def validate(self) -> None:
        integers = (self.epochs, self.gradient_accumulation)
        if min(map(int, integers)) < 1:
            raise ValueError("training integer controls must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.gradient_clip_norm) <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 < float(self.calibration_fraction) < 1.0:
            raise ValueError("calibration_fraction must be in (0,1)")
        if not 0.0 < float(self.ema_momentum) < 1.0:
            raise ValueError("ema_momentum must be in (0,1)")
        weights = (
            self.clean_energy_weight,
            self.corrupt_energy_weight,
            self.ranking_weight,
            self.type_weight,
            self.clean_recovery_weight,
            self.context_recovery_weight,
            self.variance_weight,
            self.covariance_weight,
        )
        if any(float(value) < 0.0 for value in weights):
            raise ValueError("training loss weights must be non-negative")
        if str(self.precision) not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("precision must be auto, bf16, fp16, or fp32")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")