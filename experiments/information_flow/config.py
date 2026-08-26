"""Configuration for the attention-only information-flow audit."""

from dataclasses import asdict, dataclass


VIEW_NAMES = (
    "full_trace",
    "full_final",
    "reverse_trace",
    "reverse_final",
    "last_layer",
    "layer_mean",
    "identity_trace",
    "identity_final",
)


@dataclass(frozen=True)
class FlowConfig:
    sketch_dim: int = 32
    residual_weight: float = 1.0
    unresolved: str = "self"
    calibration_fraction: float = 0.20
    seed: int = 20260827

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
