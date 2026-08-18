"""RR-only source-persistence decomposition and routing-collapse audit."""

from .components import (
    COLLAPSE_DIRECTIONS,
    COLLAPSE_FEATURE_NAMES,
    RRSignalConfig,
    RRSignalFeatures,
    extract_rr_signal_features,
)
from .geometry import RRGeometryConfig

__all__ = [
    "COLLAPSE_DIRECTIONS",
    "COLLAPSE_FEATURE_NAMES",
    "RRGeometryConfig",
    "RRSignalConfig",
    "RRSignalFeatures",
    "extract_rr_signal_features",
]
