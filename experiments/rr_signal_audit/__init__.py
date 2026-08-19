"""Evidence-grounded PR/RR attention mechanism audit."""

from .components import (
    COLLAPSE_DIRECTIONS,
    COLLAPSE_FEATURE_NAMES,
    EVIDENCE_DIRECTIONS,
    EVIDENCE_FEATURE_NAMES,
    EVIDENCE_REGISTRY,
    RRSignalConfig,
    RRSignalFeatures,
    extract_rr_signal_features,
)
from .geometry import RRGeometryConfig

__all__ = [
    "COLLAPSE_DIRECTIONS",
    "COLLAPSE_FEATURE_NAMES",
    "EVIDENCE_DIRECTIONS",
    "EVIDENCE_FEATURE_NAMES",
    "EVIDENCE_REGISTRY",
    "RRGeometryConfig",
    "RRSignalConfig",
    "RRSignalFeatures",
    "extract_rr_signal_features",
]
