"""Prompt-grounded routing-attractor audit for generated tokens."""

from .attractor import CONTROL_FEATURE_NAMES, PRIMARY_FEATURE_NAMES
from .extractor import (
    TopologyDynamicsConfig,
    TopologyDynamicsExtractor,
)
from .spectral_diagnostics import load_rr_reference

__all__ = [
    "CONTROL_FEATURE_NAMES",
    "PRIMARY_FEATURE_NAMES",
    "TopologyDynamicsConfig",
    "TopologyDynamicsExtractor",
    "load_rr_reference",
]
