"""Attention-space mechanism audit inspired by The Phenomenology of Hallucinations."""

from .config import FEATURE_NAMES, FAMILY_NAMES, PhenomenologyConfig
from .features import SamplePhenomenology, analyze_routing
from .routing import collect_routing_edges, rewire_exact_endpoints

__all__ = [
    "FEATURE_NAMES",
    "FAMILY_NAMES",
    "PhenomenologyConfig",
    "SamplePhenomenology",
    "analyze_routing",
    "collect_routing_edges",
    "rewire_exact_endpoints",
]
