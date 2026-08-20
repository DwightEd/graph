"""Attention-space tests of detection, fracture, integration, and lock-in."""

from .config import PhenomenologyConfig
from .features import SamplePhenomenology, analyze_routing
from .routing import RoutingEdges, RoutingState, build_routing_state, collect_routing_edges

__all__ = [
    "PhenomenologyConfig",
    "RoutingEdges",
    "RoutingState",
    "SamplePhenomenology",
    "collect_routing_edges",
    "build_routing_state",
    "analyze_routing",
]
