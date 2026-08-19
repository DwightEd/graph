"""Attention-space tests of detection, fracture, integration, and lock-in."""

from .config import PhenomenologyConfig
from .features import SamplePhenomenology, analyze_routing
from .routing import RoutingEdges, RoutingTensor, build_routing_tensor, collect_routing_edges

__all__ = [
    "PhenomenologyConfig",
    "RoutingEdges",
    "RoutingTensor",
    "SamplePhenomenology",
    "collect_routing_edges",
    "build_routing_tensor",
    "analyze_routing",
]
