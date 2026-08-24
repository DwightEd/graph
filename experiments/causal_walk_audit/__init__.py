"""Typed route-grammar hallucination detector."""

from .automaton import AutomatonTrace, STATE_NAMES, run_typed_automaton
from .config import AuditConfig
from .grammar import GrammarAccumulator, RouteGrammar
from .graph import RoutingGraph, build_routing_graph

__all__ = [
    "AuditConfig",
    "AutomatonTrace",
    "GrammarAccumulator",
    "RouteGrammar",
    "RoutingGraph",
    "STATE_NAMES",
    "build_routing_graph",
    "run_typed_automaton",
]
