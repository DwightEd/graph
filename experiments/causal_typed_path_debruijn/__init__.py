"""Causal typed-path and De Bruijn hallucination-routing primitives."""

from .change_lockin import (
    ChangeLockinResult,
    MedianMAD,
    RobustChangeStats,
    change_lockin_score,
    fit_median_mad,
    fit_robust_change_stats,
    prompt_lineage_drop,
)
from .config import (
    CalibrationConfig,
    ChangeConfig,
    DeBruijnConfig,
    GraphConfig,
    PathConfig,
)
from .debruijn import DeBruijnAccumulator, FrozenDeBruijn
from .graph_builder import (
    RELATION_NAMES,
    ROLE_NAMES,
    RP,
    RR_FAR,
    RR_NEAR,
    CausalRoutingGraph,
    OvershootAudit,
    build_causal_routing_graph,
)
from .layered_automaton import (
    P0,
    P_PLUS,
    R0,
    R_PLUS,
    STATE_NAMES,
    U,
    LayeredAutomatonResult,
    layered_attention_automaton,
)
from .nulls import (
    RewireResult,
    causal_endpoint_rewire,
    offline_noncausal_bucket_time_shuffle,
    rewire_exact_endpoints,
)
from .typed_path_dp import (
    ROUTE_NAMES,
    SINK_NAMES,
    TypedPathResult,
    typed_path_dp,
)

__all__ = [
    "CalibrationConfig",
    "CausalRoutingGraph",
    "ChangeConfig",
    "ChangeLockinResult",
    "DeBruijnAccumulator",
    "DeBruijnConfig",
    "FrozenDeBruijn",
    "GraphConfig",
    "LayeredAutomatonResult",
    "MedianMAD",
    "OvershootAudit",
    "P0",
    "P_PLUS",
    "PathConfig",
    "RELATION_NAMES",
    "ROLE_NAMES",
    "ROUTE_NAMES",
    "R0",
    "RP",
    "RR_FAR",
    "RR_NEAR",
    "R_PLUS",
    "RewireResult",
    "RobustChangeStats",
    "SINK_NAMES",
    "STATE_NAMES",
    "TypedPathResult",
    "U",
    "build_causal_routing_graph",
    "causal_endpoint_rewire",
    "change_lockin_score",
    "fit_median_mad",
    "fit_robust_change_stats",
    "layered_attention_automaton",
    "offline_noncausal_bucket_time_shuffle",
    "prompt_lineage_drop",
    "rewire_exact_endpoints",
    "typed_path_dp",
]
