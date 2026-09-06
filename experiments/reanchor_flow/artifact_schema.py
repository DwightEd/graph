"""Versioned schema coordinates for native mechanism-audit artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from .route_plan import RouteBudget

AUDIT_SCHEMA = 3
METHOD_VERSION = "budgeted_head_resolved_route_audit_v3"
ROUTE_EVENT_LIMIT = 32


@dataclass(frozen=True)
class NativeAuditMetadata:
    """Frozen run coordinates that are not part of the mechanism tensors."""

    dataset_sample_id: str
    source_id: str
    split: str
    task_type: str
    generator_model: str
    model_id: str
    model_dtype: str
    target_policy: str
    target_rank: int
    coverage: float
    carrier_scope: str
    query_chunk: int
    route_budget: RouteBudget
    local_window: int
    saved_edges: int


__all__ = [
    "AUDIT_SCHEMA",
    "METHOD_VERSION",
    "ROUTE_EVENT_LIMIT",
    "NativeAuditMetadata",
]
