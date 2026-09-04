"""Claim-level re-anchor path summaries and high-flow backbone selection."""

from __future__ import annotations

import numpy as np

from .claims import ClaimSpan
from .potential import PathMass, conditioned_flow, first_hit_path_mass


def claim_metrics(
    transition,
    claim: ClaimSpan,
    response_start: int,
    evidence_mask,
    *,
    anchor_width: int = 3,
    reread_window: int = 5,
) -> dict[str, float]:
    graph = np.asarray(transition, dtype=np.float64)
    evidence = np.flatnonzero(np.asarray(evidence_mask, dtype=bool)[:response_start])
    boundary = claim.start - 1
    sink = claim.stop - 2
    response = np.arange(response_start, boundary, dtype=np.int64)
    anchors = np.arange(claim.start, min(sink, claim.start + anchor_width))
    global_valid = sink > claim.start and len(anchors) > 0
    missing = PathMass(float("nan"), float("nan"), float("nan"))
    evidence_path = (
        first_hit_path_mass(graph, sink, evidence, anchors) if global_valid else missing
    )
    response_path = (
        first_hit_path_mass(graph, sink, response, anchors) if global_valid else missing
    )
    direct = float(graph[evidence, sink].sum()) if len(evidence) and sink >= 0 else 0.0
    query_span = np.arange(boundary, sink + 1)
    bag = (
        float(graph[np.ix_(evidence, query_span)].sum(axis=0).mean())
        if len(evidence) and len(query_span)
        else 0.0
    )
    boundary_inflow = (
        float(graph[evidence, boundary].sum()) if len(evidence) and boundary >= 0 else 0.0
    )
    earlier = np.arange(max(response_start - 1, boundary - reread_window), boundary)
    previous = [
        float(graph[evidence, target].sum()) for target in earlier
    ] if len(evidence) else []
    baseline = float(np.median(previous)) if previous else 0.0
    if global_valid:
        _, node_flow = conditioned_flow(graph, sink, evidence)
        throughput = float(node_flow[anchors].max())
    else:
        throughput = float("nan")
    return {
        "evidence_reach": evidence_path.total,
        "evidence_reanchor_flow": evidence_path.through_anchor,
        "evidence_closure": evidence_path.closure,
        "response_reach": response_path.total,
        "response_reanchor_flow": response_path.through_anchor,
        "response_closure": response_path.closure,
        "direct_evidence_sink": direct,
        "bag_evidence_claim": bag,
        "boundary_evidence_inflow": boundary_inflow,
        "reread_pulse": boundary_inflow - baseline,
        "anchor_throughput": throughput,
    }


def dominant_backbone(
    transition,
    sink: int,
    sources,
    *,
    cover: float = 0.8,
    max_edges: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    edge_flow, _ = conditioned_flow(transition, sink, sources)
    source, target = np.nonzero(edge_flow > 0)
    mask = np.zeros_like(edge_flow, dtype=bool)
    if not len(source):
        return mask, edge_flow
    value = edge_flow[source, target]
    total = float(value.sum())
    retained = 0.0
    for position in np.argsort(-value, kind="stable")[:max_edges]:
        mask[source[position], target[position]] = True
        retained += float(value[position])
        if total > 0 and retained / total >= cover:
            break
    return mask, edge_flow
