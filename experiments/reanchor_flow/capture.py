"""One-model capture for claim re-read, global flow, and path interventions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from experiments.constraint_routing_rhythm.intervene import (
    RelayGate,
    baseline_forward,
    rerun_gate,
)

from .claims import ClaimSpan, split_claims
from .flow import claim_metrics, dominant_backbone
from .routes import RouteAccumulator
from .graph import (
    TokenDAG,
    build_token_dag,
    capacity_bag,
    matched_endpoint_mask,
    rewire_by_role_lag,
    role_inflow,
    token_edges_to_query_mask,
)


@dataclass(frozen=True)
class SampleCapture:
    arrays: dict[str, object]
    functional_transition: np.ndarray
    attention_transition: np.ndarray
    claims: tuple[ClaimSpan, ...]


def metric_table(graph, claims, response_start, evidence_mask) -> dict[str, np.ndarray]:
    names = None
    rows = []
    for claim in claims:
        row = claim_metrics(
            graph.transition,
            claim,
            response_start,
            evidence_mask,
        )
        names = tuple(row)
        rows.append([row[name] for name in names])
    if names is None:
        return {}
    matrix = np.asarray(rows, dtype=np.float64)
    return {name: matrix[:, index] for index, name in enumerate(names)}


def all_layer_gate(edge_mask: np.ndarray, layer_count: int) -> RelayGate:
    query_mask = torch.as_tensor(token_edges_to_query_mask(edge_mask), dtype=torch.bool)
    empty = torch.zeros_like(query_mask)
    return RelayGate(
        upstream_edges=query_mask,
        downstream_edges=empty,
        split_layer=layer_count,
        cut_evidence=False,
        cut_upstream=True,
        cut_downstream=False,
        evidence_mask=torch.zeros(query_mask.shape[0], dtype=torch.bool),
    )


def audit_delta(model, cache, edge_mask: np.ndarray, event: int) -> float:
    if not np.asarray(edge_mask, dtype=bool).any():
        return float("nan")
    delta = rerun_gate(model, cache, all_layer_gate(edge_mask, cache.layer_count))
    return float(delta[event])


def audit_paths(
    model,
    cache,
    functional,
    attention,
    claims,
    response_start: int,
    evidence_mask,
    *,
    cover: float,
    max_edges: int,
) -> dict[str, object]:
    evidence = np.flatnonzero(np.asarray(evidence_mask, dtype=bool)[:response_start])
    candidates = []
    for index, claim in enumerate(claims):
        value = claim_metrics(
            functional.transition, claim, response_start, evidence_mask
        )["evidence_reanchor_flow"]
        if np.isfinite(value) and value > 0:
            candidates.append((float(value), index, claim))
    if not candidates:
        return {"audit_claim_index": -1}

    _, index, claim = max(candidates, key=lambda item: (item[0], -item[1]))
    functional_backbone, _ = dominant_backbone(
        functional.transition,
        claim.sink - 1,
        evidence,
        cover=cover,
        max_edges=max_edges,
    )
    attention_backbone, _ = dominant_backbone(
        attention.transition,
        claim.sink - 1,
        evidence,
        cover=cover,
        max_edges=max_edges,
    )
    count = int(functional_backbone.sum())
    bag = capacity_bag(functional.transition, claim.sink - 1, count)
    matched = matched_endpoint_mask(
        functional_backbone,
        functional.transition,
        response_start,
        evidence_mask,
    )
    event = claim.stop - response_start - 1
    return {
        "audit_claim_index": index,
        "audit_claim_start": claim.start,
        "audit_claim_stop": claim.stop,
        "functional_backbone_edges": count,
        "attention_backbone_edges": int(attention_backbone.sum()),
        "bag_edges": int(bag.sum()),
        "matched_edges": int(matched.sum()),
        "functional_backbone_cut_delta": audit_delta(
            model, cache, functional_backbone, event
        ),
        "attention_backbone_cut_delta": audit_delta(
            model, cache, attention_backbone, event
        ),
        "capacity_bag_cut_delta": audit_delta(model, cache, bag, event),
        "matched_endpoint_cut_delta": audit_delta(model, cache, matched, event),
    }


def capture_sample(
    model,
    tokenizer,
    token_ids,
    response_start: int,
    prompt_evidence_mask,
    *,
    sample_id: str,
    source_id: str,
    task_type: str,
    model_id: str,
    audit: bool = False,
    query_chunk: int = 128,
    layer_start: int | None = None,
    layer_stop: int | None = None,
    min_claim_tokens: int = 1,
    max_claim_tokens: int = 96,
    backbone_cover: float = 0.8,
    backbone_edges: int = 32,
) -> SampleCapture:
    """Use one frozen model for route observation, global flow, and path cuts."""

    ids = torch.as_tensor(token_ids, dtype=torch.long).cpu()
    source_count = len(ids) - 1
    evidence = np.zeros(source_count, dtype=bool)
    prompt_evidence = np.asarray(prompt_evidence_mask, dtype=bool)
    if len(prompt_evidence) != response_start:
        raise ValueError("evidence mask must cover the complete prompt")
    evidence[:response_start] = prompt_evidence

    accumulator = RouteAccumulator(
        model,
        response_start,
        query_chunk=query_chunk,
        layer_start=layer_start,
        layer_stop=layer_stop,
    )
    cache = baseline_forward(
        model,
        ids,
        response_start,
        observer=accumulator.observe,
    )
    maps = accumulator.finish()
    claims = tuple(
        split_claims(
            tokenizer,
            ids,
            response_start,
            min_tokens=min_claim_tokens,
            max_tokens=max_claim_tokens,
        )
    )
    functional = build_token_dag(maps.functional.numpy(), response_start)
    attention = build_token_dag(maps.attention.numpy(), response_start)
    seed = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")
    rewired_base = build_token_dag(maps.functional.numpy(), response_start)
    rewired = TokenDAG(
        capacity=rewired_base.capacity,
        transition=rewire_by_role_lag(
            rewired_base.transition,
            response_start,
            evidence,
            seed=seed,
        ),
        response_start=rewired_base.response_start,
        row_start=rewired_base.row_start,
    )

    functional_claim = metric_table(functional, claims, response_start, evidence)
    attention_claim = metric_table(attention, claims, response_start, evidence)
    rewired_claim = metric_table(rewired, claims, response_start, evidence)
    functional_inflow = role_inflow(functional.transition, response_start, evidence)
    attention_inflow = role_inflow(attention.transition, response_start, evidence)

    arrays: dict[str, object] = {
        "sample_id": sample_id,
        "source_id": source_id,
        "task_type": task_type,
        "model_id": model_id,
        "response_start": response_start,
        "query_position": cache.query,
        "prediction_position": cache.query + 1,
        "target_token_id": cache.target,
        "baseline_margin": cache.full_margin,
        "baseline_target_logprob": cache.baseline_target_logprob,
        "baseline_entropy": cache.baseline_entropy,
        "flow_layer_start": maps.layer_start,
        "flow_layer_stop": maps.layer_stop,
        "claim_start": np.asarray([claim.start for claim in claims], dtype=np.int64),
        "claim_stop": np.asarray([claim.stop for claim in claims], dtype=np.int64),
        "claim_sink_query": np.asarray([claim.stop - 2 for claim in claims], dtype=np.int64),
    }
    for name, value in functional_inflow.items():
        arrays[f"functional_{name}_inflow"] = value
    for name, value in attention_inflow.items():
        arrays[f"attention_{name}_inflow"] = value
    for prefix, table in (
        ("functional", functional_claim),
        ("attention", attention_claim),
        ("rewired", rewired_claim),
    ):
        for name, value in table.items():
            arrays[f"{prefix}_{name}"] = value

    audit_result = {"audit_claim_index": -1}
    if audit:
        audit_result = audit_paths(
            model,
            cache,
            functional,
            attention,
            claims,
            response_start,
            evidence,
            cover=backbone_cover,
            max_edges=backbone_edges,
        )
    arrays.update(audit_result)
    return SampleCapture(
        arrays=arrays,
        functional_transition=functional.transition,
        attention_transition=attention.transition,
        claims=claims,
    )
