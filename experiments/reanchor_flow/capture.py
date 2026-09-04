"""One-model capture for re-read, global flow, and real path-cut validation."""

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
from .graph import (
    TokenDAG,
    build_token_dag,
    capacity_bag,
    matched_endpoint_mask,
    rewire_by_role_lag,
    role_inflow,
    token_edges_to_query_mask,
)
from .routes import RouteAccumulator


@dataclass(frozen=True)
class SampleCapture:
    arrays: dict[str, object]
    functional_transition: np.ndarray
    attention_transition: np.ndarray
    claims: tuple[ClaimSpan, ...]


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def full_evidence_mask(prompt_mask, token_count: int, response_start: int) -> np.ndarray:
    prompt = np.asarray(prompt_mask, dtype=bool).reshape(-1)
    if len(prompt) != response_start:
        raise ValueError("evidence mask must align with the complete prompt")
    evidence = np.zeros(token_count, dtype=bool)
    evidence[:response_start] = prompt
    return evidence


def metric_table(
    prefix: str,
    dag: TokenDAG,
    claims: tuple[ClaimSpan, ...],
    response_start: int,
    evidence: np.ndarray,
    anchor_width: int,
    reread_window: int,
) -> dict[str, np.ndarray]:
    rows = [
        claim_metrics(
            dag.transition,
            claim,
            response_start,
            evidence,
            anchor_width=anchor_width,
            reread_window=reread_window,
        )
        for claim in claims
    ]
    names = rows[0].keys() if rows else ()
    return {
        f"{prefix}_{name}": np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in names
    }


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


def path_cut_delta(model, cache, edge_mask: np.ndarray, event: int) -> float:
    if not np.asarray(edge_mask, dtype=bool).any():
        return float("nan")
    delta = rerun_gate(model, cache, all_layer_gate(edge_mask, cache.layer_count))
    return float(delta[event])


def path_audit(
    model,
    cache,
    functional: TokenDAG,
    attention: TokenDAG,
    claims: tuple[ClaimSpan, ...],
    response_start: int,
    evidence: np.ndarray,
    *,
    cover: float,
    max_edges: int,
) -> dict[str, object]:
    """Cut one label-free strongest evidence-to-claim backbone and controls."""

    sources = np.flatnonzero(evidence[:response_start])
    candidates = []
    for index, claim in enumerate(claims):
        value = claim_metrics(
            functional.transition, claim, response_start, evidence
        )["evidence_reanchor_flow"]
        if np.isfinite(value) and value > 0:
            candidates.append((float(value), index, claim))
    if not candidates or not len(sources):
        return {"audit_claim_index": -1}

    _, index, claim = max(candidates, key=lambda item: (item[0], -item[1]))
    functional_edges, edge_flow = dominant_backbone(
        functional.transition,
        claim.sink,
        sources,
        cover=cover,
        max_edges=max_edges,
    )
    attention_edges, _ = dominant_backbone(
        attention.transition,
        claim.sink,
        sources,
        cover=cover,
        max_edges=max_edges,
    )
    count = int(functional_edges.sum())
    bag_edges = capacity_bag(functional.transition, claim.sink, count)
    matched_edges = matched_endpoint_mask(
        functional_edges,
        functional.transition,
        response_start,
        evidence,
    )
    event = claim.sink - response_start
    edge_source, edge_target = np.nonzero(functional_edges)
    return {
        "audit_claim_index": index,
        "audit_claim_start": claim.start,
        "audit_claim_stop": claim.stop,
        "audit_functional_edge_count": count,
        "audit_attention_edge_count": int(attention_edges.sum()),
        "audit_bag_edge_count": int(bag_edges.sum()),
        "audit_matched_edge_count": int(matched_edges.sum()),
        "audit_functional_backbone_delta": path_cut_delta(
            model, cache, functional_edges, event
        ),
        "audit_attention_backbone_delta": path_cut_delta(
            model, cache, attention_edges, event
        ),
        "audit_capacity_bag_delta": path_cut_delta(
            model, cache, bag_edges, event
        ),
        "audit_matched_endpoint_delta": path_cut_delta(
            model, cache, matched_edges, event
        ),
        "audit_edge_source": edge_source.astype(np.int64),
        "audit_edge_target": edge_target.astype(np.int64),
        "audit_edge_flow": edge_flow[edge_source, edge_target],
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
    min_claim_tokens: int = 2,
    max_claim_tokens: int = 96,
    anchor_width: int = 3,
    reread_window: int = 5,
    backbone_cover: float = 0.8,
    backbone_edges: int = 32,
) -> SampleCapture:
    """Build all graph views and interventions from one frozen observer model."""

    ids = torch.as_tensor(token_ids, dtype=torch.long).cpu()
    evidence = full_evidence_mask(prompt_evidence_mask, len(ids), response_start)
    accumulator = RouteAccumulator(model, response_start, query_chunk=query_chunk)
    cache = baseline_forward(
        model,
        ids,
        response_start,
        observer=accumulator.observe,
        checkpoint_layers=(0,),
    )
    maps = accumulator.finish()
    claims = tuple(
        split_claims(
            tokenizer,
            ids.numpy(),
            response_start,
            min_tokens=min_claim_tokens,
            max_tokens=max_claim_tokens,
        )
    )

    # The all-layer endpoint graph matches the all-layer endpoint cuts below.
    functional = build_token_dag(maps.functional.numpy(), response_start)
    attention = build_token_dag(maps.attention.numpy(), response_start)
    middle = build_token_dag(maps.functional_middle.numpy(), response_start)
    rewired = TokenDAG(
        capacity=functional.capacity,
        transition=rewire_by_role_lag(
            functional.transition,
            response_start,
            evidence,
            seed=stable_seed(sample_id),
        ),
        response_start=response_start,
    )

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
        "middle_layer_start": maps.middle_start,
        "middle_layer_stop": maps.middle_stop,
        "claim_start": np.asarray([claim.start for claim in claims], dtype=np.int64),
        "claim_stop": np.asarray([claim.stop for claim in claims], dtype=np.int64),
        "claim_sink": np.asarray([claim.sink for claim in claims], dtype=np.int64),
        "audit_claim_index": -1,
        "audit_functional_backbone_delta": float("nan"),
        "audit_attention_backbone_delta": float("nan"),
        "audit_capacity_bag_delta": float("nan"),
        "audit_matched_endpoint_delta": float("nan"),
        "audit_functional_edge_count": 0,
        "audit_attention_edge_count": 0,
        "audit_bag_edge_count": 0,
        "audit_matched_edge_count": 0,
        "audit_edge_source": np.empty(0, dtype=np.int64),
        "audit_edge_target": np.empty(0, dtype=np.int64),
        "audit_edge_flow": np.empty(0, dtype=np.float64),
    }
    for name, value in functional_inflow.items():
        arrays[f"functional_{name}_inflow"] = value
    for name, value in attention_inflow.items():
        arrays[f"attention_{name}_inflow"] = value
    for prefix, dag in (
        ("functional", functional),
        ("attention", attention),
        ("middle", middle),
        ("rewired", rewired),
    ):
        arrays.update(
            metric_table(
                prefix,
                dag,
                claims,
                response_start,
                evidence,
                anchor_width,
                reread_window,
            )
        )
    if audit:
        arrays.update(
            path_audit(
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
        )
    return SampleCapture(
        arrays=arrays,
        functional_transition=functional.transition,
        attention_transition=attention.transition,
        claims=claims,
    )
