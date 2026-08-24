"""Deterministic topology/time nulls for causal routing controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import torch

from .graph_builder import CausalRoutingGraph, RP, RR_FAR, RR_NEAR


def _log2_bin(lag: int) -> int:
    if lag < 1:
        raise ValueError("causal lag must be positive")
    return int(lag).bit_length() - 1


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _relation_for_source(graph: CausalRoutingGraph, target: int, source: int) -> int:
    if source < graph.response_idx:
        return RP
    return RR_NEAR if target - source <= graph.recent_lag else RR_FAR


def _candidates(
    graph: CausalRoutingGraph,
    *,
    target: int,
    relation: int,
    lag_bin: int,
) -> list[int]:
    lower, upper = (
        (0, graph.response_idx)
        if relation == RP
        else (graph.response_idx, target)
    )
    return [
        source
        for source in range(lower, upper)
        if _log2_bin(target - source) == lag_bin
        and _relation_for_source(graph, target, source) == relation
    ]


@dataclass(frozen=True)
class RewireResult:
    graph: CausalRoutingGraph
    changed_fraction: float
    eligible_fraction: float


@torch.no_grad()
def causal_endpoint_rewire(
    graph: CausalRoutingGraph,
    *,
    seed: int = 0,
) -> RewireResult:
    """Rewire exact sources within strict causal role/lag strata.

    Target, layer/head channel, corrected/raw weight, RP/RR role, near/far type,
    and coarse ``floor(log2(lag))`` are unchanged. Node role masses are reused
    verbatim. This destroys exact attention-provenance lineage without changing
    the validated first-order routing summaries.
    """

    graph.validate()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if graph.num_edges == 0:
        return RewireResult(graph=graph, changed_fraction=0.0, eligible_fraction=0.0)

    source = graph.source.detach().cpu().tolist()
    target = graph.target.detach().cpu().tolist()
    channel = graph.edge_channel.detach().cpu().tolist()
    relation = graph.relation.detach().cpu().tolist()
    rewired = list(source)
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for index, (current_source, current_target, current_channel, current_relation) in enumerate(
        zip(source, target, channel, relation, strict=True)
    ):
        key = (
            int(current_target),
            int(current_channel),
            int(current_relation),
            _log2_bin(int(current_target) - int(current_source)),
        )
        groups.setdefault(key, []).append(index)

    eligible = 0
    candidate_cache: dict[tuple[int, int, int], list[int]] = {}
    for (current_target, current_channel, current_relation, lag_bin), indices in groups.items():
        candidate_key = (current_target, current_relation, lag_bin)
        candidates = candidate_cache.get(candidate_key)
        if candidates is None:
            candidates = _candidates(
                graph,
                target=current_target,
                relation=current_relation,
                lag_bin=lag_bin,
            )
            candidate_cache[candidate_key] = candidates
        if len(candidates) <= 1 or len(candidates) < len(indices):
            continue
        eligible += len(indices)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_seed(
                seed,
                graph.sample_id,
                current_target,
                current_channel,
                current_relation,
                lag_bin,
            )
        )
        order = torch.randperm(len(candidates), generator=generator).tolist()
        chosen = [candidates[position] for position in order[: len(indices)]]
        original = [source[index] for index in indices]
        if chosen == original and len(candidates) > 1:
            if len(chosen) == 1:
                chosen = [next(value for value in candidates if value != original[0])]
            else:
                chosen = chosen[1:] + chosen[:1]
        for index, replacement in zip(indices, chosen, strict=True):
            rewired[index] = replacement

    rewired_source = torch.tensor(
        rewired, dtype=graph.source.dtype, device=graph.device
    )
    changed = int((rewired_source != graph.source).sum().item())
    rewired_graph = replace(graph, source=rewired_source).validate()
    return RewireResult(
        graph=rewired_graph,
        changed_fraction=changed / graph.num_edges,
        eligible_fraction=eligible / graph.num_edges,
    )


# A descriptive alias for callers familiar with the earlier control module.
rewire_exact_endpoints = causal_endpoint_rewire


@torch.no_grad()
def offline_noncausal_bucket_time_shuffle(
    q: torch.Tensor,
    *,
    bucket_size: int,
    seed: int,
    sample_id: str,
) -> torch.Tensor:
    """Offline-only shuffle of route rows inside fixed token-index buckets.

    The permutation is deterministic for ``(seed, sample_id, bucket)`` and
    never crosses a sample or a fixed index bucket. It can move a future row to
    an earlier position inside that bucket and therefore must never be used as
    a prefix-causal token/onset score. All channels/states in one token row move
    together, making this an offline order-destroying De Bruijn null while
    preserving the empirical row distribution.
    """

    if q.ndim < 2 or q.shape[0] < 1:
        raise ValueError("q must be a non-empty sequence tensor [R,...]")
    if isinstance(bucket_size, bool) or not isinstance(bucket_size, int) or bucket_size < 1:
        raise ValueError("bucket_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    output = q.clone()
    for start in range(0, int(q.shape[0]), bucket_size):
        stop = min(start + bucket_size, int(q.shape[0]))
        length = stop - start
        if length <= 1:
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(seed, sample_id, start, stop))
        permutation = torch.randperm(length, generator=generator, device="cpu")
        permutation = permutation.to(device=q.device) + start
        output[start:stop] = q.index_select(0, permutation)
    return output
