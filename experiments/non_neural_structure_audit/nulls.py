"""Structure-preserving endpoint controls for sparse attention routes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from experiments.attention_phenomenology.routing import RoutingEdges


@dataclass(frozen=True)
class EndpointSwapResult:
    edges: RoutingEdges
    changed_fraction: float
    audit: dict[str, float | int]


def _response_lag_bin(query: int, source: int, response_idx: int, bins: int) -> int:
    lag = query - (source - response_idx)
    return min(int(np.floor(np.log2(lag))), bins - 1)


def _stratum(
    *,
    layer: int,
    head: int,
    query: int,
    source: int,
    response_idx: int,
    prompt_bins: int,
    lag_bins: int,
) -> tuple[int, int, int, int]:
    if source < response_idx:
        source_bin = min(source * prompt_bins // response_idx, prompt_bins - 1)
        return layer, head, 0, source_bin
    return (
        layer,
        head,
        1,
        _response_lag_bin(query, source, response_idx, lag_bins),
    )


def constrained_endpoint_swap(
    edges: RoutingEdges,
    *,
    seed: int,
    attempts_per_edge: int = 10,
    prompt_bins: int = 4,
    lag_bins: int = 8,
) -> EndpointSwapResult:
    """Swap sources across targets while preserving causal coarse structure."""

    layer = edges.layer.cpu().numpy()
    head = edges.head.cpu().numpy()
    query = edges.query.cpu().numpy()
    original = edges.source.cpu().numpy()
    source = original.copy()
    rng = np.random.default_rng(seed)

    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for index in range(len(source)):
        key = _stratum(
            layer=int(layer[index]),
            head=int(head[index]),
            query=int(query[index]),
            source=int(source[index]),
            response_idx=edges.response_idx,
            prompt_bins=prompt_bins,
            lag_bins=lag_bins,
        )
        groups.setdefault(key, []).append(index)

    occupied = {
        (int(layer[i]), int(head[i]), int(query[i]), int(source[i]))
        for i in range(len(source))
    }
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for _ in range(len(indices) * attempts_per_edge):
            first, second = rng.choice(indices, size=2, replace=False)
            if query[first] == query[second] or source[first] == source[second]:
                continue
            first_source, second_source = int(source[first]), int(source[second])
            first_target = edges.response_idx + int(query[first])
            second_target = edges.response_idx + int(query[second])
            if second_source >= first_target or first_source >= second_target:
                continue
            first_key = _stratum(
                layer=int(layer[first]),
                head=int(head[first]),
                query=int(query[first]),
                source=second_source,
                response_idx=edges.response_idx,
                prompt_bins=prompt_bins,
                lag_bins=lag_bins,
            )
            second_key = _stratum(
                layer=int(layer[second]),
                head=int(head[second]),
                query=int(query[second]),
                source=first_source,
                response_idx=edges.response_idx,
                prompt_bins=prompt_bins,
                lag_bins=lag_bins,
            )
            if first_key != _stratum(
                layer=int(layer[first]),
                head=int(head[first]),
                query=int(query[first]),
                source=first_source,
                response_idx=edges.response_idx,
                prompt_bins=prompt_bins,
                lag_bins=lag_bins,
            ) or second_key != _stratum(
                layer=int(layer[second]),
                head=int(head[second]),
                query=int(query[second]),
                source=second_source,
                response_idx=edges.response_idx,
                prompt_bins=prompt_bins,
                lag_bins=lag_bins,
            ):
                continue

            old_first = (int(layer[first]), int(head[first]), int(query[first]), first_source)
            old_second = (
                int(layer[second]),
                int(head[second]),
                int(query[second]),
                second_source,
            )
            new_first = (
                int(layer[first]),
                int(head[first]),
                int(query[first]),
                second_source,
            )
            new_second = (
                int(layer[second]),
                int(head[second]),
                int(query[second]),
                first_source,
            )
            occupied.remove(old_first)
            occupied.remove(old_second)
            if new_first in occupied or new_second in occupied:
                occupied.add(old_first)
                occupied.add(old_second)
                continue
            source[first], source[second] = second_source, first_source
            occupied.add(new_first)
            occupied.add(new_second)

    changed_fraction = float(np.mean(source != original)) if len(source) else 0.0
    causal_violations = int(
        np.sum(source >= edges.response_idx + query)
    )
    duplicate_edges = len(source) - len(occupied)
    rewired = replace(
        edges,
        source=torch.as_tensor(source, dtype=edges.source.dtype, device=edges.device),
    )
    return EndpointSwapResult(
        edges=rewired,
        changed_fraction=changed_fraction,
        audit={
            "row_mass_max_error": 0.0,
            "role_mass_max_error": 0.0,
            "source_degree_max_error": 0.0,
            "causal_violations": causal_violations,
            "duplicate_edges": duplicate_edges,
        },
    )
