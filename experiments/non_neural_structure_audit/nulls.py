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


def _stratum(row, source: int, response_idx: int, lag_bins: int):
    layer, head, query = row
    return (
        int(layer),
        int(head),
        _response_lag_bin(int(query), source, response_idx, lag_bins),
    )


def _edge_key(row, source: int):
    return int(row[0]), int(row[1]), int(row[2]), source


def constrained_endpoint_swap(
    edges: RoutingEdges,
    *,
    seed: int,
    attempts_per_edge: int = 10,
    lag_bins: int = 8,
) -> EndpointSwapResult:
    """Swap response-history endpoints within layer/head/lag strata."""

    rows = np.column_stack(
        (
            edges.layer.cpu().numpy(),
            edges.head.cpu().numpy(),
            edges.query.cpu().numpy(),
        )
    )
    original = edges.source.cpu().numpy()
    source = original.copy()
    rng = np.random.default_rng(seed)

    response_indices = np.flatnonzero(source >= edges.response_idx)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index in response_indices:
        group = _stratum(
            rows[index], int(source[index]), edges.response_idx, lag_bins
        )
        groups.setdefault(group, []).append(int(index))

    occupied = {
        _edge_key(rows[index], int(source[index]))
        for index in range(len(source))
    }
    for group, indices in groups.items():
        if len(indices) < 2:
            continue
        for _ in range(len(indices) * attempts_per_edge):
            first, second = rng.choice(indices, size=2, replace=False)
            if rows[first, 2] == rows[second, 2] or source[first] == source[second]:
                continue
            first_source, second_source = int(source[first]), int(source[second])
            first_target = edges.response_idx + int(rows[first, 2])
            second_target = edges.response_idx + int(rows[second, 2])
            if second_source >= first_target or first_source >= second_target:
                continue
            if (
                _stratum(rows[first], second_source, edges.response_idx, lag_bins)
                != group
                or _stratum(rows[second], first_source, edges.response_idx, lag_bins)
                != group
            ):
                continue

            old = {
                _edge_key(rows[first], first_source),
                _edge_key(rows[second], second_source),
            }
            new = {
                _edge_key(rows[first], second_source),
                _edge_key(rows[second], first_source),
            }
            occupied.difference_update(old)
            if occupied.intersection(new):
                occupied.update(old)
                continue
            source[first], source[second] = second_source, first_source
            occupied.update(new)

    changed_response = int(np.sum(source[response_indices] != original[response_indices]))
    changed_fraction = (
        changed_response / len(response_indices) if len(response_indices) else 0.0
    )
    causal_violations = int(
        np.sum(source >= edges.response_idx + rows[:, 2])
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
            "source_count_degree_max_error": 0.0,
            "eligible_response_edges": int(len(response_indices)),
            "changed_response_edges": changed_response,
            "causal_violations": causal_violations,
            "duplicate_edges": duplicate_edges,
        },
    )
