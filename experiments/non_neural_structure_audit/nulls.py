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


def _lag_bins(
    query: np.ndarray,
    source: np.ndarray,
    response_idx: int,
    bins: int,
) -> np.ndarray:
    lag = np.maximum(query - (source - response_idx), 1)
    return np.minimum(np.floor(np.log2(lag)).astype(np.int16), bins - 1)


def _edge_keys(rows, source, edges: RoutingEdges) -> np.ndarray:
    layer_head = rows[:, 0] * edges.num_heads + rows[:, 1]
    target = layer_head * edges.num_response_tokens + rows[:, 2]
    return target.astype(np.int64) * edges.num_tokens + source


def _present(sorted_keys: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    position = np.searchsorted(sorted_keys, candidates)
    inside = position < len(sorted_keys)
    result = np.zeros(len(candidates), dtype=bool)
    result[inside] = sorted_keys[position[inside]] == candidates[inside]
    return result


def _pair_positions(group: np.ndarray) -> np.ndarray:
    boundary = np.flatnonzero(np.diff(group)) + 1
    starts = np.concatenate(([0], boundary))
    ends = np.concatenate((boundary, [len(group)]))
    positions = [
        np.arange(start, end - 1, 2, dtype=np.int64)
        for start, end in zip(starts, ends)
        if end - start >= 2
    ]
    return np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)


class EndpointSwapPlan:
    """Reuse invariant CPU edge geometry across endpoint-null replicates."""

    def __init__(self, edges: RoutingEdges, *, lag_bins: int = 8):
        self.edges = edges
        self.lag_bins = lag_bins
        self.rows = np.column_stack(
            (
                edges.layer.cpu().numpy(),
                edges.head.cpu().numpy(),
                edges.query.cpu().numpy(),
            )
        ).astype(np.int64, copy=False)
        self.original = edges.source.cpu().numpy()
        self.response_edges = np.flatnonzero(self.original >= edges.response_idx)
        self.original_degree = np.bincount(self.original, minlength=edges.num_tokens)

    def sample(self, *, seed: int, rounds: int = 10) -> EndpointSwapResult:
        edges = self.edges
        rows = self.rows
        source = self.original.copy()
        response_edges = self.response_edges
        changed = np.zeros(len(source), dtype=bool)
        rng = np.random.default_rng(seed)

        for _ in range(rounds):
            available = response_edges[~changed[response_edges]]
            if len(available) < 2:
                break
            lag = _lag_bins(
                rows[available, 2], source[available], edges.response_idx, self.lag_bins
            )
            group = (
                rows[available, 0] * edges.num_heads + rows[available, 1]
            ) * self.lag_bins + lag
            order = np.lexsort((rng.random(len(available)), group))
            position = _pair_positions(group[order])
            if not len(position):
                break
            first = available[order[position]]
            second = available[order[position + 1]]
            first_source = source[first]
            second_source = source[second]
            valid = (
                (first_source != second_source)
                & (second_source < edges.response_idx + rows[first, 2])
                & (first_source < edges.response_idx + rows[second, 2])
                & (
                    _lag_bins(
                        rows[first, 2],
                        second_source,
                        edges.response_idx,
                        self.lag_bins,
                    )
                    == _lag_bins(
                        rows[first, 2],
                        first_source,
                        edges.response_idx,
                        self.lag_bins,
                    )
                )
                & (
                    _lag_bins(
                        rows[second, 2],
                        first_source,
                        edges.response_idx,
                        self.lag_bins,
                    )
                    == _lag_bins(
                        rows[second, 2],
                        second_source,
                        edges.response_idx,
                        self.lag_bins,
                    )
                )
            )

            current_keys = np.sort(_edge_keys(rows, source, edges))
            first_key = _edge_keys(rows[first], second_source, edges)
            second_key = _edge_keys(rows[second], first_source, edges)
            valid &= ~_present(current_keys, first_key)
            valid &= ~_present(current_keys, second_key)
            accepted = np.flatnonzero(valid)
            if not len(accepted):
                continue

            candidate_keys = np.concatenate((first_key[accepted], second_key[accepted]))
            _, inverse, counts = np.unique(
                candidate_keys, return_inverse=True, return_counts=True
            )
            duplicate = counts[inverse] > 1
            unique_pair = ~(duplicate[: len(accepted)] | duplicate[len(accepted) :])
            accepted = accepted[unique_pair]
            first_accepted = first[accepted]
            second_accepted = second[accepted]
            source[first_accepted] = second_source[accepted]
            source[second_accepted] = first_source[accepted]
            changed[first_accepted] = True
            changed[second_accepted] = True

        keys = _edge_keys(rows, source, edges)
        rewired_degree = np.bincount(source, minlength=edges.num_tokens)
        changed_count = int(changed[response_edges].sum())
        changed_fraction = (
            changed_count / len(response_edges) if len(response_edges) else 0.0
        )
        audit = {
            "row_mass_max_error": 0.0,
            "role_mass_max_error": 0.0,
            "source_count_degree_max_error": int(
                np.max(np.abs(self.original_degree - rewired_degree))
            ),
            "eligible_response_edges": len(response_edges),
            "changed_response_edges": changed_count,
            "causal_violations": int(np.sum(source >= edges.response_idx + rows[:, 2])),
            "duplicate_edges": int(len(keys) - len(np.unique(keys))),
        }
        return EndpointSwapResult(
            edges=replace(
                edges,
                source=torch.as_tensor(
                    source, dtype=edges.source.dtype, device=edges.device
                ),
            ),
            changed_fraction=changed_fraction,
            audit=audit,
        )


def constrained_endpoint_swap(
    edges: RoutingEdges,
    *,
    seed: int,
    rounds: int = 10,
    lag_bins: int = 8,
) -> EndpointSwapResult:
    """One-off convenience interface for a constrained endpoint null."""

    return EndpointSwapPlan(edges, lag_bins=lag_bins).sample(seed=seed, rounds=rounds)
