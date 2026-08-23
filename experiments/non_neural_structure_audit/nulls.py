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


def _edge_keys(prefix: np.ndarray, source: np.ndarray) -> np.ndarray:
    return prefix + source


def _present(sorted_keys: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    position = np.searchsorted(sorted_keys, candidates)
    inside = position < len(sorted_keys)
    result = np.zeros(len(candidates), dtype=bool)
    result[inside] = sorted_keys[position[inside]] == candidates[inside]
    return result


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
        ).astype(np.int32, copy=False)
        self.original = edges.source.cpu().numpy().astype(np.int32, copy=False)
        self.response_edges = np.flatnonzero(
            self.original >= edges.response_idx
        ).astype(np.int32, copy=False)
        if len(self.response_edges):
            lag = _lag_bins(
                self.rows[self.response_edges, 2],
                self.original[self.response_edges],
                edges.response_idx,
                self.lag_bins,
            )
            group = (
                self.rows[self.response_edges, 0] * edges.num_heads
                + self.rows[self.response_edges, 1]
            ) * self.lag_bins + lag
            order = np.argsort(group, kind="stable")
            self.response_edges = self.response_edges[order]
            sorted_group = group[order]
            boundary = np.flatnonzero(np.diff(sorted_group)) + 1
            starts = np.concatenate(([0], boundary))
            ends = np.concatenate((boundary, [len(sorted_group)]))
            self.group_slices = tuple(
                slice(int(start), int(end)) for start, end in zip(starts, ends)
            )
        else:
            self.group_slices = ()
        self.key_prefix = (
            (self.rows[:, 0].astype(np.int64) * edges.num_heads + self.rows[:, 1])
            * edges.num_response_tokens
            + self.rows[:, 2]
        ) * edges.num_tokens
        self.original_degree = np.bincount(self.original, minlength=edges.num_tokens)

    def sample(self, *, seed: int, rounds: int = 10) -> EndpointSwapResult:
        edges = self.edges
        rows = self.rows
        source = self.original.copy()
        response_edges = self.response_edges
        changed = np.zeros(len(source), dtype=bool)
        rng = np.random.default_rng(seed)

        for _ in range(rounds):
            current_keys = np.sort(_edge_keys(self.key_prefix, source))
            paired = False
            for group_slice in self.group_slices:
                members = response_edges[group_slice]
                available = members[~changed[members]]
                if len(available) < 2:
                    continue
                paired = True
                rng.shuffle(available)
                pair_stop = len(available) - len(available) % 2
                first = available[:pair_stop:2]
                second = available[1:pair_stop:2]
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

                first_key = _edge_keys(self.key_prefix[first], second_source)
                second_key = _edge_keys(self.key_prefix[second], first_source)
                valid &= ~_present(current_keys, first_key)
                valid &= ~_present(current_keys, second_key)
                accepted = np.flatnonzero(valid)
                if not len(accepted):
                    continue

                candidate_keys = np.concatenate(
                    (first_key[accepted], second_key[accepted])
                )
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
            if not paired:
                break

        keys = _edge_keys(self.key_prefix, source)
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
