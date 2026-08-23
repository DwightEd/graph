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


def _source_count_max_error(
    original_keys: np.ndarray,
    rewired_keys: np.ndarray,
) -> int:
    original_keys.sort()
    rewired_keys.sort()
    if np.array_equal(original_keys, rewired_keys):
        return 0

    original_index = 0
    rewired_index = 0
    max_error = 0
    while original_index < len(original_keys) or rewired_index < len(rewired_keys):
        if rewired_index == len(rewired_keys) or (
            original_index < len(original_keys)
            and original_keys[original_index] < rewired_keys[rewired_index]
        ):
            key = original_keys[original_index]
        else:
            key = rewired_keys[rewired_index]
        original_end = int(np.searchsorted(original_keys, key, side="right"))
        rewired_end = int(np.searchsorted(rewired_keys, key, side="right"))
        max_error = max(
            max_error,
            abs((original_end - original_index) - (rewired_end - rewired_index)),
        )
        original_index = original_end
        rewired_index = rewired_end
    return max_error


def _stratified_source_count_max_error(
    original: np.ndarray,
    rewired: np.ndarray,
    response_edges: np.ndarray,
    group_slices: tuple[slice, ...],
) -> int:
    max_error = 0
    for group_slice in group_slices:
        members = response_edges[group_slice]
        error = _source_count_max_error(
            original[members],
            rewired[members],
        )
        max_error = max(max_error, error)
    return max_error


class EndpointSwapPlan:
    """Reuse invariant CPU edge geometry across endpoint-null replicates."""

    def __init__(self, edges: RoutingEdges, *, lag_bins: int = 8):
        self.edges = edges
        self.lag_bins = lag_bins
        edge_count = len(edges.source)
        self.rows = np.empty((edge_count, 3), dtype=np.int32)
        for column, values in enumerate((edges.layer, edges.head, edges.query)):
            self.rows[:, column] = values.cpu().numpy()
        self.original = np.empty(edge_count, dtype=np.int32)
        self.original[:] = edges.source.cpu().numpy()
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
            self.response_lag = lag[order]
            sorted_group = group[order]
            boundary = np.flatnonzero(np.diff(sorted_group)) + 1
            starts = np.concatenate(([0], boundary))
            ends = np.concatenate((boundary, [len(sorted_group)]))
            self.group_slices = tuple(
                slice(int(start), int(end)) for start, end in zip(starts, ends)
            )
        else:
            self.response_lag = np.empty(0, dtype=np.int16)
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
            current_keys = self.key_prefix[response_edges]
            current_keys += source[response_edges]
            current_keys.sort()
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
            del current_keys
            if not paired:
                break

        keys = self.key_prefix[response_edges]
        keys += source[response_edges]
        keys.sort()
        duplicate_edges = int(np.count_nonzero(keys[1:] == keys[:-1]))
        del keys
        rewired_degree = np.bincount(source, minlength=edges.num_tokens)
        rewired_lag = _lag_bins(
            rows[response_edges, 2],
            source[response_edges],
            edges.response_idx,
            self.lag_bins,
        )
        coarse_lag_violations = int(np.count_nonzero(rewired_lag != self.response_lag))
        stratified_source_count_max_error = _stratified_source_count_max_error(
            self.original,
            source,
            response_edges,
            self.group_slices,
        )
        del rewired_lag
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
            "stratified_source_count_max_error": stratified_source_count_max_error,
            "eligible_response_edges": len(response_edges),
            "changed_response_edges": changed_count,
            "causal_violations": int(np.sum(source >= edges.response_idx + rows[:, 2])),
            "coarse_lag_violations": coarse_lag_violations,
            "duplicate_edges": duplicate_edges,
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
