"""Topology-destroying controls for exact attention source endpoints."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from .config import PhenomenologyConfig
from .routing import RoutingEdges


@dataclass(frozen=True)
class RewireResult:
    edges: RoutingEdges
    changed_fraction: float


def _prompt_candidates(prompt_count: int, bin_index: int, bins: int) -> np.ndarray:
    start = (bin_index * prompt_count) // bins
    stop = max(((bin_index + 1) * prompt_count) // bins, start + 1)
    return np.arange(start, min(stop, prompt_count), dtype=np.int64)


def _response_candidates(query: int, bin_index: int, bins: int) -> np.ndarray:
    source = np.arange(query, dtype=np.int64)
    lag = query - source
    current_bin = np.minimum(np.floor(np.log2(lag)).astype(np.int64), bins - 1)
    return source[current_bin == bin_index]


def rewire_exact_endpoints(
    edges: RoutingEdges,
    *,
    config: PhenomenologyConfig | None = None,
    seed: int | None = None,
) -> RewireResult:
    """Resample exact endpoints while preserving every coarse routing role."""

    config = PhenomenologyConfig() if config is None else config
    if not edges.weight.numel():
        return RewireResult(edges=edges, changed_fraction=0.0)

    rng = np.random.default_rng(config.random_seed if seed is None else seed)
    layer = edges.layer.cpu().numpy()
    head = edges.head.cpu().numpy()
    query = edges.query.cpu().numpy()
    source = edges.source.cpu().numpy()
    rewired = source.copy()

    groups: dict[tuple[int, int, int, int, int], list[int]] = {}
    for index, (current_layer, current_head, current_query, current_source) in enumerate(
        zip(layer, head, query, source)
    ):
        if current_source < edges.response_idx:
            edge_type = 0
            bin_index = min(
                current_source * config.prompt_bins // max(edges.response_idx, 1),
                config.prompt_bins - 1,
            )
        else:
            edge_type = 1
            lag = current_query - (current_source - edges.response_idx)
            bin_index = min(int(np.floor(np.log2(lag))), config.rr_lag_bins - 1)
        groups.setdefault(
            (current_layer, current_head, current_query, edge_type, bin_index), []
        ).append(index)

    for (_, _, current_query, edge_type, bin_index), indices in groups.items():
        candidates = (
            _prompt_candidates(edges.response_idx, bin_index, config.prompt_bins)
            if edge_type == 0
            else _response_candidates(current_query, bin_index, config.rr_lag_bins)
            + edges.response_idx
        )
        if len(candidates) <= 1 or len(candidates) < len(indices):
            continue
        chosen = rng.choice(candidates, size=len(indices), replace=False)
        original = source[np.asarray(indices)]
        if np.array_equal(chosen, original):
            chosen = np.roll(chosen, 1)
        rewired[np.asarray(indices)] = chosen

    changed_fraction = float(np.mean(rewired != source))
    rewired_edges = replace(
        edges,
        source=torch.as_tensor(rewired, dtype=edges.source.dtype, device=edges.device),
    )
    return RewireResult(edges=rewired_edges, changed_fraction=changed_fraction)
