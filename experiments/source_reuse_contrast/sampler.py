"""Strict hard-negative candidates for masked source prediction."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from .config import SourceReuseConfig
from .data import PROMPT, RESPONSE, SourceReuseGraph


@dataclass(frozen=True)
class CandidateBatch:
    """One true source and several matched alternatives per source pair."""

    true_source: torch.Tensor
    candidate_source: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_distance: torch.Tensor
    pool_size: torch.Tensor
    valid: torch.Tensor

    @property
    def pair_count(self) -> int:
        return int(self.true_source.numel())


def relation_of(graph: SourceReuseGraph, source: int) -> int:
    return PROMPT if source < graph.response_idx else RESPONSE


def prompt_position_bin(graph: SourceReuseGraph, source: int, bins: int) -> int:
    return min(source * bins // max(graph.response_idx, 1), bins - 1)


def response_lag(graph: SourceReuseGraph, query: int, source: int) -> int:
    return query - (source - graph.response_idx)


def response_lag_bin(
    graph: SourceReuseGraph,
    query: int,
    source: int,
    bins: int,
) -> int:
    lag = response_lag(graph, query, source)
    return min(int(math.floor(math.log2(max(lag, 1)))), bins - 1)


def source_bin(
    graph: SourceReuseGraph,
    query: int,
    source: int,
    config: SourceReuseConfig,
) -> int:
    if source < graph.response_idx:
        return prompt_position_bin(graph, source, config.prompt_position_bins)
    return response_lag_bin(graph, query, source, config.response_lag_bins)


def usage_bucket(count: int, bins: int) -> int:
    return min(int(math.floor(math.log2(count + 1))), bins - 1)


def _last_gap(query: int, last_used: int) -> float:
    return float(query + 1 if last_used < 0 else query - last_used)


def _position_is_matched(
    graph: SourceReuseGraph,
    *,
    query: int,
    source: int,
    candidate: int,
    config: SourceReuseConfig,
) -> bool:
    if source_bin(graph, query, source, config) != source_bin(
        graph, query, candidate, config
    ):
        return False
    if source < graph.response_idx:
        denominator = float(max(graph.response_idx - 1, 1))
        distance = abs(source - candidate) / denominator
        return distance <= config.prompt_position_tolerance
    source_lag = response_lag(graph, query, source)
    candidate_lag = response_lag(graph, query, candidate)
    relative = abs(source_lag - candidate_lag) / float(max(source_lag, 1))
    return relative <= config.response_lag_tolerance


def _candidate_distance(
    *,
    source: int,
    candidate: int,
    query: int,
    use_count: torch.Tensor,
    cumulative_mass: torch.Tensor,
    last_used: torch.Tensor,
    memory_norm: torch.Tensor,
) -> float:
    usage = abs(
        math.log1p(int(use_count[source]))
        - math.log1p(int(use_count[candidate]))
    )
    mass = abs(
        math.log1p(float(cumulative_mass[source]))
        - math.log1p(float(cumulative_mass[candidate]))
    )
    gap = abs(
        math.log1p(_last_gap(query, int(last_used[source])))
        - math.log1p(_last_gap(query, int(last_used[candidate])))
    )
    norm = abs(
        math.log1p(float(memory_norm[source]))
        - math.log1p(float(memory_norm[candidate]))
    )
    return usage + mass + gap + norm


def matched_candidate_batch(
    graph: SourceReuseGraph,
    *,
    query: int,
    true_sources: torch.Tensor,
    use_count: torch.Tensor,
    cumulative_mass: torch.Tensor,
    last_used: torch.Tensor,
    memory_norm: torch.Tensor,
    config: SourceReuseConfig,
    rng: random.Random,
) -> CandidateBatch:
    """Construct strict, unique, hard alternatives without silent fallback."""

    pair_count = int(true_sources.numel())
    width = config.negative_count + 1
    candidate_source = torch.full(
        (pair_count, width),
        -1,
        dtype=torch.long,
        device=graph.device,
    )
    candidate_mask = torch.zeros(
        (pair_count, width), dtype=torch.bool, device=graph.device
    )
    candidate_distance = torch.full(
        (pair_count, width),
        float("inf"),
        dtype=torch.float32,
        device=graph.device,
    )
    pool_size = torch.zeros(pair_count, dtype=torch.long, device=graph.device)
    valid = torch.zeros(pair_count, dtype=torch.bool, device=graph.device)
    current_set = set(int(value) for value in true_sources.tolist())

    for row, source_tensor in enumerate(true_sources):
        source = int(source_tensor)
        relation = relation_of(graph, source)
        if relation == PROMPT:
            domain = range(graph.response_idx)
        else:
            domain = range(graph.response_idx, graph.response_idx + query)

        source_usage = usage_bucket(int(use_count[source]), config.usage_bins)
        choices: list[tuple[float, int]] = []
        for candidate in domain:
            if candidate == source or candidate in current_set:
                continue
            if relation_of(graph, candidate) != relation:
                continue
            if not _position_is_matched(
                graph,
                query=query,
                source=source,
                candidate=candidate,
                config=config,
            ):
                continue
            if usage_bucket(int(use_count[candidate]), config.usage_bins) != source_usage:
                continue
            distance = _candidate_distance(
                source=source,
                candidate=candidate,
                query=query,
                use_count=use_count,
                cumulative_mass=cumulative_mass,
                last_used=last_used,
                memory_norm=memory_norm,
            )
            choices.append((distance, candidate))

        choices.sort(key=lambda item: item[0])
        pool = choices[: config.negative_pool_size]
        pool_size[row] = len(pool)
        candidate_source[row, 0] = source
        candidate_mask[row, 0] = True
        candidate_distance[row, 0] = 0.0
        if len(pool) < config.negative_count:
            continue

        selected = rng.sample(pool, config.negative_count)
        for column, (distance, candidate) in enumerate(selected, start=1):
            candidate_source[row, column] = candidate
            candidate_mask[row, column] = True
            candidate_distance[row, column] = float(distance)
        valid[row] = True

    return CandidateBatch(
        true_source=true_sources,
        candidate_source=candidate_source,
        candidate_mask=candidate_mask,
        candidate_distance=candidate_distance,
        pool_size=pool_size,
        valid=valid,
    )
