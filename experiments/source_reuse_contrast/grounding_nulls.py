"""Strict relation-, position-, and origin-matched endpoint controls."""

from __future__ import annotations

import math

import torch

from .data import SourceReuseGraph
from .grounding_config import GroundingGraphConfig


def _position_bin(
    graph: SourceReuseGraph,
    *,
    token: int,
    source: int,
    bins: int,
) -> int:
    if source < graph.response_idx:
        return min(source * bins // max(graph.response_idx, 1), bins - 1)
    lag = token - (source - graph.response_idx)
    return min(int(math.floor(math.log2(max(lag, 1)))), bins - 1)


def matched_endpoint_rewire(
    graph: SourceReuseGraph,
    *,
    token: int,
    sources: torch.Tensor,
    pair_origin: torch.Tensor,
    all_source_state: torch.Tensor,
    source_origin: torch.Tensor,
    config: GroundingGraphConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rewire exact source identity without changing role/coarse position/origin."""

    if sources.numel() == 0:
        changed = torch.empty(0, dtype=torch.bool, device=graph.device)
        return sources, changed
    current = set(sources.tolist())
    used: set[int] = set()
    rewired = sources.clone()
    changed = torch.zeros_like(sources, dtype=torch.bool)
    state_norm = all_source_state.norm(dim=-1)

    for row, source_tensor in enumerate(sources):
        source = int(source_tensor)
        prompt = source < graph.response_idx
        if prompt:
            domain = range(graph.response_idx)
        else:
            domain = range(graph.response_idx, graph.response_idx + token)
        expected_position = _position_bin(
            graph,
            token=token,
            source=source,
            bins=config.response_lag_bins,
        )
        expected_origin = int(
            torch.floor(pair_origin[row] * 4.0).clamp(0, 3).item()
        )
        candidates: list[tuple[float, int]] = []
        for candidate in domain:
            if candidate == source or candidate in current or candidate in used:
                continue
            if (candidate < graph.response_idx) != prompt:
                continue
            if _position_bin(
                graph,
                token=token,
                source=candidate,
                bins=config.response_lag_bins,
            ) != expected_position:
                continue
            candidate_origin = int(
                torch.floor(source_origin[candidate] * 4.0)
                .clamp(0, 3)
                .item()
            )
            if candidate_origin != expected_origin:
                continue
            distance = float(
                (state_norm[candidate] - state_norm[source]).abs()
            )
            candidates.append((distance, candidate))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        pool = candidates[: min(8, len(candidates))]
        selected_index = int(
            torch.randint(
                len(pool),
                (1,),
                generator=generator,
                device=graph.device,
            ).item()
        )
        replacement = pool[selected_index][1]
        rewired[row] = replacement
        changed[row] = True
        used.add(replacement)
    return rewired, changed
