"""Deterministic exact-endpoint null for typed lineage."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import torch

from .graph import PROMPT, RESPONSE_FAR, RESPONSE_NEAR, RoutingGraph


def _lag_bin(lag: int) -> int:
    return int(lag).bit_length() - 1


def _seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _relation(graph: RoutingGraph, target: int, source: int) -> int:
    if source < graph.response_idx:
        return PROMPT
    return (
        RESPONSE_NEAR
        if target - source <= graph.recent_lag
        else RESPONSE_FAR
    )


@dataclass(frozen=True)
class RewireResult:
    graph: RoutingGraph
    changed_fraction: float
    eligible_fraction: float


@torch.no_grad()
def rewire_endpoints(
    graph: RoutingGraph,
    *,
    seed: int,
) -> RewireResult:
    """Change sources while preserving target, channel, role, lag bin and weight."""

    if not graph.weight.numel():
        return RewireResult(graph, 0.0, 0.0)

    source = graph.source.cpu().tolist()
    target = graph.target.cpu().tolist()
    channel = graph.channel.cpu().tolist()
    relation = graph.relation.cpu().tolist()
    rewired = list(source)

    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for index, values in enumerate(zip(source, target, channel, relation)):
        current_source, current_target, current_channel, current_relation = values
        key = (
            int(current_target),
            int(current_channel),
            int(current_relation),
            _lag_bin(int(current_target) - int(current_source)),
        )
        groups.setdefault(key, []).append(index)

    eligible = 0
    for (current_target, current_channel, current_relation, lag_bin), indices in groups.items():
        lower, upper = (
            (0, graph.response_idx)
            if current_relation == PROMPT
            else (graph.response_idx, current_target)
        )
        candidates = [
            candidate
            for candidate in range(lower, upper)
            if _lag_bin(current_target - candidate) == lag_bin
            and _relation(graph, current_target, candidate) == current_relation
        ]
        if len(candidates) <= 1 or len(candidates) < len(indices):
            continue
        eligible += len(indices)
        generator = torch.Generator().manual_seed(
            _seed(
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
        if chosen == original:
            chosen = chosen[1:] + chosen[:1] if len(chosen) > 1 else [
                next(value for value in candidates if value != original[0])
            ]
        for index, replacement in zip(indices, chosen):
            rewired[index] = replacement

    rewired_source = torch.tensor(
        rewired,
        dtype=graph.source.dtype,
        device=graph.device,
    )
    changed = int((rewired_source != graph.source).sum().item())
    return RewireResult(
        graph=replace(graph, source=rewired_source).validate(),
        changed_fraction=changed / max(len(source), 1),
        eligible_fraction=eligible / max(len(source), 1),
    )
