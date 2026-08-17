"""Deterministic topology controls for causal multiplex events."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import torch

from .causal_events import CausalMultiplexEvents, RR, log_lag_bin


def _stable_uint64(*parts) -> int:
    digest = hashlib.sha256(
        "\0".join(map(str, parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def event_target(events: CausalMultiplexEvents, edge_index: int) -> int:
    edge_index = int(edge_index)
    if not 0 <= edge_index < events.num_events:
        raise IndexError("event index is outside the selected event list")
    return int(
        torch.searchsorted(
            events.target_ptr[1:],
            torch.as_tensor(edge_index, device=events.target_ptr.device),
            right=True,
        ).item()
    )


def lag_preserving_alternative(
    events: CausalMultiplexEvents,
    edge_index: int,
    *,
    seed: int,
) -> tuple[int, bool]:
    """Choose a different legal source in the same coarse log2-lag bin."""
    edge_index = int(edge_index)
    if int(events.relation[edge_index]) != RR:
        raise ValueError("source rewiring is defined only for RR events")
    target = event_target(events, edge_index)
    true_source = int(events.source[edge_index])
    if not 0 <= true_source < target:
        raise ValueError("RR source is outside the causal prefix")
    target_bin = log_lag_bin(target - true_source)
    candidates = [
        source
        for source in range(target)
        if source != true_source and log_lag_bin(target - source) == target_bin
    ]
    if not candidates:
        return true_source, False
    chosen = min(
        candidates,
        key=lambda source: _stable_uint64(
            "lag-preserving-rewire-v2",
            int(seed),
            events.sample_id,
            edge_index,
            target,
            int(events.layer[edge_index]),
            int(events.head[edge_index]),
            true_source,
            source,
        ),
    )
    return int(chosen), True


def rewire_causal_sources(
    events: CausalMultiplexEvents,
    *,
    seed: int,
) -> tuple[CausalMultiplexEvents, torch.Tensor]:
    """Rewire eligible RR sources while preserving typed edge marginals.

    Target, layer, head, relation, retained weight and coarse lag bin remain
    fixed. ``changed`` is edge-aligned. Early-prefix events without a legal
    same-bin alternative remain unchanged.
    """
    source = events.source.clone()
    lag = events.lag.clone()
    changed = torch.zeros(
        events.num_events,
        dtype=torch.bool,
        device=events.source.device,
    )
    rr_indices = torch.nonzero(events.relation == RR, as_tuple=False).flatten()
    for edge_index in rr_indices.tolist():
        alternative, available = lag_preserving_alternative(
            events, edge_index, seed=seed
        )
        if not available:
            continue
        target = event_target(events, edge_index)
        source[edge_index] = int(alternative)
        lag[edge_index] = int(target - alternative)
        changed[edge_index] = True
    return replace(events, source=source, lag=lag).validate(), changed


def token_rewire_mask(
    events: CausalMultiplexEvents,
    changed: torch.Tensor,
) -> torch.Tensor:
    """Return one boolean per response token with at least one rewired edge."""
    if changed.shape != (events.num_events,):
        raise ValueError("changed mask has the wrong shape")
    result = torch.zeros(
        events.response_count,
        dtype=torch.bool,
        device=changed.device,
    )
    if bool(changed.any()):
        result[events.target_index()[changed]] = True
    return result
