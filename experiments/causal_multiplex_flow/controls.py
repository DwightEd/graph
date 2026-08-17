"""Deterministic source candidates and topology counterfactuals for CMRP."""

from __future__ import annotations

import hashlib

import torch

from .events import CausalEventSample, RR, log_lag_bin


def _stable_uint64(*parts) -> int:
    digest = hashlib.sha256(
        "\0".join(map(str, parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _ordered_without_replacement(values: list[int], *, key_parts) -> list[int]:
    return sorted(
        values,
        key=lambda value: _stable_uint64(*key_parts, int(value)),
    )


def event_target(events: CausalEventSample, edge_index: int) -> int:
    """Recover the response-relative target token of one selected event."""
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


def lag_preserving_rewired_source(
    events: CausalEventSample,
    edge_index: int,
    *,
    seed: int,
) -> int:
    """Choose a different legal prior source in the same coarse lag bin.

    If the lag bin contains no legal alternative, return the true source.  This
    keeps the topology gate honest: fallback negatives remain useful for source
    prediction, but they are not mislabeled as lag-preserving rewires.
    """
    edge_index = int(edge_index)
    if int(events.relation[edge_index]) != RR:
        raise ValueError("source rewiring is defined only for RR events")
    token = event_target(events, edge_index)
    true_source = int(events.source[edge_index].item())
    if not 0 <= true_source < token:
        raise ValueError("RR event source is outside the causal prefix")
    target_bin = log_lag_bin(token - true_source)
    same_bin = [
        source
        for source in range(token)
        if source != true_source and log_lag_bin(token - source) == target_bin
    ]
    if not same_bin:
        return true_source
    ordered = _ordered_without_replacement(
        same_bin,
        key_parts=(
            "cmrp-rewire-v1",
            seed,
            events.sample_id,
            token,
            int(events.channel[edge_index]),
            true_source,
        ),
    )
    return int(ordered[0])


def source_candidates(
    events: CausalEventSample,
    edge_index: int,
    *,
    negatives: int,
    seed: int,
) -> torch.Tensor:
    """Return true source first, then deterministic hard/fallback negatives."""
    negatives = int(negatives)
    if negatives < 1:
        raise ValueError("negatives must be positive")
    edge_index = int(edge_index)
    if int(events.relation[edge_index]) != RR:
        raise ValueError("source candidates are defined only for RR events")
    token = event_target(events, edge_index)
    true_source = int(events.source[edge_index].item())
    target_bin = log_lag_bin(token - true_source)
    same_bin = [
        source
        for source in range(token)
        if source != true_source and log_lag_bin(token - source) == target_bin
    ]
    same_bin_set = set(same_bin)
    fallback = [
        source
        for source in range(token)
        if source != true_source and source not in same_bin_set
    ]
    key = (
        "cmrp-candidates-v1",
        seed,
        events.sample_id,
        token,
        int(events.channel[edge_index]),
        true_source,
    )
    ordered = _ordered_without_replacement(same_bin, key_parts=key + ("same",))
    ordered.extend(
        _ordered_without_replacement(fallback, key_parts=key + ("fallback",))
    )
    values = [true_source, *ordered[:negatives]]
    return torch.as_tensor(
        values,
        dtype=torch.long,
        device=events.source.device,
    )


def first_lag_preserving_candidate(
    events: CausalEventSample,
    edge_index: int,
    candidates: torch.Tensor,
) -> tuple[int, bool]:
    """Return the first same-lag-bin non-true candidate, if one exists."""
    if candidates.ndim != 1 or len(candidates) < 1:
        raise ValueError("candidate source tensor must be non-empty and one-dimensional")
    token = event_target(events, edge_index)
    true_source = int(candidates[0])
    true_bin = log_lag_bin(token - true_source)
    for index in range(1, len(candidates)):
        source = int(candidates[index])
        if source != true_source and log_lag_bin(token - source) == true_bin:
            return index, True
    return 0, False
