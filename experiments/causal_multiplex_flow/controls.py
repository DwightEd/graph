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


def lag_preserving_rewired_source(
    events: CausalEventSample,
    edge_index: int,
    *,
    seed: int,
) -> int:
    """Choose a different legal prior source in the same coarse lag bin.

    If the bin contains no alternative source, fall back to any other prior
    source.  If the target has no alternative prior source at all, return the
    true source; callers should then mark the counterfactual unavailable.
    """
    edge_index = int(edge_index)
    if int(events.relation[edge_index]) != RR:
        raise ValueError("source rewiring is defined only for RR events")
    token = int(
        torch.searchsorted(
            events.target_ptr[1:],
            torch.as_tensor(edge_index, device=events.target_ptr.device),
            right=True,
        ).item()
    )
    true_source = int(events.source[edge_index].item())
    if not 0 <= true_source < token:
        raise ValueError("RR event source is outside the causal prefix")
    target_bin = log_lag_bin(token - true_source)
    same_bin = [
        source
        for source in range(token)
        if source != true_source and log_lag_bin(token - source) == target_bin
    ]
    candidates = same_bin or [
        source for source in range(token) if source != true_source
    ]
    if not candidates:
        return true_source
    ordered = _ordered_without_replacement(
        candidates,
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
    token = int(
        torch.searchsorted(
            events.target_ptr[1:],
            torch.as_tensor(edge_index, device=events.target_ptr.device),
            right=True,
        ).item()
    )
    true_source = int(events.source[edge_index].item())
    target_bin = log_lag_bin(token - true_source)
    same_bin = [
        source
        for source in range(token)
        if source != true_source and log_lag_bin(token - source) == target_bin
    ]
    fallback = [
        source
        for source in range(token)
        if source != true_source and source not in set(same_bin)
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
    chosen = ordered[:negatives]
    values = [true_source, *chosen]
    return torch.as_tensor(
        values,
        dtype=torch.long,
        device=events.source.device,
    )


def first_rewired_candidate(candidates: torch.Tensor) -> tuple[int, bool]:
    """Return the first non-true candidate index and its availability flag."""
    if candidates.ndim != 1 or len(candidates) < 1:
        raise ValueError("candidate source tensor must be non-empty and one-dimensional")
    true_source = int(candidates[0])
    for index in range(1, len(candidates)):
        if int(candidates[index]) != true_source:
            return index, True
    return 0, False
