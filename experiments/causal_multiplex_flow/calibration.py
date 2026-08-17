"""Leakage-resistant group splitting and monotone score calibration for CMRP."""

from __future__ import annotations

import hashlib

import numpy as np

from experiment_protocol import canonical_source_group


def _stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(
        f"cmrp-group-split-v1\0{int(seed)}\0{value}".encode("utf-8")
    ).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64)


def sample_group(sample) -> str:
    return canonical_source_group(sample)


def split_source_groups(
    dataset,
    *,
    calibration_fraction: float,
    seed: int,
    limit=None,
):
    """Split complete source groups into disjoint fit/calibration sample IDs."""
    calibration_fraction = float(calibration_fraction)
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0,1)")
    sample_ids = list(map(str, dataset.sample_ids))
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        sample_ids = sample_ids[:limit]
    if len(sample_ids) < 2:
        raise ValueError("CMRP needs at least two selected train samples")

    groups: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            groups.setdefault(sample_group(sample), []).append(sample_id)
        finally:
            sample.release_attention()
    if len(groups) < 2:
        raise ValueError("CMRP needs at least two complete source groups")

    calibration_groups = {
        group
        for group in groups
        if _stable_fraction(seed, group) < calibration_fraction
    }
    ordered = sorted(groups, key=lambda value: _stable_fraction(seed, value))
    if not calibration_groups:
        calibration_groups.add(ordered[0])
    if len(calibration_groups) == len(groups):
        calibration_groups.remove(ordered[-1])
    fit_groups = set(groups).difference(calibration_groups)
    if not fit_groups or not calibration_groups or fit_groups & calibration_groups:
        raise RuntimeError("invalid CMRP fit/calibration group split")

    fit_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_group(dataset[sample_id]) in fit_groups
    ]
    calibration_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_group(dataset[sample_id]) in calibration_groups
    ]
    # The group lookup above opens lightweight sample metadata only.  Explicitly
    # release any cached attention handle in case a dataset implementation
    # materialized one while resolving the sample.
    for sample_id in sample_ids:
        dataset[sample_id].release_attention()
    if not fit_ids or not calibration_ids:
        raise RuntimeError("CMRP source-group split produced an empty stream")
    return {
        "fit_sample_ids": tuple(fit_ids),
        "calibration_sample_ids": tuple(calibration_ids),
        "fit_group_ids": tuple(sorted(fit_groups)),
        "calibration_group_ids": tuple(sorted(calibration_groups)),
    }


def finite_reference(values, *, minimum: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < int(minimum):
        raise ValueError("calibration reference has too few finite values")
    return np.sort(values)


def empirical_upper_tail(reference, values, *, epsilon: float = 1e-12) -> np.ndarray:
    """Finite-sample monotone ``-log P(reference >= value)`` transform."""
    reference = finite_reference(reference)
    values = np.asarray(values, dtype=np.float64)
    index = np.searchsorted(reference, values, side="left")
    count = len(reference) - index
    probability = (count + 1.0) / (len(reference) + 1.0)
    return -np.log(np.maximum(probability, float(epsilon)))


def topology_gate_summary(rewire_edge_gap, *, selected_edge_count: int) -> dict:
    """Summarize the preregistered true-versus-rewired edge comparison."""

    gap = np.asarray(rewire_edge_gap, dtype=np.float64)
    gap = gap[np.isfinite(gap)]
    selected_edge_count = int(selected_edge_count)
    if len(gap) == 0:
        return {
            "evaluated_edge_count": 0,
            "selected_edge_count": selected_edge_count,
            "coverage": 0.0,
            "mean_gap": None,
            "median_gap": None,
            "positive_fraction": None,
            "pass": False,
        }
    mean = float(gap.mean())
    return {
        "evaluated_edge_count": int(len(gap)),
        "selected_edge_count": selected_edge_count,
        "coverage": float(len(gap) / selected_edge_count),
        "mean_gap": mean,
        "median_gap": float(np.median(gap)),
        "positive_fraction": float(np.mean(gap > 0.0)),
        "pass": bool(mean > 0.0),
    }
