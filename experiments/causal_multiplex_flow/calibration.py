"""Leakage-resistant group splitting and monotone score calibration for CMRP."""

from __future__ import annotations

import numpy as np

from experiment_protocol import partition_source_groups


def split_source_groups(
    dataset,
    *,
    calibration_fraction: float,
    seed: int,
    limit=None,
):
    """Split complete source groups into disjoint fit/calibration sample IDs."""
    sample_ids = list(map(str, dataset.sample_ids))
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        sample_ids = sample_ids[:limit]
    if len(sample_ids) < 2:
        raise ValueError("CMRP needs at least two selected train samples")
    return partition_source_groups(
        dataset,
        sample_ids,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )


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
