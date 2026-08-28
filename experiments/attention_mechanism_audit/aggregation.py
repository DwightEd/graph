"""Preregistered answer-level aggregation of mechanism trajectories."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ANSWER_STATISTICS = (
    "mean",
    "early",
    "late",
    "late_minus_early",
    "max",
    "max_adjacent_drop",
)


def _finite_mean(value: np.ndarray, axis: int = 0) -> np.ndarray:
    count = np.isfinite(value).sum(axis=axis)
    total = np.where(np.isfinite(value), value, 0.0).sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=np.float64),
        where=count > 0,
    )


def _finite_max(value: np.ndarray, axis: int = 0) -> np.ndarray:
    valid = np.isfinite(value)
    result = np.where(valid, value, -np.inf).max(axis=axis)
    return np.where(valid.any(axis=axis), result, np.nan)


def aggregate_trajectory(
    values: np.ndarray,
    availability: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate one ``[response, ...]`` trajectory without zero imputation.

    Early and late windows are positional thirds of the complete response, so
    an unavailable first-token mechanism value does not shift temporal
    boundaries.  Availability is broadcast over feature dimensions and is
    represented as NaN before computing statistics.
    """

    value = np.asarray(values, dtype=np.float64)
    if value.ndim == 0:
        raise ValueError("a mechanism trajectory needs a response-token axis")
    response_count = value.shape[0]
    if availability is not None:
        mask = np.asarray(availability, dtype=bool)
        if mask.shape != (response_count,):
            raise ValueError("availability must have one boolean per response token")
        expand = mask.reshape((response_count,) + (1,) * (value.ndim - 1))
        value = np.where(expand, value, np.nan)

    third = max((response_count + 2) // 3, 1)
    early_values = value[: min(third, response_count)]
    late_values = value[max(response_count - third, 0) :]
    early = _finite_mean(early_values)
    late = _finite_mean(late_values)

    if response_count > 1:
        adjacent_valid = np.isfinite(value[:-1]) & np.isfinite(value[1:])
        drop = np.where(adjacent_valid, value[:-1] - value[1:], np.nan)
        max_drop = _finite_max(drop)
        adjacent_count = adjacent_valid.sum(axis=0)
    else:
        max_drop = np.full(value.shape[1:], np.nan, dtype=np.float64)
        adjacent_count = np.zeros(value.shape[1:], dtype=np.int64)

    finite = np.isfinite(value)
    return {
        "mean": _finite_mean(value),
        "early": early,
        "late": late,
        "late_minus_early": late - early,
        "max": _finite_max(value),
        "max_adjacent_drop": max_drop,
        "available_count": finite.sum(axis=0).astype(np.int64),
        "available_fraction": finite.mean(axis=0),
        "early_available_count": np.isfinite(early_values).sum(axis=0).astype(
            np.int64
        ),
        "late_available_count": np.isfinite(late_values).sum(axis=0).astype(
            np.int64
        ),
        "adjacent_available_count": adjacent_count.astype(np.int64),
        "response_count": np.asarray(response_count, dtype=np.int64),
    }


def aggregate_named_trajectories(
    trajectories: Mapping[str, np.ndarray],
    availability: Mapping[str, np.ndarray] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate named raw traces while preserving deterministic feature names."""

    output: dict[str, np.ndarray] = {}
    for name, value in trajectories.items():
        if name.endswith("_available") or name.endswith("_names"):
            continue
        if not np.issubdtype(np.asarray(value).dtype, np.number):
            continue
        if isinstance(availability, Mapping):
            mask = availability.get(name)
        else:
            mask = availability
        for statistic, result in aggregate_trajectory(value, mask).items():
            output[f"{name}__{statistic}"] = result
    return output


__all__ = [
    "ANSWER_STATISTICS",
    "aggregate_named_trajectories",
    "aggregate_trajectory",
]
