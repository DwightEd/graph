"""Empirical conditional tails for preregistered RR-collapse variables."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .components import COLLAPSE_DIRECTIONS, COLLAPSE_FEATURE_NAMES


def _finite_reference(
    reference_values: np.ndarray,
    reference_conditions: np.ndarray,
    *,
    condition: str,
    column: int,
    minimum: int,
) -> np.ndarray:
    selected = (
        np.asarray(reference_conditions, dtype=str) == str(condition)
    )
    values = np.asarray(reference_values, dtype=np.float64)[selected, int(column)]
    values = values[np.isfinite(values)]
    if len(values) >= int(minimum):
        return np.sort(values)
    values = np.asarray(reference_values, dtype=np.float64)[:, int(column)]
    return np.sort(values[np.isfinite(values)])


def _tail_probability(reference: np.ndarray, values: np.ndarray, direction: int):
    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(reference) < 2:
        raise ValueError("collapse calibration requires at least two finite rows")
    if int(direction) > 0:
        count = len(reference) - np.searchsorted(reference, values, side="left")
    elif int(direction) < 0:
        count = np.searchsorted(reference, values, side="right")
    else:
        lower = np.searchsorted(reference, values, side="right")
        upper = len(reference) - np.searchsorted(reference, values, side="left")
        count = np.minimum(lower, upper) * 2
        count = np.minimum(count, len(reference))
    return (count + 1.0) / (len(reference) + 1.0)


def collapse_scores(
    reference_values: np.ndarray,
    reference_conditions: np.ndarray,
    values: np.ndarray,
    conditions: np.ndarray,
    *,
    minimum_condition_rows: int,
    epsilon: float = 1e-12,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return fixed-direction, two-sided, and composite collapse scores."""

    reference_values = np.asarray(reference_values, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    reference_conditions = np.asarray(reference_conditions, dtype=str)
    conditions = np.asarray(conditions, dtype=str)
    if (
        reference_values.ndim != 2
        or values.ndim != 2
        or reference_values.shape[1] != len(COLLAPSE_FEATURE_NAMES)
        or values.shape[1] != len(COLLAPSE_FEATURE_NAMES)
        or len(reference_values) != len(reference_conditions)
        or len(values) != len(conditions)
    ):
        raise ValueError("collapse score inputs do not match the feature contract")

    directed = np.zeros_like(values, dtype=np.float32)
    two_sided = np.zeros_like(values, dtype=np.float32)
    for condition in np.unique(conditions):
        rows = np.flatnonzero(conditions == condition)
        for column, direction in enumerate(COLLAPSE_DIRECTIONS):
            reference = _finite_reference(
                reference_values,
                reference_conditions,
                condition=str(condition),
                column=column,
                minimum=minimum_condition_rows,
            )
            current = values[rows, column]
            two_probability = _tail_probability(reference, current, 0)
            two_sided[rows, column] = -np.log(
                np.maximum(two_probability, epsilon)
            ).astype(np.float32)
            if int(direction) != 0:
                probability = _tail_probability(reference, current, int(direction))
                directed[rows, column] = -np.log(
                    np.maximum(probability, epsilon)
                ).astype(np.float32)

    active = COLLAPSE_DIRECTIONS != 0
    composite = directed[:, active].mean(axis=1).astype(np.float32)
    result = {
        f"collapse.{name}.{('upper' if direction > 0 else 'lower')}_tail": directed[
            :, index
        ]
        for index, (name, direction) in enumerate(
            zip(COLLAPSE_FEATURE_NAMES, COLLAPSE_DIRECTIONS, strict=True)
        )
        if int(direction) != 0
    }
    result.update(
        {
            f"collapse.{name}.two_sided": two_sided[:, index]
            for index, name in enumerate(COLLAPSE_FEATURE_NAMES)
        }
    )
    result["collapse.composite"] = composite
    return result, directed, two_sided


def collapse_reference_fields(
    values: np.ndarray,
    relative_conditions: np.ndarray,
    causal_conditions: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "calibration_collapse_values": np.asarray(values, dtype=np.float32),
        "calibration_collapse_relative_conditions": np.asarray(
            relative_conditions,
            dtype=str,
        ),
        "calibration_collapse_causal_conditions": np.asarray(
            causal_conditions,
            dtype=str,
        ),
        "collapse_feature_names": np.asarray(COLLAPSE_FEATURE_NAMES, dtype=str),
        "collapse_directions": np.asarray(COLLAPSE_DIRECTIONS, dtype=np.int8),
    }


def collapse_reference(reference: Mapping[str, np.ndarray]):
    names = tuple(
        map(
            str,
            np.asarray(reference["collapse_feature_names"], dtype=str).tolist(),
        )
    )
    if names != COLLAPSE_FEATURE_NAMES:
        raise ValueError("collapse feature contract changed")
    directions = np.asarray(reference["collapse_directions"], dtype=np.int8)
    if not np.array_equal(directions, COLLAPSE_DIRECTIONS):
        raise ValueError("collapse direction contract changed")
    return (
        np.asarray(reference["calibration_collapse_values"], dtype=np.float32),
        np.asarray(
            reference["calibration_collapse_relative_conditions"],
            dtype=str,
        ),
        np.asarray(
            reference["calibration_collapse_causal_conditions"],
            dtype=str,
        ),
    )
