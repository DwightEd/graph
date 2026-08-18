"""Causal empirical calibration for learned MG-CASF energy scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CORRUPTION_NAMES


ENERGY_NAMES = (
    "general_energy",
    "token_energy",
    "channel_energy",
    "channel_energy_max",
    *(f"type_{name}" for name in CORRUPTION_NAMES),
)
PRIMARY_ENERGY = "general_energy"


@dataclass(frozen=True)
class CalibrationConfig:
    min_condition_rows: int = 32
    epsilon: float = 1e-12

    def validate(self) -> None:
        if int(self.min_condition_rows) < 2:
            raise ValueError("min_condition_rows must be at least two")
        if not 0.0 < float(self.epsilon) < 1.0:
            raise ValueError("calibration epsilon must be in (0,1)")


def causal_condition(task_type, token_index) -> np.ndarray:
    task = np.asarray(task_type, dtype=str)
    token = np.asarray(token_index, dtype=np.int64)
    if task.ndim != 1 or token.ndim != 1 or len(task) != len(token):
        raise ValueError("condition columns are not aligned")
    bucket = np.floor(np.log2(token + 1)).astype(np.int64)
    return np.asarray(
        [
            f"{name}\x1f{int(value)}"
            for name, value in zip(task, bucket, strict=True)
        ],
        dtype=str,
    )


def energy_matrix(rows: dict[str, np.ndarray]) -> np.ndarray:
    missing = {
        "general_energy",
        "token_energy",
        "channel_energy",
        "channel_energy_max",
        "type_energy",
    }.difference(rows)
    if missing:
        raise ValueError(f"energy rows miss fields: {sorted(missing)}")
    type_energy = np.asarray(rows["type_energy"], dtype=np.float32)
    if type_energy.ndim != 2 or type_energy.shape[1] != len(CORRUPTION_NAMES):
        raise ValueError("type-energy geometry is invalid")
    matrix = np.column_stack(
        (
            np.asarray(rows["general_energy"], dtype=np.float32),
            np.asarray(rows["token_energy"], dtype=np.float32),
            np.asarray(rows["channel_energy"], dtype=np.float32),
            np.asarray(rows["channel_energy_max"], dtype=np.float32),
            type_energy,
        )
    ).astype(np.float32)
    if matrix.shape[1] != len(ENERGY_NAMES) or not bool(np.isfinite(matrix).all()):
        raise FloatingPointError("learned energy matrix is invalid")
    return matrix


def empirical_upper_tail(
    reference_values: np.ndarray,
    reference_conditions: np.ndarray,
    values: np.ndarray,
    conditions: np.ndarray,
    *,
    min_condition_rows: int,
    epsilon: float = 1e-12,
) -> np.ndarray:
    reference_values = np.asarray(reference_values, dtype=np.float64)
    reference_conditions = np.asarray(reference_conditions, dtype=str)
    values = np.asarray(values, dtype=np.float64)
    conditions = np.asarray(conditions, dtype=str)
    result = np.empty(len(values), dtype=np.float32)
    global_reference = np.sort(reference_values[np.isfinite(reference_values)])
    if len(global_reference) < 2:
        raise ValueError("calibration needs at least two finite rows")
    for condition in np.unique(conditions):
        rows = np.flatnonzero(conditions == condition)
        selected = reference_values[reference_conditions == condition]
        selected = np.sort(selected[np.isfinite(selected)])
        if len(selected) < int(min_condition_rows):
            selected = global_reference
        count = len(selected) - np.searchsorted(
            selected, values[rows], side="left"
        )
        probability = (count + 1.0) / (len(selected) + 1.0)
        result[rows] = -np.log(np.maximum(probability, epsilon)).astype(np.float32)
    return result


def calibrate_energy_matrix(
    calibration_values: np.ndarray,
    calibration_conditions: np.ndarray,
    values: np.ndarray,
    conditions: np.ndarray,
    *,
    min_condition_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate every frozen energy; primary is general energy only."""

    calibration_values = np.asarray(calibration_values, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if calibration_values.ndim != 2 or values.ndim != 2:
        raise ValueError("energy calibration expects matrices")
    if calibration_values.shape[1] != len(ENERGY_NAMES):
        raise ValueError("calibration energy width differs from schema")
    if values.shape[1] != calibration_values.shape[1]:
        raise ValueError("calibration/test energy widths differ")
    tail = np.column_stack(
        [
            empirical_upper_tail(
                calibration_values[:, column],
                calibration_conditions,
                values[:, column],
                conditions,
                min_condition_rows=min_condition_rows,
            )
            for column in range(values.shape[1])
        ]
    ).astype(np.float32)
    primary_index = ENERGY_NAMES.index(PRIMARY_ENERGY)
    return tail, tail[:, primary_index].astype(np.float32, copy=False)