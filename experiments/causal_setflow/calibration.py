"""Label-free latent reference and empirical component calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


COMPONENT_NAMES = (
    "route_element",
    "memory_element",
    "head_reconstruction",
    "layer_reconstruction",
    "temporal_prediction",
    "latent_mahalanobis",
)


@dataclass(frozen=True)
class CalibrationConfig:
    min_condition_rows: int = 32
    latent_trim_fraction: float = 0.90
    epsilon: float = 1e-8

    def validate(self) -> None:
        if int(self.min_condition_rows) < 2:
            raise ValueError("min_condition_rows must be at least two")
        if not 0.5 <= float(self.latent_trim_fraction) <= 1.0:
            raise ValueError("latent_trim_fraction must be in [0.5,1]")
        if not 0.0 < float(self.epsilon) < 1.0:
            raise ValueError("epsilon must be in (0,1)")


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


def _robust_location_scale(values, epsilon):
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    standard = np.std(values, axis=0)
    scale = np.where(mad > epsilon, mad, np.where(standard > epsilon, standard, 1.0))
    return center.astype(np.float32), scale.astype(np.float32)


def fit_latent_reference(
    embeddings: np.ndarray,
    *,
    trim_fraction: float,
    epsilon: float = 1e-8,
) -> dict[str, np.ndarray]:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("latent reference needs at least three rows")
    center, scale = _robust_location_scale(values, epsilon)
    standardized = (values - center) / scale
    provisional = np.mean(np.square(standardized), axis=1)
    threshold = float(np.quantile(provisional, float(trim_fraction)))
    keep = provisional <= threshold
    if int(keep.sum()) < 3:
        keep = np.ones(len(values), dtype=bool)
    covariance = LedoitWolf(assume_centered=False).fit(standardized[keep])
    return {
        "latent_center": center,
        "latent_scale": scale,
        "latent_precision_center": covariance.location_.astype(np.float32),
        "latent_precision": covariance.precision_.astype(np.float32),
        "latent_trim_threshold": np.asarray(threshold, dtype=np.float32),
        "latent_retained_rows": np.asarray(int(keep.sum()), dtype=np.int32),
    }


def latent_mahalanobis(embeddings: np.ndarray, reference) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    standardized = (
        values - np.asarray(reference["latent_center"], dtype=np.float32)
    ) / np.asarray(reference["latent_scale"], dtype=np.float32)
    centered = standardized - np.asarray(
        reference["latent_precision_center"], dtype=np.float32
    )
    score = np.einsum(
        "ni,ij,nj->n",
        centered,
        np.asarray(reference["latent_precision"], dtype=np.float32),
        centered,
        optimize=True,
    ) / max(1, centered.shape[1])
    return np.maximum(score, 0.0).astype(np.float32)


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


def component_matrix(components: dict[str, np.ndarray]) -> np.ndarray:
    missing = set(COMPONENT_NAMES).difference(components)
    if missing:
        raise ValueError(f"score components miss fields: {sorted(missing)}")
    matrix = np.column_stack(
        [np.asarray(components[name], dtype=np.float32) for name in COMPONENT_NAMES]
    )
    if matrix.ndim != 2 or not bool(np.isfinite(matrix).all()):
        raise FloatingPointError("component score matrix is invalid")
    return matrix


def calibrate_component_matrix(
    calibration_values: np.ndarray,
    calibration_conditions: np.ndarray,
    values: np.ndarray,
    conditions: np.ndarray,
    *,
    min_condition_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    calibration_values = np.asarray(calibration_values, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if calibration_values.ndim != 2 or values.ndim != 2:
        raise ValueError("component calibration expects matrices")
    if calibration_values.shape[1] != values.shape[1]:
        raise ValueError("calibration/test component widths differ")
    calibration_tail = np.column_stack(
        [
            empirical_upper_tail(
                calibration_values[:, column],
                calibration_conditions,
                calibration_values[:, column],
                calibration_conditions,
                min_condition_rows=min_condition_rows,
            )
            for column in range(calibration_values.shape[1])
        ]
    ).astype(np.float32)
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
    calibration_fisher = 2.0 * calibration_tail.sum(axis=1)
    fisher = 2.0 * tail.sum(axis=1)
    final = empirical_upper_tail(
        calibration_fisher,
        calibration_conditions,
        fisher,
        conditions,
        min_condition_rows=min_condition_rows,
    )
    return tail, final
