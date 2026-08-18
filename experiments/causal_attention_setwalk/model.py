"""Train-only robust reference models for frozen SetWalk embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class ReferenceConfig:
    reference_per_sample: int = 8
    position_bins: int = 8
    min_task_bin_rows: int = 8
    trim_fraction: float = 0.90

    def validate(self) -> None:
        if min(
            int(self.reference_per_sample),
            int(self.position_bins),
            int(self.min_task_bin_rows),
        ) < 1:
            raise ValueError("reference integer settings must be positive")
        if not 0.5 <= float(self.trim_fraction) <= 1.0:
            raise ValueError("trim_fraction must be in [0.5, 1]")


def response_position_bin(token_index: int, response_count: int, bins: int) -> int:
    if response_count <= 1:
        return 0
    return min(int(bins) - 1, int(token_index) * int(bins) // response_count)


def reference_positions(response_count: int, count: int) -> np.ndarray:
    if response_count < 1:
        raise ValueError("response_count must be positive")
    keep = min(int(response_count), int(count))
    return np.unique(
        np.rint(np.linspace(0, response_count - 1, keep)).astype(np.int32)
    )


def _robust_location_scale(values, *, epsilon=1e-6):
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    standard = np.std(values, axis=0)
    scale = np.where(mad > epsilon, mad, np.where(standard > epsilon, standard, 1.0))
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > epsilon), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _conditioned_location_scale(values, position_bin, task, config):
    values = np.asarray(values, dtype=np.float32)
    position_bin = np.asarray(position_bin, dtype=np.int16)
    task = np.asarray(task, dtype=str)
    global_center, global_scale = _robust_location_scale(values)
    position_center = np.empty(
        (config.position_bins, values.shape[1]), dtype=np.float32
    )
    position_scale = np.empty_like(position_center)
    for current_bin in range(config.position_bins):
        selected = position_bin == current_bin
        if int(selected.sum()) >= 2:
            position_center[current_bin], position_scale[current_bin] = (
                _robust_location_scale(values[selected])
            )
        else:
            position_center[current_bin] = global_center
            position_scale[current_bin] = global_scale

    task_names = np.asarray(sorted(set(task.tolist())), dtype=str)
    task_center = np.empty(
        (len(task_names), config.position_bins, values.shape[1]), dtype=np.float32
    )
    task_scale = np.empty_like(task_center)
    task_count = np.zeros(
        (len(task_names), config.position_bins), dtype=np.int32
    )
    for task_index, task_name in enumerate(task_names):
        for current_bin in range(config.position_bins):
            selected = (task == task_name) & (position_bin == current_bin)
            count = int(selected.sum())
            task_count[task_index, current_bin] = count
            if count >= config.min_task_bin_rows:
                task_center[task_index, current_bin], task_scale[
                    task_index, current_bin
                ] = _robust_location_scale(values[selected])
            else:
                task_center[task_index, current_bin] = position_center[current_bin]
                task_scale[task_index, current_bin] = position_scale[current_bin]
    return {
        "global_center": global_center,
        "global_scale": global_scale,
        "position_center": position_center,
        "position_scale": position_scale,
        "task_names": task_names,
        "task_center": task_center,
        "task_scale": task_scale,
        "task_count": task_count,
    }


def standardize(values, position_bin, task, model):
    values = np.asarray(values, dtype=np.float32)
    position_bin = np.asarray(position_bin, dtype=np.int64)
    task = np.asarray(task, dtype=str)
    result = np.empty_like(values)
    lookup = {str(name): index for index, name in enumerate(model["task_names"])}
    for task_name in np.unique(task):
        selected = task == task_name
        task_index = lookup.get(str(task_name))
        if task_index is None:
            center = model["position_center"][position_bin[selected]]
            scale = model["position_scale"][position_bin[selected]]
        else:
            center = model["task_center"][task_index, position_bin[selected]]
            scale = model["task_scale"][task_index, position_bin[selected]]
        result[selected] = (values[selected] - center) / scale
    return result.astype(np.float32, copy=False)


def fit_reference_model(values, position_bin, task, config: ReferenceConfig):
    """Fit a contamination-trimmed shrinkage precision without labels."""

    config.validate()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("reference embeddings must contain at least three rows")
    if not bool(np.isfinite(values).all()):
        raise FloatingPointError("reference embeddings contain non-finite values")
    model = _conditioned_location_scale(values, position_bin, task, config)
    standardized = standardize(values, position_bin, task, model)
    provisional = np.mean(np.square(standardized), axis=1)
    threshold = np.quantile(provisional, config.trim_fraction)
    keep = provisional <= threshold
    if int(keep.sum()) < max(3, min(values.shape[1] + 1, len(values) // 2)):
        keep = np.ones(len(values), dtype=bool)
    covariance = LedoitWolf(assume_centered=False).fit(standardized[keep])
    model.update(
        {
            "precision": covariance.precision_.astype(np.float32),
            "precision_center": covariance.location_.astype(np.float32),
            "trim_threshold": np.asarray(threshold, dtype=np.float32),
            "retained_rows": np.asarray(int(keep.sum()), dtype=np.int32),
        }
    )
    return model


def anomaly_score(values, position_bin, task, model):
    standardized = standardize(values, position_bin, task, model)
    centered = standardized - np.asarray(model["precision_center"])
    score = np.einsum(
        "ni,ij,nj->n",
        centered,
        np.asarray(model["precision"]),
        centered,
        optimize=True,
    ) / max(1, centered.shape[1])
    return np.maximum(score, 0.0).astype(np.float32)


MODEL_FIELDS = (
    "global_center",
    "global_scale",
    "position_center",
    "position_scale",
    "task_names",
    "task_center",
    "task_scale",
    "task_count",
    "precision",
    "precision_center",
    "trim_threshold",
    "retained_rows",
)


def pack_model(artifact, view_name, model):
    for field in MODEL_FIELDS:
        artifact[f"model_{view_name}_{field}"] = np.asarray(model[field])


def unpack_model(artifact, view_name):
    return {
        field: np.asarray(artifact[f"model_{view_name}_{field}"])
        for field in MODEL_FIELDS
    }
