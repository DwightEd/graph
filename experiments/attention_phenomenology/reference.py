"""Unlabeled task/position reference for routing-state atypicality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FAMILY_FEATURES, FAMILY_NAMES, FEATURE_NAMES, PhenomenologyConfig


@dataclass(frozen=True)
class PhenomenologyReference:
    task: np.ndarray
    bucket: np.ndarray
    center: np.ndarray  # [condition, layer, feature]
    scale: np.ndarray
    feature_names: np.ndarray
    family_names: np.ndarray
    config_json: str

    @property
    def layers(self) -> int:
        return int(self.center.shape[1])


class Reservoir:
    """Uniform bounded sample of token-level layer fields."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self.values: list[np.ndarray] = []
        self.seen = 0

    def add(self, value: np.ndarray) -> None:
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(np.asarray(value, dtype=np.float32).copy())
            return
        index = int(self.rng.integers(self.seen))
        if index < self.capacity:
            self.values[index] = np.asarray(value, dtype=np.float32).copy()

    def matrix(self) -> np.ndarray:
        return np.stack(self.values)


def causal_position_bucket(position: int, bins: int) -> int:
    """Causal log2 prefix bucket; it never uses final response length."""

    return min(int(np.floor(np.log2(int(position) + 1))), int(bins) - 1)


def token_buckets(response_count: int, bins: int) -> np.ndarray:
    return np.asarray(
        [causal_position_bucket(position, bins) for position in range(response_count)],
        dtype=np.int16,
    )


def robust_center_scale(
    values: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    standard_deviation = np.std(values, axis=0)
    scale = np.where(
        mad > epsilon,
        mad,
        np.where(standard_deviation > epsilon, standard_deviation, 1.0),
    )
    return center.astype(np.float32), scale.astype(np.float32)


def fit_reference_from_reservoirs(
    reservoirs: dict[tuple[str, int], Reservoir],
    *,
    config: PhenomenologyConfig,
    config_json: str,
) -> PhenomenologyReference:
    keys = sorted(reservoirs)
    task = []
    bucket = []
    center = []
    scale = []
    for current_task, current_bucket in keys:
        values = reservoirs[(current_task, current_bucket)].matrix()
        current_center, current_scale = robust_center_scale(values, config.epsilon)
        task.append(current_task)
        bucket.append(current_bucket)
        center.append(current_center)
        scale.append(current_scale)
    return PhenomenologyReference(
        task=np.asarray(task, dtype=str),
        bucket=np.asarray(bucket, dtype=np.int16),
        center=np.stack(center),
        scale=np.stack(scale),
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
        family_names=np.asarray(FAMILY_NAMES, dtype=str),
        config_json=config_json,
    )


def save_reference(path, reference: PhenomenologyReference) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray("attention-phenomenology-reference-v1"),
        task=reference.task,
        bucket=reference.bucket,
        center=reference.center,
        scale=reference.scale,
        feature_names=reference.feature_names,
        family_names=reference.family_names,
        config_json=np.asarray(reference.config_json),
    )


def load_reference(path) -> PhenomenologyReference:
    with np.load(path, allow_pickle=False) as arrays:
        return PhenomenologyReference(
            task=arrays["task"].astype(str),
            bucket=arrays["bucket"].astype(np.int16),
            center=arrays["center"].astype(np.float32),
            scale=arrays["scale"].astype(np.float32),
            feature_names=arrays["feature_names"].astype(str),
            family_names=arrays["family_names"].astype(str),
            config_json=str(arrays["config_json"].item()),
        )


def _condition_map(reference: PhenomenologyReference) -> dict[tuple[str, int], int]:
    return {
        (str(task), int(bucket)): index
        for index, (task, bucket) in enumerate(zip(reference.task, reference.bucket))
    }


def standardize_features(
    layer_features: np.ndarray,
    *,
    task: str,
    buckets: np.ndarray,
    reference: PhenomenologyReference,
) -> np.ndarray:
    mapping = _condition_map(reference)
    available_global = sorted(
        int(bucket) for current_task, bucket in mapping if current_task == "__all__"
    )
    result = np.empty_like(layer_features, dtype=np.float32)
    for token, bucket in enumerate(np.asarray(buckets, dtype=np.int16)):
        current_bucket = int(bucket)
        index = mapping.get((str(task), current_bucket))
        if index is None:
            fallback_bucket = min(
                available_global, key=lambda value: abs(value - current_bucket)
            )
            index = mapping[("__all__", fallback_bucket)]
        result[token] = (
            layer_features[token] - reference.center[index]
        ) / reference.scale[index]
    return result


def family_atypicality(standardized: np.ndarray) -> np.ndarray:
    """Direction-free RMS deviations for each pre-registered mechanism family."""

    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    scores = []
    for family in FAMILY_NAMES:
        names = (
            tuple(name for name in FEATURE_NAMES if not name.endswith("mass_mean"))
            if family == "all"
            else FAMILY_FEATURES[family]
        )
        selected = standardized[:, :, [feature_index[name] for name in names]]
        scores.append(np.sqrt(np.mean(np.square(selected), axis=(1, 2))))
    return np.column_stack(scores).astype(np.float32)
