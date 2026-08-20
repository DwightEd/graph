"""Unlabeled task/position references for attention mechanism fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .artifacts import REFERENCE_SCHEMA
from .config import PhenomenologyConfig
from .hypotheses import FAMILY_FEATURES, FAMILY_NAMES, FEATURE_INDEX, FEATURE_NAMES


@dataclass(frozen=True)
class PhenomenologyReference:
    task: np.ndarray
    bucket: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    feature_names: np.ndarray
    family_names: np.ndarray
    config_json: str


class Reservoir:
    """Uniform bounded sample of token-level layer fields."""

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
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
    """Causal log2 prefix bucket; final response length is never used."""

    return min(int(np.floor(np.log2(position + 1))), bins - 1)


def token_buckets(response_count: int, bins: int) -> np.ndarray:
    return np.asarray(
        [causal_position_bucket(position, bins) for position in range(response_count)],
        dtype=np.int16,
    )


def robust_center_scale(
    values: np.ndarray, minimum_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    standard_deviation = np.std(values, axis=0)
    scale = np.where(mad > minimum_scale, mad, standard_deviation)
    scale = np.maximum(scale, minimum_scale)
    return center.astype(np.float32), scale.astype(np.float32)


def fit_reference_from_reservoirs(
    reservoirs: dict[tuple[str, int], Reservoir],
    *,
    config: PhenomenologyConfig,
    config_json: str,
) -> PhenomenologyReference:
    tasks = []
    buckets = []
    centers = []
    scales = []
    for task, bucket in sorted(reservoirs):
        center, scale = robust_center_scale(
            reservoirs[(task, bucket)].matrix(), config.reference_minimum_scale
        )
        tasks.append(task)
        buckets.append(bucket)
        centers.append(center)
        scales.append(scale)
    return PhenomenologyReference(
        task=np.asarray(tasks, dtype=str),
        bucket=np.asarray(buckets, dtype=np.int16),
        center=np.stack(centers),
        scale=np.stack(scales),
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
        family_names=np.asarray(FAMILY_NAMES, dtype=str),
        config_json=config_json,
    )


def save_reference(path, reference: PhenomenologyReference) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray(REFERENCE_SCHEMA),
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
    maximum_value: float = 10.0,
) -> np.ndarray:
    mapping = _condition_map(reference)
    global_buckets = sorted(
        bucket for current_task, bucket in mapping if current_task == "__all__"
    )
    standardized = np.empty_like(layer_features, dtype=np.float32)
    for token, bucket in enumerate(np.asarray(buckets, dtype=np.int16)):
        condition = mapping.get((task, int(bucket)))
        if condition is None:
            nearest = min(global_buckets, key=lambda value: abs(value - int(bucket)))
            condition = mapping[("__all__", nearest)]
        standardized[token] = (
            layer_features[token] - reference.center[condition]
        ) / reference.scale[condition]
    return np.clip(standardized, -maximum_value, maximum_value)


def family_layer_atypicality(standardized: np.ndarray) -> np.ndarray:
    """Direction-free RMS deviation for every token, layer, and family."""

    scores = []
    for family in FAMILY_NAMES:
        indices = [FEATURE_INDEX[name] for name in FAMILY_FEATURES[family]]
        selected = standardized[:, :, indices]
        scores.append(np.sqrt(np.mean(np.square(selected), axis=2)))
    return np.stack(scores, axis=2).astype(np.float32)


def family_atypicality(family_layer_scores: np.ndarray) -> np.ndarray:
    """Aggregate layer-resolved family deviations without learning weights."""

    return np.sqrt(np.mean(np.square(family_layer_scores), axis=1)).astype(np.float32)
