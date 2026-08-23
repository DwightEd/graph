"""Train-only robust reference for task and causal-position adjustment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment_protocol import scalar_text, sha256_text
from experiments.attention_phenomenology.reference import Reservoir, robust_center_scale

from .features import FEATURE_NAMES


@dataclass(frozen=True)
class StructureReference:
    task: np.ndarray
    bucket: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    feature_names: np.ndarray
    settings_json: str
    train_dataset_manifest_sha256: str
    source_ids: np.ndarray
    sample_ids: np.ndarray
    audit_scope: str


class ReferenceAccumulator:
    """Bounded train-token reservoirs with one small ``finish`` interface."""

    def __init__(
        self,
        *,
        capacity: int,
        seed: int,
        minimum_scale: float,
        settings_json: str = "{}",
    ):
        self.capacity = capacity
        self.minimum_scale = minimum_scale
        self.rng = np.random.default_rng(seed)
        self.settings_json = settings_json
        self.reservoirs: dict[tuple[str, int], Reservoir] = {}

    def add(self, task: str, bucket: int, value: np.ndarray) -> None:
        for current_task in (str(task), "__all__"):
            key = (current_task, int(bucket))
            self.reservoirs.setdefault(key, Reservoir(self.capacity, self.rng)).add(
                value
            )

    def finish(
        self,
        *,
        train_dataset_manifest_sha256: str,
        source_ids,
        sample_ids,
        audit_scope: str,
    ) -> StructureReference:
        tasks, buckets, centers, scales = [], [], [], []
        for task, bucket in sorted(self.reservoirs):
            center, scale = robust_center_scale(
                self.reservoirs[(task, bucket)].matrix(), self.minimum_scale
            )
            tasks.append(task)
            buckets.append(bucket)
            centers.append(center)
            scales.append(scale)
        return StructureReference(
            task=np.asarray(tasks, dtype=str),
            bucket=np.asarray(buckets, dtype=np.int16),
            center=np.stack(centers),
            scale=np.stack(scales),
            feature_names=np.asarray(FEATURE_NAMES, dtype=str),
            settings_json=self.settings_json,
            train_dataset_manifest_sha256=str(train_dataset_manifest_sha256),
            source_ids=np.asarray(sorted(set(source_ids)), dtype=str),
            sample_ids=np.asarray(sample_ids, dtype=str),
            audit_scope=str(audit_scope),
        )


def fit_reference(
    rows,
    *,
    minimum_scale: float = 1e-3,
    capacity: int = 2048,
    seed: int = 20260823,
    settings_json: str = "{}",
    train_dataset_manifest_sha256: str,
    source_ids,
    sample_ids,
    audit_scope: str = "selected_samples",
) -> StructureReference:
    accumulator = ReferenceAccumulator(
        capacity=capacity,
        seed=seed,
        minimum_scale=minimum_scale,
        settings_json=settings_json,
    )
    for task, bucket, value in rows:
        accumulator.add(task, bucket, value)
    return accumulator.finish(
        train_dataset_manifest_sha256=train_dataset_manifest_sha256,
        source_ids=source_ids,
        sample_ids=sample_ids,
        audit_scope=audit_scope,
    )


def save_reference(path, reference: StructureReference) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray("non-neural-structure-reference-v2"),
        task=reference.task,
        bucket=reference.bucket,
        center=reference.center,
        scale=reference.scale,
        feature_names=reference.feature_names,
        settings_json=np.asarray(reference.settings_json),
        train_dataset_manifest_sha256=np.asarray(
            reference.train_dataset_manifest_sha256
        ),
        reference_source_id=reference.source_ids,
        reference_sample_id=reference.sample_ids,
        audit_scope=np.asarray(reference.audit_scope),
    )


def load_reference(path) -> StructureReference:
    with np.load(path, allow_pickle=False) as arrays:
        if scalar_text(arrays, "schema") != "non-neural-structure-reference-v2":
            raise ValueError("unsupported structure reference schema")
        return StructureReference(
            task=arrays["task"].astype(str),
            bucket=arrays["bucket"].astype(np.int16),
            center=arrays["center"].astype(np.float32),
            scale=arrays["scale"].astype(np.float32),
            feature_names=arrays["feature_names"].astype(str),
            settings_json=str(arrays["settings_json"].item()),
            train_dataset_manifest_sha256=sha256_text(
                arrays, "train_dataset_manifest_sha256"
            ),
            source_ids=arrays["reference_source_id"].astype(str),
            sample_ids=arrays["reference_sample_id"].astype(str),
            audit_scope=scalar_text(arrays, "audit_scope"),
        )


def standardize(
    features: np.ndarray,
    *,
    task: str,
    buckets: np.ndarray,
    reference: StructureReference,
    maximum: float = 10.0,
) -> np.ndarray:
    mapping = {
        (str(task_name), int(bucket)): index
        for index, (task_name, bucket) in enumerate(
            zip(reference.task, reference.bucket)
        )
    }
    global_buckets = [bucket for name, bucket in mapping if name == "__all__"]
    result = np.empty_like(features, dtype=np.float32)
    for token, bucket in enumerate(np.asarray(buckets, dtype=np.int16)):
        index = mapping.get((str(task), int(bucket)))
        if index is None:
            nearest = min(global_buckets, key=lambda value: abs(value - int(bucket)))
            index = mapping[("__all__", nearest)]
        result[token] = (features[token] - reference.center[index]) / reference.scale[
            index
        ]
    return np.clip(result, -maximum, maximum)
