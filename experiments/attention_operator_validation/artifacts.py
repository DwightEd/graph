"""Small, auditable artifacts for answer-level operator-code validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import numpy as np


FEATURE_SCHEMA = "attention-operator-answer-features"
FEATURE_VERSION = 1


@dataclass(frozen=True)
class FeatureTable:
    sample_id: np.ndarray
    source_id: np.ndarray
    task_type: np.ndarray
    response_length: np.ndarray
    feature_names: tuple[str, ...]
    feature: np.ndarray
    metadata: dict[str, object]

    def validate(self) -> "FeatureTable":
        rows = len(self.sample_id)
        if any(
            len(value) != rows
            for value in (self.source_id, self.task_type, self.response_length, self.feature)
        ):
            raise ValueError("feature-table rows are misaligned")
        if self.feature.ndim != 2 or self.feature.shape[1] != len(self.feature_names):
            raise ValueError("feature matrix and feature names are misaligned")
        if len(set(map(str, self.sample_id.tolist()))) != rows:
            raise ValueError("feature table contains duplicate sample IDs")
        return self


def save_feature_table(path, table: FeatureTable) -> None:
    table = table.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": np.asarray(FEATURE_SCHEMA),
        "version": np.asarray(FEATURE_VERSION, dtype=np.int32),
        "sample_id": table.sample_id.astype(str),
        "source_id": table.source_id.astype(str),
        "task_type": table.task_type.astype(str),
        "response_length": table.response_length.astype(np.int32),
        "feature_names": np.asarray(table.feature_names, dtype=str),
        "feature": table.feature.astype(np.float32),
        "metadata_json": np.asarray(
            json.dumps(table.metadata, sort_keys=True, separators=(",", ":"))
        ),
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
        temporary = Path(file.name)
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_feature_table(path) -> FeatureTable:
    with np.load(path, allow_pickle=False) as arrays:
        if (
            str(arrays["schema"].item()) != FEATURE_SCHEMA
            or int(arrays["version"].item()) != FEATURE_VERSION
        ):
            raise ValueError("unsupported answer-feature artifact")
        table = FeatureTable(
            sample_id=arrays["sample_id"].astype(str),
            source_id=arrays["source_id"].astype(str),
            task_type=arrays["task_type"].astype(str),
            response_length=arrays["response_length"].astype(np.int32),
            feature_names=tuple(arrays["feature_names"].astype(str).tolist()),
            feature=arrays["feature"].astype(np.float32),
            metadata=json.loads(str(arrays["metadata_json"].item())),
        )
    return table.validate()
