"""Small, auditable artifacts for answer-level operator-code validation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping
import zipfile

import numpy as np


FEATURE_SCHEMA = "attention-operator-answer-features"
FEATURE_VERSION = 2
PACKAGE_IMPLEMENTATION_FILES = (
    "experiments/attention_operator_validation/artifacts.py",
    "experiments/attention_operator_validation/evaluate.py",
    "experiments/attention_operator_validation/features.py",
    "experiments/attention_operator_validation/operators.py",
    "experiments/attention_operator_validation/pair_codes.py",
    "experiments/attention_operator_validation/pipeline.py",
)
DEPENDENCY_IMPLEMENTATION_FILES = (
    "experiment_protocol.py",
    "research_dataset.py",
    "experiments/grounded_route/config.py",
    "experiments/grounded_route/graph.py",
)
IMPLEMENTATION_FILES = (
    *PACKAGE_IMPLEMENTATION_FILES,
    *DEPENDENCY_IMPLEMENTATION_FILES,
)
PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "version",
        "sample_id",
        "source_id",
        "task_type",
        "response_length",
        "feature_names",
        "feature",
        "metadata_json",
    }
)


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lower-case SHA-256 digest")
    return text


def implementation_sha256() -> str:
    """Fingerprint the feature semantics, not the output location or run time."""

    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for name in IMPLEMENTATION_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"implementation dependency is missing: {path}")
        data = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


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
        if rows < 1:
            raise ValueError("feature table contains no answers")
        for name, value in (
            ("sample_id", self.sample_id),
            ("source_id", self.source_id),
            ("task_type", self.task_type),
            ("response_length", self.response_length),
        ):
            if np.asarray(value).ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional vector")
        if any(
            len(value) != rows
            for value in (self.source_id, self.task_type, self.response_length, self.feature)
        ):
            raise ValueError("feature-table rows are misaligned")
        if self.feature.ndim != 2 or self.feature.shape[1] != len(self.feature_names):
            raise ValueError("feature matrix and feature names are misaligned")
        if not self.feature_names or any(
            not isinstance(name, str) or not name for name in self.feature_names
        ):
            raise ValueError("feature names must be non-empty strings")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be unique")
        if len(set(map(str, self.sample_id.tolist()))) != rows:
            raise ValueError("feature table contains duplicate sample IDs")
        if any(not str(value).strip() for value in self.sample_id.tolist()):
            raise ValueError("sample IDs must be non-empty")
        lengths = np.asarray(self.response_length)
        if lengths.dtype.kind not in "iu" or bool((lengths < 1).any()):
            raise ValueError("response lengths must be positive integers")
        if bool(np.isinf(np.asarray(self.feature, dtype=np.float64)).any()):
            raise ValueError("feature values may be unavailable NaN, but never infinite")

        if not isinstance(self.metadata, Mapping):
            raise ValueError("feature metadata must be an object")
        implementation = str(self.metadata.get("implementation_sha256", ""))
        _sha256(implementation, "implementation_sha256")
        if implementation != implementation_sha256():
            raise ValueError(
                "feature artifact implementation differs from the running code"
            )
        _sha256(
            self.metadata.get("dataset_manifest_sha256", ""),
            "dataset_manifest_sha256",
        )
        _sha256(self.metadata.get("operator_sha256", ""), "operator_sha256")
        if bool(self.metadata.get("labels_used", True)):
            raise ValueError("operator features must be frozen without using labels")
        if self.metadata.get("audit_scope") != "selected_samples":
            raise ValueError("operator features must declare selected_samples scope")

        directions = self.metadata.get("feature_directions")
        if not isinstance(directions, Mapping) or set(directions) != set(
            self.feature_names
        ):
            raise ValueError("frozen feature directions do not cover the schema")
        if any(
            direction not in {"high", "low", "exploratory"}
            for direction in directions.values()
        ):
            raise ValueError("frozen feature direction is invalid")

        groups = self.metadata.get("probe_groups")
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError("frozen probe groups are missing")
        names = set(self.feature_names)
        for group, selected in groups.items():
            if not isinstance(group, str) or not group or not isinstance(selected, list):
                raise ValueError("frozen probe groups have an invalid schema")
            if not selected or len(selected) != len(set(selected)):
                raise ValueError("each frozen probe group must be non-empty and unique")
            if not set(selected).issubset(names):
                raise ValueError("a frozen probe group contains an unknown feature")
        return self


def _payload(table: FeatureTable) -> dict[str, np.ndarray]:
    table.validate()
    return {
        "schema": np.asarray(FEATURE_SCHEMA),
        "version": np.asarray(FEATURE_VERSION, dtype=np.int32),
        "sample_id": table.sample_id.astype(str),
        "source_id": table.source_id.astype(str),
        "task_type": table.task_type.astype(str),
        "response_length": table.response_length.astype(np.int32),
        "feature_names": np.asarray(table.feature_names, dtype=str),
        "feature": table.feature.astype(np.float32),
        "metadata_json": np.asarray(
            json.dumps(
                table.metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    }


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def save_feature_table(path, table: FeatureTable) -> None:
    """Atomically write a byte-deterministic compressed feature artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(table)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(payload):
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    _npy_bytes(payload[name]),
                    compresslevel=9,
                )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_feature_table(path) -> FeatureTable:
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != PAYLOAD_FIELDS:
            raise ValueError("answer-feature artifact fields differ from the schema")
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
