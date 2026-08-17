"""Validated in-memory contracts shared by the benchmark modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment_protocol import EvaluationLabels


def _one_dimensional(name: str, value, length: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} has {len(array)} rows; expected {length}")
    return array


@dataclass(frozen=True)
class MethodScore:
    """One detector output with a frozen anomaly direction."""

    name: str
    values: np.ndarray
    direction: str = "higher"
    protocol: str = "unknown"
    source_field: str = ""
    source_direction: str = ""

    def oriented(self, length: int) -> np.ndarray:
        values = _one_dimensional(self.name, self.values, length).astype(
            np.float64, copy=False
        )
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"method {self.name} contains non-finite scores")
        if self.direction not in {"higher", "lower"}:
            raise ValueError(
                f"method {self.name} direction must be 'higher' or 'lower'"
            )
        return values if self.direction == "higher" else -values


@dataclass
class ScoreArtifact:
    """Rows and detector scores loaded from one frozen artifact."""

    name: str
    path: str
    schema: str
    sample_id: np.ndarray
    source_id: np.ndarray
    token_index: np.ndarray
    response_length: np.ndarray
    dataset_manifest_sha256: str
    methods: dict[str, MethodScore]

    def validate(self) -> ScoreArtifact:
        self.sample_id = _one_dimensional("sample_id", self.sample_id).astype(str)
        length = len(self.sample_id)
        self.token_index = _one_dimensional(
            "token_index", self.token_index, length
        ).astype(np.int64)
        self.source_id = _one_dimensional("source_id", self.source_id, length).astype(
            str
        )
        self.response_length = _one_dimensional(
            "response_length", self.response_length, length
        ).astype(np.int64)
        if length == 0:
            raise ValueError(f"artifact {self.name} has no rows")
        if bool((self.token_index < 0).any()):
            raise ValueError(f"artifact {self.name} has negative token indices")
        keys = list(zip(self.sample_id.tolist(), self.token_index.tolist()))
        if len(set(keys)) != length:
            raise ValueError(f"artifact {self.name} has duplicate token rows")
        if not self.methods:
            raise ValueError(f"artifact {self.name} exposes no detector scores")
        self.methods = {
            name: MethodScore(
                name=method.name,
                values=method.oriented(length),
                direction="higher",
                protocol=method.protocol,
                source_field=method.source_field,
                source_direction=method.source_direction or method.direction,
            )
            for name, method in self.methods.items()
        }
        return self

    def evaluation_rows(self) -> dict[str, np.ndarray]:
        """Return the complete dataset-binding rows used to unlock labels."""

        return {
            "dataset_manifest_sha256": np.asarray(self.dataset_manifest_sha256),
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "token_index": self.token_index,
            "response_length": self.response_length,
        }


@dataclass(frozen=True)
class EvaluatedArtifact:
    """A strict score artifact bound to canonical evaluation facts."""

    score: ScoreArtifact
    labels: EvaluationLabels


@dataclass
class BenchmarkFrame:
    """Aligned detector rows plus evaluation-only labels and metadata."""

    sample_id: np.ndarray
    token_index: np.ndarray
    methods: dict[str, MethodScore]
    source_id: np.ndarray
    task_type: np.ndarray
    data_source: np.ndarray
    generator_model: np.ndarray
    response_length: np.ndarray
    relative_position: np.ndarray
    labels: np.ndarray
    response_positive: np.ndarray

    def validate(self) -> BenchmarkFrame:
        self.sample_id = _one_dimensional("sample_id", self.sample_id).astype(str)
        length = len(self.sample_id)
        for name in (
            "token_index",
            "source_id",
            "task_type",
            "data_source",
            "generator_model",
            "response_length",
            "relative_position",
            "labels",
            "response_positive",
        ):
            value = _one_dimensional(name, getattr(self, name), length)
            setattr(self, name, value)
        if not bool(np.isin(self.labels, (0, 1)).all()):
            raise ValueError("labels must be binary")
        if not bool(np.isin(self.response_positive, (0, 1)).all()):
            raise ValueError("response_positive must be binary")
        if not bool(np.isfinite(self.relative_position).all()):
            raise ValueError("relative positions contain non-finite values")
        for method in self.methods.values():
            method.oriented(length)
        for sample_id in np.unique(self.sample_id):
            selected = self.sample_id == sample_id
            if (
                len(np.unique(self.source_id[selected])) != 1
                or len(np.unique(self.response_length[selected])) != 1
                or len(np.unique(self.response_positive[selected])) != 1
            ):
                raise ValueError("canonical response facts vary within a sample")
        return self

    def subset(self, selected) -> BenchmarkFrame:
        selected = np.asarray(selected)
        return BenchmarkFrame(
            sample_id=self.sample_id[selected],
            token_index=self.token_index[selected],
            methods={
                name: MethodScore(
                    name=name,
                    values=method.values[selected],
                    direction="higher",
                    protocol=method.protocol,
                    source_field=method.source_field,
                    source_direction=method.source_direction or method.direction,
                )
                for name, method in self.methods.items()
            },
            source_id=self.source_id[selected],
            task_type=self.task_type[selected],
            data_source=self.data_source[selected],
            generator_model=self.generator_model[selected],
            response_length=self.response_length[selected],
            relative_position=self.relative_position[selected],
            labels=self.labels[selected],
            response_positive=self.response_positive[selected],
        ).validate()
