"""Shared leakage and evaluation protocol for frozen experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SourceGroupAudit:
    """The complete source groups used for one fit/calibration/test run."""

    fit_source_ids: tuple[str, ...]
    calibration_source_ids: tuple[str, ...]
    test_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationLabels:
    """Canonical response labels aligned to frozen artifact rows."""

    token_label: np.ndarray
    response_positive: np.ndarray
    source_id: np.ndarray
    response_length: np.ndarray


@dataclass(frozen=True)
class FrozenFile:
    """An artifact identity and content digest captured before evaluation."""

    path: Path
    sha256: str

    @classmethod
    def capture(cls, path) -> "FrozenFile":
        path = Path(path).resolve()
        return cls(path=path, sha256=_file_sha256(path))

    def verify(self, path) -> None:
        path = Path(path).resolve()
        if path != self.path:
            raise ValueError("frozen file identity differs from the captured artifact")
        if _file_sha256(path) != self.sha256:
            raise ValueError("frozen file digest differs from the captured artifact")


def audit_source_groups(
    *,
    fit_source_ids,
    calibration_source_ids,
    test_source_ids,
) -> SourceGroupAudit:
    """Verify that complete source groups do not cross protocol roles."""

    fit = tuple(sorted({str(source_id) for source_id in fit_source_ids}))
    calibration = tuple(
        sorted({str(source_id) for source_id in calibration_source_ids})
    )
    test = tuple(sorted({str(source_id) for source_id in test_source_ids}))
    groups = (set(fit), set(calibration), set(test))
    if not all(groups):
        raise ValueError("fit, calibration, and test source groups must be non-empty")
    if any(
        left & right
        for index, left in enumerate(groups)
        for right in groups[index + 1 :]
    ):
        raise ValueError("fit, calibration, and test source groups must be disjoint")
    return SourceGroupAudit(fit, calibration, test)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _labels_for_evaluation(dataset):
    try:
        return dataset.labels()
    except RuntimeError as error:
        if "every attention sample" not in str(error):
            raise
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        sample.attention()
        sample.release_attention()
    return dataset.labels()


def _as_numpy_labels(values) -> np.ndarray:
    if hasattr(values, "cpu"):
        values = values.cpu().numpy()
    return np.asarray(values, dtype=np.int8)


def align_evaluation_labels(
    dataset,
    *,
    sample_ids,
    token_indices,
) -> EvaluationLabels:
    """Unlock labels at evaluation and align complete canonical response facts."""

    sample_ids = np.asarray(sample_ids, dtype=str)
    token_indices = np.asarray(token_indices, dtype=np.int64)
    if sample_ids.ndim != 1 or token_indices.ndim != 1:
        raise ValueError("sample_ids and token_indices must be one-dimensional")
    if len(sample_ids) != len(token_indices):
        raise ValueError("sample_ids and token_indices must have the same length")
    if bool((token_indices < 0).any()):
        raise ValueError("token_indices must be non-negative")

    labels = _labels_for_evaluation(dataset)
    canonical = {}
    for sample_id in dict.fromkeys(sample_ids.tolist()):
        sample = dataset[sample_id]
        try:
            token_label = _as_numpy_labels(labels.response_labels(sample))
            canonical[sample_id] = (
                token_label,
                str(sample.source_id),
                int(len(token_label)),
                int(token_label.any()),
            )
        finally:
            sample.release_attention()

    token_label = np.empty(len(sample_ids), dtype=np.int8)
    response_positive = np.empty(len(sample_ids), dtype=np.int8)
    response_length = np.empty(len(sample_ids), dtype=np.int32)
    source_ids = []
    for row, (sample_id, token_index) in enumerate(
        zip(sample_ids, token_indices, strict=True)
    ):
        labels_for_sample, source, length, positive = canonical[str(sample_id)]
        if token_index >= length:
            raise ValueError("token_index is outside canonical response length")
        token_label[row] = labels_for_sample[token_index]
        response_positive[row] = positive
        source_ids.append(source)
        response_length[row] = length
    return EvaluationLabels(
        token_label=token_label,
        response_positive=response_positive,
        source_id=np.asarray(source_ids, dtype=str),
        response_length=response_length,
    )
