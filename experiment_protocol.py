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
    test_sample_ids: tuple[str, ...]
    test_scope: str


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


@dataclass(frozen=True)
class FrozenEvaluation:
    """Evaluation permission bound to one frozen artifact and dataset split."""

    artifact: FrozenFile
    expected_split: str = "test"

    @classmethod
    def capture(cls, artifact_path, *, expected_split="test") -> "FrozenEvaluation":
        return cls(FrozenFile.capture(artifact_path), str(expected_split))

    def load_and_align(self, dataset, loader) -> tuple[object, EvaluationLabels]:
        """Load the captured artifact, reverify it, then align test labels."""

        rows = loader(self.artifact.path)
        self.artifact.verify(self.artifact.path)
        actual_split = str(dataset.manifest.get("split"))
        if actual_split != self.expected_split:
            raise ValueError(
                f"evaluation dataset split {actual_split!r} does not match "
                f"expected split {self.expected_split!r}"
            )
        try:
            sample_ids = rows["sample_id"]
            token_indices = rows["token_index"]
        except KeyError as error:
            raise ValueError("artifact rows require sample_id and token_index") from error
        labels = _align_evaluation_labels(dataset, sample_ids, token_indices)
        return rows, labels


class HeldOutSourceAudit:
    """Stream one held-out split through a frozen fit/calibration audit."""

    def __init__(
        self,
        dataset,
        *,
        selected_sample_ids,
        fit_source_ids,
        calibration_source_ids,
        require_complete_split=True,
    ):
        self.fit_source_ids = _recorded_source_groups(fit_source_ids)
        self.calibration_source_ids = _recorded_source_groups(
            calibration_source_ids
        )
        if not self.fit_source_ids or not self.calibration_source_ids:
            raise ValueError("fit and calibration source groups must be non-empty")
        if set(self.fit_source_ids) & set(self.calibration_source_ids):
            raise ValueError("fit and calibration source groups must be disjoint")
        self._dataset = dataset
        self._reference_source_ids = set(self.fit_source_ids) | set(
            self.calibration_source_ids
        )
        self.test_sample_ids = _selected_sample_ids(
            dataset,
            selected_sample_ids,
            require_complete_split=require_complete_split,
        )
        self.test_scope = (
            "complete_split" if require_complete_split else "selected_samples"
        )
        self._observed_source_ids: dict[str, str] = {}

    def observe(self, sample) -> None:
        """Record a loaded sample before its held-out features are scored."""

        if sample.dataset is not self._dataset:
            raise ValueError("observed sample belongs to a different dataset")
        sample_id = str(sample.sample_id)
        if sample_id not in self.test_sample_ids:
            raise ValueError("observed sample is outside the selected test scope")
        if sample_id in self._observed_source_ids:
            raise ValueError("selected test sample was observed more than once")
        source_id = canonical_source_group(sample)
        if source_id in self._reference_source_ids:
            raise ValueError("fit, calibration, and test source groups must be disjoint")
        self._observed_source_ids[sample_id] = source_id

    def finish(self) -> SourceGroupAudit:
        """Require one observation for every selected sample and persist it."""

        missing = set(self.test_sample_ids).difference(self._observed_source_ids)
        if missing:
            raise ValueError("selected test samples were not observed")
        return SourceGroupAudit(
            self.fit_source_ids,
            self.calibration_source_ids,
            tuple(sorted(set(self._observed_source_ids.values()))),
            self.test_sample_ids,
            self.test_scope,
        )


def canonical_source_group(sample) -> str:
    """Return a valid source ID, or isolate an ungrouped sample by its ID."""

    source_id = getattr(sample, "source_id", None)
    text = _valid_source_id(source_id)
    if text is not None:
        return text
    return str(sample.sample_id)


def _recorded_source_groups(source_ids) -> tuple[str, ...]:
    groups = []
    for source_id in source_ids:
        text = _valid_source_id(source_id)
        if text is None:
            raise ValueError("frozen source groups must contain valid source IDs")
        groups.append(text)
    return tuple(sorted(set(groups)))


def _valid_source_id(source_id) -> str | None:
    if source_id is None:
        return None
    text = str(source_id).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _selected_sample_ids(
    dataset,
    sample_ids,
    *,
    require_complete_split: bool,
) -> tuple[str, ...]:
    selected = tuple(map(str, sample_ids))
    available = tuple(map(str, dataset.sample_ids))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected sample IDs must be non-empty and unique")
    if not set(selected).issubset(available):
        raise ValueError("selected sample IDs are outside the dataset split")
    if require_complete_split and set(selected) != set(available):
        raise ValueError("source audit requires the complete split by default")
    return selected


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


def _align_evaluation_labels(dataset, sample_ids, token_indices) -> EvaluationLabels:
    """Unlock labels at evaluation and align complete canonical response facts."""

    sample_ids = np.asarray(sample_ids, dtype=str)
    raw_token_indices = np.asarray(token_indices)
    if sample_ids.ndim != 1 or raw_token_indices.ndim != 1:
        raise ValueError("sample_ids and token_indices must be one-dimensional")
    if len(sample_ids) != len(raw_token_indices):
        raise ValueError("sample_ids and token_indices must have the same length")
    if not np.issubdtype(raw_token_indices.dtype, np.integer):
        raise ValueError("token_indices must use an integer dtype")
    token_indices = raw_token_indices.astype(np.int64, copy=False)
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
                canonical_source_group(sample),
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
