"""Shared leakage and evaluation protocol for frozen experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SourceGroupAudit:
    """The canonical held-out groups and sample scope observed during scoring."""

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
        return cls(path=path, sha256=file_sha256(path))

    def verify(self, path) -> None:
        path = Path(path).resolve()
        if path != self.path:
            raise ValueError("frozen file identity differs from the captured artifact")
        if file_sha256(path) != self.sha256:
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
        required = {
            "dataset_manifest_sha256",
            "sample_id",
            "source_id",
            "token_index",
            "response_length",
        }
        missing = required.difference(rows)
        if missing:
            raise ValueError(
                f"evaluation artifact misses dataset binding fields: {sorted(missing)}"
            )
        recorded_manifest = np.asarray(rows["dataset_manifest_sha256"])
        if recorded_manifest.ndim != 0 or recorded_manifest.dtype.kind not in {
            "U",
            "S",
        }:
            raise ValueError("evaluation artifact has an invalid dataset manifest digest")
        if str(recorded_manifest.item()) != dataset_manifest_sha256(dataset):
            raise ValueError("evaluation dataset manifest differs from score artifact")

        sample_ids = rows["sample_id"]
        token_indices = rows["token_index"]
        facts = _complete_token_row_facts(
            sample_ids,
            rows["source_id"],
            token_indices,
            rows["response_length"],
        )
        for sample_id, (recorded_source, recorded_length) in facts.items():
            sample = dataset[sample_id]
            try:
                response_length = int(sample.attention().num_response_tokens)
                if canonical_source_group(sample) != recorded_source:
                    raise ValueError(
                        "evaluation canonical source differs from score artifact"
                    )
                if response_length != recorded_length:
                    raise ValueError(
                        "evaluation response length differs from score artifact"
                    )
            finally:
                sample.release_attention()
        labels = _align_evaluation_labels(dataset, sample_ids, token_indices)
        return rows, labels


class HeldOutSourceAudit:
    """Stream one held-out split against frozen reserved source groups."""

    def __init__(
        self,
        dataset,
        *,
        selected_sample_ids,
        reserved_source_ids,
        require_complete_split=True,
    ):
        reserved = _recorded_source_groups(reserved_source_ids)
        if not reserved:
            raise ValueError("reserved source groups must be non-empty")
        self._dataset = dataset
        self._reserved_source_ids = set(reserved)
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
        if source_id in self._reserved_source_ids:
            raise ValueError("reserved and test source groups must be disjoint")
        self._observed_source_ids[sample_id] = source_id

    def finish(self) -> SourceGroupAudit:
        """Require one observation for every selected sample and persist it."""

        missing = set(self.test_sample_ids).difference(self._observed_source_ids)
        if missing:
            raise ValueError("selected test samples were not observed")
        return SourceGroupAudit(
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


def validate_source_audit(
    *,
    reserved_source_ids,
    test_source_ids,
    test_sample_ids,
    row_sample_ids,
    row_source_ids,
    audit_scope,
) -> None:
    """Validate one frozen held-out source audit against its artifact rows."""

    reserved = set(_declared_identifiers(reserved_source_ids))
    test_groups = set(_declared_identifiers(test_source_ids))
    selected_samples = set(_declared_identifiers(test_sample_ids))
    if reserved & test_groups:
        raise ValueError("frozen source-group audit contains reserved test groups")

    row_samples = np.asarray(row_sample_ids)
    row_sources = np.asarray(row_source_ids)
    if (
        row_samples.ndim != 1
        or row_sources.ndim != 1
        or row_samples.dtype.kind not in {"U", "S"}
        or row_sources.dtype.kind not in {"U", "S"}
        or len(row_samples) == 0
        or len(row_samples) != len(row_sources)
    ):
        raise ValueError("frozen source-group audit has invalid row mappings")
    row_sample_values = tuple(map(str, row_samples.tolist()))
    if any(_valid_source_id(sample_id) is None for sample_id in row_sample_values):
        raise ValueError("frozen source-group audit has invalid row sample IDs")
    if set(row_sample_values) != selected_samples:
        raise ValueError("frozen source-group audit does not match selected samples")

    sample_groups: dict[str, str] = {}
    for sample_id, source_id in zip(
        row_sample_values, row_sources.tolist(), strict=True
    ):
        group = _valid_source_id(source_id) or sample_id
        previous = sample_groups.setdefault(sample_id, group)
        if previous != group:
            raise ValueError("frozen source-group audit maps one sample to many groups")
    if set(sample_groups.values()) != test_groups:
        raise ValueError("frozen source-group audit does not match test groups")
    if str(audit_scope) not in {"complete_split", "selected_samples"}:
        raise ValueError("frozen source-group audit has an invalid scope")


def validate_complete_token_rows(
    sample_id,
    source_id,
    token_index,
    response_length,
) -> None:
    """Require one complete, self-consistent row set per scored response."""

    _complete_token_row_facts(
        sample_id,
        source_id,
        token_index,
        response_length,
    )


def _complete_token_row_facts(
    sample_id,
    source_id,
    token_index,
    response_length,
) -> dict[str, tuple[str, int]]:
    samples = np.asarray(sample_id)
    sources = np.asarray(source_id)
    tokens = np.asarray(token_index)
    lengths = np.asarray(response_length)
    if (
        samples.ndim != 1
        or sources.ndim != 1
        or tokens.ndim != 1
        or lengths.ndim != 1
        or samples.dtype.kind not in {"U", "S"}
        or sources.dtype.kind not in {"U", "S"}
        or len(samples) == 0
        or len({len(samples), len(sources), len(tokens), len(lengths)}) != 1
    ):
        raise ValueError("complete token rows have invalid columns")
    if not np.issubdtype(tokens.dtype, np.integer) or not np.issubdtype(
        lengths.dtype, np.integer
    ):
        raise ValueError("complete token rows must use integer token geometry")
    if bool((tokens < 0).any()) or bool((lengths < 1).any()):
        raise ValueError("complete token rows have invalid token geometry")

    sample_values = samples.astype(str, copy=False)
    source_values = sources.astype(str, copy=False)
    facts: dict[str, tuple[str, int]] = {}
    for sample in dict.fromkeys(sample_values.tolist()):
        if _valid_source_id(sample) is None:
            raise ValueError("complete token rows have invalid sample IDs")
        selected = sample_values == sample
        canonical_sources = {
            _valid_source_id(value) or sample for value in source_values[selected]
        }
        response_lengths = set(map(int, lengths[selected].tolist()))
        if len(canonical_sources) != 1 or len(response_lengths) != 1:
            raise ValueError("complete token rows are inconsistent within a response")
        length = next(iter(response_lengths))
        if not np.array_equal(
            np.sort(tokens[selected].astype(np.int64, copy=False)),
            np.arange(length, dtype=np.int64),
        ):
            raise ValueError("complete token rows do not cover the full response")
        facts[sample] = (next(iter(canonical_sources)), length)
    return facts


def partition_source_groups(
    dataset,
    sample_ids,
    *,
    calibration_fraction: float,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    """Deterministically split complete source groups by an exact group count."""

    calibration_fraction = float(calibration_fraction)
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0,1)")
    selected = _selected_sample_ids(
        dataset,
        sample_ids,
        require_complete_split=False,
    )
    groups: dict[str, list[str]] = {}
    group_by_sample: dict[str, str] = {}
    for sample_id in selected:
        sample = dataset[sample_id]
        try:
            group_id = canonical_source_group(sample)
            groups.setdefault(group_id, []).append(sample_id)
            group_by_sample[sample_id] = group_id
        finally:
            sample.release_attention()
    if len(groups) < 2:
        raise ValueError("source partition needs at least two source groups")

    def group_order(group_id: str) -> bytes:
        return hashlib.sha256(
            f"source-group-split-v1\0{int(seed)}\0{group_id}".encode("utf-8")
        ).digest()

    ordered = sorted(groups, key=group_order)
    calibration_count = min(
        len(ordered) - 1,
        max(1, int(round(len(ordered) * calibration_fraction))),
    )
    calibration_groups = set(ordered[:calibration_count])
    fit_groups = set(ordered).difference(calibration_groups)
    return {
        "fit_sample_ids": tuple(
            sample_id
            for sample_id in selected
            if group_by_sample[sample_id] in fit_groups
        ),
        "calibration_sample_ids": tuple(
            sample_id
            for sample_id in selected
            if group_by_sample[sample_id] in calibration_groups
        ),
        "fit_group_ids": tuple(sorted(fit_groups)),
        "calibration_group_ids": tuple(sorted(calibration_groups)),
    }


def _declared_identifiers(values) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError("frozen source-group audit fields must be vectors")
    identifiers = tuple(map(str, array.tolist()))
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(_valid_source_id(value) is None for value in identifiers)
    ):
        raise ValueError("frozen source-group audit has invalid identifiers")
    return identifiers


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


def file_sha256(path) -> str:
    """Return the SHA-256 digest of one frozen experiment file."""

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_manifest_sha256(dataset) -> str:
    """Return the identity digest of a dataset's on-disk manifest."""

    return file_sha256(Path(dataset.root) / "manifest.json")


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
