"""Dataset boundary for label-free capture and post-capture labels.

The audit consumes already aligned token/source-unit coordinates.  RAGTruth is
one adapter for that contract, not part of the mechanism algorithm.  A plain
JSON manifest is the second adapter and lets another dataset enter without
changing target selection, route capture, or causal interventions.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from experiments.common.ragtruth_alignment import (
    canonical_task_type,
    load_source_info,
)
from research_dataset import open_research_dataset

from .units import build_source_units
from .worlds import SourceUnits

DATASET_SCHEMA = 1
LABEL_SCHEMA = 1


@dataclass(frozen=True)
class SampleRecord:
    """Metadata allowed to participate in label-free cohort selection."""

    sample_id: str
    source_id: str
    task_type: str
    generator_model: str


@dataclass(frozen=True)
class AuditSample:
    """Model-ready sample with explicit semantic source coordinates."""

    record: SampleRecord
    token_ids: Tensor
    response_start: int
    units: SourceUnits
    evidence_unit_id: tuple[int, ...]

    @property
    def full_response_tokens(self) -> int:
        return len(self.token_ids) - self.response_start

    def check(self) -> AuditSample:
        if (
            self.token_ids.ndim != 1
            or self.token_ids.dtype != torch.long
            or self.token_ids.device.type != "cpu"
        ):
            raise ValueError("audit token IDs must be one CPU int64 vector")
        if not 0 < self.response_start < len(self.token_ids):
            raise ValueError("response_start does not define a non-empty response")
        self.units.check(len(self.token_ids) - 1)
        if not self.evidence_unit_id:
            raise ValueError("audit sample has no represented evidence unit")
        for unit_id in self.evidence_unit_id:
            if not 0 <= unit_id < self.units.count:
                raise ValueError("evidence unit ID is outside the source-unit table")
            if self.units.kind[unit_id] in {"other_prompt", "response"}:
                raise ValueError("evidence unit ID names a non-evidence unit")
            if not bool((self.units.token_unit_id == unit_id).any()):
                raise ValueError("evidence unit has no represented source token")
        return self

    def truncate_response(self, limit: int | None) -> AuditSample:
        """Return a prefix while keeping source-unit coordinates aligned."""

        if limit is None or self.full_response_tokens <= limit:
            return self
        if limit < 1:
            raise ValueError("response token limit must be positive")
        stop = self.response_start + limit
        token_unit_id = self.units.token_unit_id[: stop - 1].clone()
        represented_units = int(token_unit_id.max()) + 1
        return AuditSample(
            self.record,
            self.token_ids[:stop].clone(),
            self.response_start,
            SourceUnits(
                token_unit_id,
                self.units.name[:represented_units],
                self.units.kind[:represented_units],
            ),
            self.evidence_unit_id,
        ).check()


@dataclass(frozen=True)
class CorpusIdentity:
    """External input identity persisted in the resumable run manifest."""

    kind: str
    location: str
    split: str
    tokenizer_id: str
    observer_model_id: str = ""
    source_info: str = ""
    name: str = ""

    def manifest_fields(self) -> dict[str, object]:
        if self.kind == "ragtruth":
            return {
                "dataset_root": self.location,
                "source_info": self.source_info,
                "split": self.split,
            }
        return {
            "dataset_format": self.kind,
            "dataset_manifest": self.location,
            "dataset_name": self.name,
            "split": self.split,
        }

    def validate_runtime(self, model_id: str, tokenizer_id: str) -> None:
        if Path(self.tokenizer_id).name != Path(tokenizer_id).name:
            raise ValueError("dataset and loaded tokenizer differ")
        if not _same_model(self.observer_model_id, model_id):
            raise ValueError("dataset observer and current model differ")


class AuditCorpus(Protocol):
    """Small input interface used by :class:`SubsetAuditRunner`."""

    identity: CorpusIdentity
    records: tuple[SampleRecord, ...]

    def load_sample(self, record: SampleRecord) -> AuditSample: ...


class LabelSource(Protocol):
    """Separate interface opened only after label-free artifacts are frozen."""

    def load(self, sample_ids: tuple[str, ...]) -> dict[str, np.ndarray]: ...


def _same_model(recorded: str, requested: str | Path) -> bool:
    if not recorded:
        return True
    recorded_path = Path(recorded)
    requested_path = Path(requested)
    if recorded_path.is_absolute():
        return recorded_path.resolve() == requested_path.resolve()
    return recorded_path.name == requested_path.name


def _record_task(
    sample_id: str,
    cached_task,
    source: Mapping | None,
) -> str:
    cache_value = None
    if cached_task is not None and str(cached_task).strip():
        cache_value = canonical_task_type(cached_task)
    source_value = None if source is None else canonical_task_type(source["task_type"])
    if (
        cache_value is not None
        and source_value is not None
        and cache_value != source_value
    ):
        raise ValueError(f"sample {sample_id} and source_info disagree on task type")
    task = source_value or cache_value
    if task is None:
        raise ValueError(
            f"sample {sample_id} has no task type in formal metadata or source_info"
        )
    return task


def inspect_records(
    dataset,
    *,
    sample_ids: Iterable[str] | None = None,
    source_info: Mapping[str, Mapping] | None = None,
) -> tuple[SampleRecord, ...]:
    """Read only allow-listed metadata; do not dereference labels."""

    records = []
    available = tuple(map(str, dataset.sample_ids))
    selected = available if sample_ids is None else tuple(map(str, sample_ids))
    available_set = set(available)
    missing = [sample_id for sample_id in selected if sample_id not in available_set]
    if missing:
        raise ValueError(f"sample IDs not found: {', '.join(missing)}")
    metadata_reader = getattr(dataset, "metadata", None)
    for sample_id in selected:
        if callable(metadata_reader):
            metadata = metadata_reader(sample_id)
            source_id = str(metadata["source_id"])
            cached_task = metadata.get("task_type")
            generator_model = str(metadata.get("generator_model") or "")
        else:
            sample = dataset[sample_id]
            try:
                source_id = str(sample.source_id)
                cached_task = sample.task_type
                generator_model = str(getattr(sample, "generator_model", "") or "")
            finally:
                sample.release_attention()
        source = None if source_info is None else source_info.get(source_id)
        records.append(
            SampleRecord(
                sample_id,
                source_id,
                _record_task(sample_id, cached_task, source),
                generator_model,
            )
        )
    return tuple(records)


def select_records(
    records: Iterable[SampleRecord],
    *,
    tasks: tuple[str, ...],
    samples_per_task: int,
    seed: int,
    sample_ids: tuple[str, ...] = (),
) -> tuple[SampleRecord, ...]:
    """Select a deterministic, source-diverse cohort without label access."""

    if samples_per_task < 0:
        raise ValueError("samples_per_task cannot be negative")
    records = tuple(records)
    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("dataset contains duplicate sample IDs")
    if sample_ids:
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample IDs not found: {', '.join(missing)}")
        selected = tuple(by_id[sample_id] for sample_id in sample_ids)
        invalid = [item.sample_id for item in selected if item.task_type not in tasks]
        if invalid:
            raise ValueError(
                "explicit samples fall outside --task: " + ", ".join(invalid)
            )
        return selected

    selected: list[SampleRecord] = []
    randomizer = random.Random(seed)
    for task in tasks:
        candidates = sorted(
            (record for record in records if record.task_type == task),
            key=lambda item: (item.source_id, item.sample_id),
        )
        if samples_per_task == 0:
            selected.extend(sorted(candidates, key=lambda item: item.sample_id))
            continue
        if len(candidates) < samples_per_task:
            raise ValueError(
                f"task {task} has {len(candidates)} available samples; "
                f"{samples_per_task} requested"
            )
        by_source: dict[str, list[SampleRecord]] = {}
        for record in candidates:
            by_source.setdefault(record.source_id, []).append(record)
        source_ids = sorted(by_source)
        randomizer.shuffle(source_ids)
        primary = [randomizer.choice(by_source[source_id]) for source_id in source_ids]
        chosen = primary[:samples_per_task]
        if len(chosen) < samples_per_task:
            chosen_ids = {record.sample_id for record in chosen}
            remaining = [
                record for record in candidates if record.sample_id not in chosen_ids
            ]
            randomizer.shuffle(remaining)
            chosen.extend(remaining[: samples_per_task - len(chosen)])
        selected.extend(chosen)
    if not selected:
        raise ValueError("no samples match the requested task subset")
    return tuple(selected)


def sample_tokens(dataset, sample_id: str) -> tuple[Tensor, int]:
    """Return one sample's token IDs and response boundary without labels."""
    sample = dataset[sample_id]
    try:
        cached = sample.attention()
        return cached.token_ids.detach().cpu().long().clone(), int(cached.response_idx)
    finally:
        sample.release_attention()


@dataclass
class RagTruthAuditCorpus:
    """Adapter from the existing formal attention cache to ``AuditCorpus``."""

    identity: CorpusIdentity
    records: tuple[SampleRecord, ...]
    _dataset: object
    _sources: Mapping[str, Mapping]
    _tokenizer: object

    @classmethod
    def open(
        cls,
        dataset_root: str | Path,
        source_info: str | Path,
        *,
        split: str,
        model_id: str,
        tokenizer,
        sample_ids: tuple[str, ...] = (),
    ) -> RagTruthAuditCorpus:
        root = Path(dataset_root).resolve()
        source_path = Path(source_info).resolve()
        dataset = open_research_dataset(
            root,
            device="cpu",
            retain_embedded_labels=False,
        )
        dataset_split = str(dataset.manifest.get("split", ""))
        if dataset_split.casefold() != split.casefold():
            raise ValueError(
                f"dataset split {dataset_split!r} differs from requested split {split!r}"
            )
        recorded_model = str(getattr(dataset, "spec", {}).get("model_path", ""))
        if not _same_model(recorded_model, model_id):
            raise ValueError("cached observer and current model differ")
        tokenizer_id = Path(tokenizer.name_or_path).name
        sources = load_source_info(source_path)
        records = inspect_records(
            dataset,
            sample_ids=sample_ids or None,
            source_info=sources,
        )
        for record in records:
            if record.source_id not in sources:
                raise ValueError(
                    f"source_info lacks {record.source_id} for {record.sample_id}"
                )
        dataset.verify_hashes = True
        identity = CorpusIdentity(
            kind="ragtruth",
            location=str(root),
            split=split,
            tokenizer_id=tokenizer_id,
            observer_model_id=recorded_model,
            source_info=str(source_path),
            name="RAGTruth",
        )
        return cls(identity, records, dataset, sources, tokenizer)

    def load_sample(self, record: SampleRecord) -> AuditSample:
        token_ids, response_start = sample_tokens(self._dataset, record.sample_id)
        source = self._sources[record.source_id]
        if canonical_task_type(source["task_type"]) != record.task_type:
            raise ValueError(
                f"sample {record.sample_id} and source_info disagree on task type"
            )
        units = build_source_units(
            source,
            self._tokenizer,
            token_ids,
            response_start,
        )
        evidence_units = tuple(
            unit_id
            for unit_id, kind in enumerate(units.kind)
            if kind not in {"other_prompt", "response"}
            and bool((units.token_unit_id == unit_id).any())
        )
        return AuditSample(
            record,
            token_ids,
            response_start,
            units,
            evidence_units,
        ).check()


@dataclass
class ManifestAuditCorpus:
    """Dataset-neutral adapter for already aligned audit inputs."""

    identity: CorpusIdentity
    records: tuple[SampleRecord, ...]
    _samples: dict[str, AuditSample]

    @classmethod
    def open(cls, path: str | Path) -> ManifestAuditCorpus:
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("audit_dataset_schema") != DATASET_SCHEMA:
            raise ValueError("unsupported audit dataset schema")
        split = str(payload.get("split") or "")
        tokenizer_id = str(payload.get("tokenizer_id") or "")
        if not split or not tokenizer_id:
            raise ValueError("dataset manifest requires split and tokenizer_id")
        samples: dict[str, AuditSample] = {}
        records = []
        for row in payload.get("samples", []):
            record = SampleRecord(
                str(row["sample_id"]),
                str(row["source_id"]),
                str(row["task_type"]),
                str(row.get("generator_model") or ""),
            )
            if not record.sample_id or record.sample_id in samples:
                raise ValueError("dataset manifest contains an empty or duplicate sample ID")
            unit_row = row["source_units"]
            units = SourceUnits(
                torch.as_tensor(unit_row["token_unit_id"], dtype=torch.long),
                tuple(map(str, unit_row["name"])),
                tuple(map(str, unit_row["kind"])),
            )
            sample = AuditSample(
                record,
                torch.as_tensor(row["token_ids"], dtype=torch.long),
                int(row["response_start"]),
                units,
                tuple(map(int, row["evidence_unit_id"])),
            ).check()
            records.append(record)
            samples[record.sample_id] = sample
        if not records:
            raise ValueError("dataset manifest contains no samples")
        identity = CorpusIdentity(
            kind="manifest",
            location=str(source),
            split=split,
            tokenizer_id=tokenizer_id,
            observer_model_id=str(payload.get("model_id") or ""),
            name=str(payload.get("name") or source.stem),
        )
        return cls(identity, tuple(records), samples)

    def load_sample(self, record: SampleRecord) -> AuditSample:
        sample = self._samples.get(record.sample_id)
        if sample is None or sample.record != record:
            raise ValueError("sample record does not belong to this dataset manifest")
        return AuditSample(
            sample.record,
            sample.token_ids.clone(),
            sample.response_start,
            SourceUnits(
                sample.units.token_unit_id.clone(),
                sample.units.name,
                sample.units.kind,
            ),
            sample.evidence_unit_id,
        )


@dataclass(frozen=True)
class RagTruthLabelSource:
    dataset_root: Path

    def load(self, sample_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
        dataset = open_research_dataset(
            self.dataset_root,
            device="cpu",
            retain_embedded_labels=True,
        )
        labels = dataset.prepare_evaluation_labels(list(sample_ids))
        result = {}
        for sample_id in sample_ids:
            sample = dataset[sample_id]
            try:
                result[sample_id] = labels.response_labels(sample).detach().cpu().numpy()
            finally:
                sample.release_attention()
        return result


@dataclass(frozen=True)
class ManifestLabelSource:
    path: Path
    _labels: dict[str, np.ndarray]

    @classmethod
    def open(cls, path: str | Path) -> ManifestLabelSource:
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("audit_labels_schema") != LABEL_SCHEMA:
            raise ValueError("unsupported audit label schema")
        labels = {
            str(sample_id): np.asarray(values, dtype=np.int64)
            for sample_id, values in payload.get("samples", {}).items()
        }
        if any(values.ndim != 1 for values in labels.values()):
            raise ValueError("each label row must be a one-dimensional sequence")
        if any(not np.isin(values, (0, 1)).all() for values in labels.values()):
            raise ValueError("audit labels must contain only 0 and 1")
        return cls(source, labels)

    def load(self, sample_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
        missing = [sample_id for sample_id in sample_ids if sample_id not in self._labels]
        if missing:
            raise ValueError(f"label manifest lacks samples: {', '.join(missing)}")
        return {sample_id: self._labels[sample_id].copy() for sample_id in sample_ids}
