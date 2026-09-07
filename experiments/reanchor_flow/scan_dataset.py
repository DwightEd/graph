"""Streaming scan access and a separate, post-hoc correctness-label join.

The detector only receives :class:`ScanDataset`.  Labels require constructing
``ScanLabelStore`` explicitly after fitting and scoring.  Loading one scan
decompresses only the requested NPZ members; no raw scan collection is copied.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

BUCKET_NAMES = (
    "prompt_evidence",
    "other_prompt",
    "remote_response",
    "recent_local",
)
BUCKET_FIELDS = (
    "reanchor_bucket_transport",
    "reanchor_bucket_attention",
    "reanchor_bucket_source_position",
    "reanchor_bucket_source_unit_id",
)
SCAN_FIELDS = (*BUCKET_FIELDS, "reanchor_score", "token_ids")
DEFAULT_FIELDS = ("reanchor_bucket_transport", "reanchor_score")


@dataclass(frozen=True)
class ScanRecord:
    sample_id: str
    source_id: str
    task_type: str
    path: Path


@dataclass(frozen=True)
class ScanSample:
    """One scan, retaining the original ``[layer, head, row, bucket]`` axes.

    ``metadata`` contains capture coverage and identity, including eventual
    response length.  Such future-dependent metadata is for validation and
    coverage reports only. ``response_index`` uses exclusively the current
    predictor position and the known prompt boundary.
    """

    sample_id: str
    source_id: str
    task_type: str
    response_start: int
    row_position: np.ndarray
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    @property
    def response_index(self) -> np.ndarray:
        """Index of the token predicted by each row: ``q + 1 - prompt_len``."""

        return self.row_position + 1 - self.response_start

    @property
    def rows(self) -> int:
        return len(self.row_position)


class ScanDataset:
    """Read a completed train or test scan manifest without opening labels."""

    def __init__(
        self, split_root: str | Path, *, tasks: Iterable[str] | None = None
    ) -> None:
        self.root = Path(split_root)
        manifest = json.loads((self.root / "run_manifest.json").read_text())
        if manifest.get("subset_manifest_schema") != 3:
            raise ValueError("unsupported scan manifest schema")
        if not manifest.get("analysis_complete"):
            raise ValueError("scan capture is incomplete")
        if manifest.get("labels_used_for_capture") is not False:
            raise ValueError("scan manifest violates the label firewall")
        self.config = MappingProxyType(manifest["config"])
        self.split = str(self.config["split"])
        wanted = None if tasks is None else set(tasks)
        records = []
        for sample_id, entry in manifest["samples"].items():
            if wanted is not None and entry["task_type"] not in wanted:
                continue
            records.append(
                ScanRecord(
                    str(sample_id),
                    str(entry["source_id"]),
                    str(entry["task_type"]),
                    self.root / entry["scan"],
                )
            )
        self.records = tuple(records)
        self._by_id = {record.sample_id: record for record in self.records}
        self.sample_ids = tuple(self._by_id)
        for selected in manifest.get("selection", ()):
            if wanted is not None and selected["task_type"] not in wanted:
                continue
            record = self._by_id.get(str(selected["sample_id"]))
            if record is None or (
                record.source_id != str(selected["source_id"])
                or record.task_type != str(selected["task_type"])
            ):
                raise ValueError("scan manifest samples disagree with frozen selection")

    def __len__(self) -> int:
        return len(self.records)

    def select(self, sample_ids: Iterable[str]) -> ScanDataset:
        """Make a metadata-only view for source-disjoint fit/calibration splits."""

        selected = tuple(dict.fromkeys(map(str, sample_ids)))
        missing = set(selected).difference(self._by_id)
        if missing:
            raise ValueError(f"scan sample IDs not found: {sorted(missing)}")
        view = object.__new__(ScanDataset)
        view.root = self.root
        view.config = self.config
        view.split = self.split
        view.records = tuple(self._by_id[sample_id] for sample_id in selected)
        view._by_id = {record.sample_id: record for record in view.records}
        view.sample_ids = selected
        return view

    def sample_weights(self, *, by_source: bool = True) -> dict[str, float]:
        """Equal total weight per source, then equal samples within a source.

        A streaming learner can divide each weight by that sample's row count
        to also avoid overweighting long answers.  Normalization is left to
        the learner; every source's total weight is one.
        """

        count = Counter(record.source_id for record in self.records)
        return {
            record.sample_id: 1.0 / count[record.source_id] if by_source else 1.0
            for record in self.records
        }

    def load(
        self, sample_id: str, *, fields: Iterable[str] = DEFAULT_FIELDS
    ) -> ScanSample:
        """Load selected fields from one NPZ, validating their shared identity."""

        fields = tuple(dict.fromkeys(fields))
        unknown = set(fields).difference(SCAN_FIELDS)
        if unknown:
            raise ValueError(f"unsupported scan fields: {sorted(unknown)}")
        record = self._by_id[str(sample_id)]
        with np.load(record.path, allow_pickle=False) as stored:
            if int(stored["sample_scan_schema"]) != 1:
                raise ValueError("unsupported sample scan schema")
            for name, expected in (
                ("dataset_sample_id", record.sample_id),
                ("source_id", record.source_id),
                ("task_type", record.task_type),
            ):
                if str(stored[name]) != expected:
                    raise ValueError(f"scan {name} disagrees with manifest")
            if bool(stored["labels_used_for_capture"]):
                raise ValueError("sample scan violates the label firewall")
            bucket_names = tuple(stored["reanchor_bucket_name"].astype(str))
            if bucket_names != BUCKET_NAMES:
                raise ValueError("scan bucket order is incompatible")
            metadata = {
                name: int(stored[name])
                for name in (
                    "response_start",
                    "sequence_length",
                    "full_response_tokens",
                    "processed_response_tokens",
                    "local_window",
                )
            }
            positions = stored["route_row_position"]
            if positions.dtype.kind not in "iu":
                raise ValueError("scan predictor positions must be integer indices")
            positions = positions.astype(np.int64, copy=False)
            arrays = {name: stored[name] for name in fields}

        start = metadata["response_start"]
        length = metadata["sequence_length"]
        processed = metadata["processed_response_tokens"]
        if (
            not 1 <= start < length
            or processed != length - start
            or metadata["full_response_tokens"] < processed
            or metadata["local_window"] < 1
        ):
            raise ValueError("scan response coverage is inconsistent")
        if not np.array_equal(positions, np.arange(start - 1, length - 1)):
            raise ValueError(
                "scan predictor rows must cover contiguous response tokens"
            )
        shape = None
        for name, array in arrays.items():
            if name == "token_ids":
                if array.shape != (length,) or array.dtype.kind not in "iu":
                    raise ValueError("scan token IDs disagree with sequence length")
                continue
            is_bucket = name in BUCKET_FIELDS
            if (
                array.ndim != (4 if is_bucket else 3)
                or array.shape[2] != processed
                or min(array.shape[:2]) < 1
                or (is_bucket and array.shape[-1] != len(BUCKET_NAMES))
            ):
                raise ValueError(f"scan {name} has inconsistent layer/head/row axes")
            if shape is not None and array.shape[:3] != shape:
                raise ValueError("scan fields disagree on layer/head/row axes")
            shape = array.shape[:3]
            if name in {
                "reanchor_bucket_source_position",
                "reanchor_bucket_source_unit_id",
            }:
                if array.dtype.kind not in "iu" or np.any(array < -1):
                    raise ValueError(f"scan {name} must contain integer IDs or -1")
                if name == "reanchor_bucket_source_position" and np.any(
                    array > positions[None, None, :, None]
                ):
                    raise ValueError("scan source position is in the future")
            elif not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"scan {name} must be finite and nonnegative")
        return ScanSample(
            record.sample_id,
            record.source_id,
            record.task_type,
            start,
            positions,
            MappingProxyType(arrays),
            MappingProxyType(metadata),
        )


class ScanLabelStore:
    """Explicit post-hoc join to research-dataset labels; never a fit input."""

    def __init__(
        self, scans: ScanDataset, *, dataset_root: str | Path | None = None
    ) -> None:
        self.scans = scans
        self.dataset_root = Path(dataset_root or scans.config["dataset_root"])
        self._dataset = None

    def load(self, scan: ScanSample) -> np.ndarray:
        """Return labels for the predicted tokens, using -1 for unannotated rows."""

        if self._dataset is None:
            from research_dataset import open_research_dataset

            self._dataset = open_research_dataset(
                self.dataset_root, device="cpu", retain_embedded_labels=True
            )
            if str(self._dataset.manifest["split"]) != self.scans.split:
                raise ValueError("label dataset split disagrees with scan split")
        dataset = self._dataset
        if scan.sample_id not in self.scans.sample_ids:
            raise ValueError("label request belongs to another scan dataset")
        metadata = dataset.metadata(scan.sample_id)
        if str(metadata["source_id"]) != scan.source_id:
            raise ValueError("label source_id disagrees with scan source_id")
        if metadata.get("task_type") and metadata["task_type"] != scan.task_type:
            raise ValueError("label task_type disagrees with scan task_type")
        label_store = dataset.prepare_evaluation_labels([scan.sample_id])
        sample = dataset[scan.sample_id]
        try:
            labels = label_store.response_labels(sample).detach().cpu().numpy()
        finally:
            sample.release_attention()
        if labels.shape != (scan.metadata["full_response_tokens"],):
            raise ValueError("label response length disagrees with original scan")
        aligned = labels[scan.response_index]
        return np.where(np.isin(aligned, (0, 1)), aligned, -1).astype(np.int8)
