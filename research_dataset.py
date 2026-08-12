"""Canonical dataset access for attention-graph experiments.

This module only owns validated data access and evaluation labels. Graph
construction, feature learning, anomaly scoring, and visualization live in the
``attention_graph`` package.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cache import AttentionDataset, load_attention_sample, sha256


def open_research_dataset(
    split_root,
    *,
    device="cpu",
    verify_hashes=False,
    retain_embedded_labels=False,
):
    """Open canonical NPZ or the formal sparse PT cache without repacking it."""

    root = Path(split_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"split has no manifest: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("attention_cache_spec") is not None:
        return FormalResearchDataset(
            root,
            device=device,
            retain_labels=retain_embedded_labels,
        )
    return ResearchDataset(root, device=device, verify_hashes=verify_hashes)


class ResearchDataset:
    """Lazy access to one canonical attention split."""

    def __init__(self, split_root, *, device="cpu", verify_hashes=False):
        self.root = Path(split_root)
        self.device = device
        self.verify_hashes = bool(verify_hashes)
        self.attention_dataset = AttentionDataset(
            self.root, device=device, verify_hashes=verify_hashes
        )
        self.manifest = self.attention_dataset.manifest
        self.rows = {
            str(row["sample_id"]): row for row in self.attention_dataset.rows
        }
        self.source_to_ids: dict[str, list[str]] = {}
        for sample_id, row in self.rows.items():
            self.source_to_ids.setdefault(str(row["source_id"]), []).append(sample_id)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for sample_id in self.rows:
            yield self[sample_id]

    def __contains__(self, sample_id):
        return str(sample_id) in self.rows

    def __getitem__(self, sample_id):
        sample_id = str(sample_id)
        if sample_id not in self.rows:
            raise KeyError(sample_id)
        return ResearchSample(self, sample_id)

    @property
    def sample_ids(self):
        return list(self.rows)

    @property
    def source_ids(self):
        return list(self.source_to_ids)

    def samples_from_source(self, source_id):
        return [self[sample_id] for sample_id in self.source_to_ids.get(str(source_id), [])]

    def filter(self, **metadata):
        return [
            self[sample_id]
            for sample_id, row in self.rows.items()
            if all(row.get(key) == value for key, value in metadata.items())
        ]

    def labels(self):
        return LabelStore(self)


class ResearchSample:
    """One canonical attention sample plus lightweight metadata."""

    def __init__(self, dataset: ResearchDataset, sample_id: str):
        self.dataset = dataset
        self.sample_id = sample_id
        self.row = dataset.rows[sample_id]
        self._attention = None

    @property
    def source_id(self):
        return str(self.row["source_id"])

    @property
    def split(self):
        return self.row.get("split", self.dataset.manifest.get("split"))

    @property
    def task_type(self):
        return self.row.get("task_type")

    @property
    def data_source(self):
        return self.row.get("data_source")

    @property
    def generator_model(self):
        return self.row.get(
            "generator_model", self.dataset.manifest.get("generator_model")
        )

    @property
    def observer_model(self):
        return self.dataset.manifest.get("observer_model")

    @property
    def temperature(self):
        return self.row.get("temperature")

    @property
    def quality(self):
        return self.row.get("quality")

    @property
    def metadata(self):
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "split": self.split,
            "task_type": self.task_type,
            "data_source": self.data_source,
            "generator_model": self.generator_model,
            "observer_model": self.observer_model,
            "temperature": self.temperature,
            "quality": self.quality,
        }

    def attention(self):
        if self._attention is not None:
            return self._attention
        path = self.dataset.root / self.row["path"]
        if not path.is_file() or path.stat().st_size != int(self.row["bytes"]):
            raise ValueError("attention sample byte count does not match index")
        if self.dataset.verify_hashes and sha256(path) != self.row["sha256"]:
            raise ValueError("attention sample SHA256 does not match index")
        sample = load_attention_sample(
            path,
            sample_id=self.sample_id,
            source_id=self.source_id,
            attention_floor=self.dataset.attention_dataset.attention_floor,
            device=self.dataset.device,
        )
        if (
            sample.num_layers != int(self.dataset.manifest["num_layers"])
            or sample.num_heads != int(self.dataset.manifest["num_heads"])
        ):
            raise ValueError("attention geometry does not match split manifest")
        self._attention = sample
        return sample

    def release_attention(self):
        self._attention = None


class LabelStore:
    """Evaluation-only token labels from ``labels.jsonl``."""

    def __init__(self, dataset: ResearchDataset):
        self.dataset = dataset
        path = dataset.root / "labels.jsonl"
        if not path.is_file():
            raise ValueError("labels.jsonl is missing")
        expected_hash = dataset.manifest.get("labels_sha256")
        if expected_hash is not None and sha256(path) != expected_hash:
            raise ValueError("labels_sha256 does not match labels.jsonl")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.rows = {str(row["sample_id"]): row for row in rows}
        if len(self.rows) != len(rows) or set(self.rows) != set(dataset.rows):
            raise ValueError("label/canonical sample ID sets do not match")

    def positive_runs(self, sample_id, *, response_count=None):
        runs = self.rows[str(sample_id)].get("positive_runs", [])
        previous_end = 0
        normalized = []
        if response_count is None:
            response_count = self.dataset[str(sample_id)].attention().num_response_tokens
        for run in runs:
            if len(run) != 2:
                raise ValueError("each positive run must contain [start, end)")
            start, end = map(int, run)
            if not 0 <= start < end <= response_count or start < previous_end:
                raise ValueError("positive runs must be sorted valid response-relative spans")
            normalized.append([start, end])
            previous_end = end
        return normalized

    def response_labels(self, sample: ResearchSample):
        self._check_dataset(sample)
        attention = sample.attention()
        labels = torch.zeros(
            attention.num_response_tokens,
            dtype=torch.long,
            device=attention.token_ids.device,
        )
        for start, end in self.positive_runs(
            sample.sample_id, response_count=attention.num_response_tokens
        ):
            labels[start:end] = 1
        return labels

    def _check_dataset(self, sample: ResearchSample):
        if sample.dataset is not self.dataset:
            raise ValueError("labels belong to a different dataset")


class FormalResearchDataset:
    """Lazy adapter over the existing sparse ``attention_*.pt`` archive.

    No attention tensor is copied or re-serialized.  The adapter normalizes the
    in-memory object to ``AttentionSample`` and keeps embedded labels sealed
    until ``labels()`` is explicitly requested after pattern discovery.
    """

    def __init__(self, split_root, *, device="cpu", retain_labels=False):
        from archive import _formal_manifest

        self.root = Path(split_root)
        raw_manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        split = str(raw_manifest["attention_cache_spec"]["split"]).casefold()
        formal_manifest, spec, files = _formal_manifest(self.root, split)
        self.device = device
        self.split_name = split
        self.spec = spec
        self.retain_labels = bool(retain_labels)
        self._label_cache = {}
        self.manifest = {
            "schema": formal_manifest["attention_cache_spec"][
                "attention_cache_schema"
            ],
            "split": split,
            "num_layers": int(spec["num_hidden_layers"]),
            "num_heads": int(spec["num_attention_heads"]),
            "attention_floor": float(spec["attention_floor"]),
            "count": len(files),
            "generator_model": spec.get("generator_model"),
            "observer_model": Path(str(spec.get("model_path", ""))).name or None,
        }
        self.rows = {}
        for path, digest in files:
            stem = path.stem
            sample_id = stem[len("attention_"):] if stem.startswith("attention_") else stem
            if not sample_id or sample_id in self.rows:
                raise ValueError("formal cache file names do not identify unique samples")
            self.rows[sample_id] = {
                "sample_id": sample_id,
                "path": path,
                "sha256": digest,
            }

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for sample_id in self.rows:
            yield self[sample_id]

    def __contains__(self, sample_id):
        return str(sample_id) in self.rows

    def __getitem__(self, sample_id):
        sample_id = str(sample_id)
        if sample_id not in self.rows:
            raise KeyError(sample_id)
        return FormalResearchSample(self, sample_id)

    @property
    def sample_ids(self):
        return list(self.rows)

    def labels(self):
        if not self.retain_labels:
            raise RuntimeError("embedded labels were not retained for this dataset")
        if len(self._label_cache) != len(self.rows):
            raise RuntimeError(
                "formal labels become available only after every attention sample "
                "has been processed"
            )
        return FormalLabelStore(self)


class FormalResearchSample:
    def __init__(self, dataset: FormalResearchDataset, sample_id: str):
        self.dataset = dataset
        self.sample_id = sample_id
        self.row = dataset.rows[sample_id]
        self._attention = None
        self._metadata = None

    def _load(self):
        if self._attention is not None:
            return
        from archive import _load_formal

        sample, labels, payload = _load_formal(
            self.row["path"],
            self.row["sha256"],
            split=self.dataset.split_name,
            spec=self.dataset.spec,
            return_payload=True,
        )
        if sample.sample_id != self.sample_id:
            raise ValueError(
                "formal cache response_id does not match its attention file name"
            )
        for name in (
            "token_ids",
            "attention_diagonal",
            "response_row_ptr",
            "response_column_indices",
            "response_values",
        ):
            setattr(sample, name, getattr(sample, name).to(self.dataset.device))
        self._attention = sample
        self._metadata = {
            "source_id": sample.source_id,
            "task_type": payload.get("task_type", self.dataset.spec.get("task_type")),
            "data_source": payload.get(
                "data_source", payload.get("source", self.dataset.spec.get("data_source"))
            ),
            "generator_model": payload.get(
                "generator_model", self.dataset.spec.get("generator_model")
            ),
            "temperature": payload.get("temperature"),
            "quality": payload.get("quality"),
        }
        if self.dataset.retain_labels:
            self.dataset._label_cache[self.sample_id] = (
                labels[sample.response_idx:].to(dtype=torch.long, device="cpu")
            )

    @property
    def source_id(self):
        self._load()
        return str(self._metadata["source_id"])

    @property
    def split(self):
        return self.dataset.split_name

    @property
    def task_type(self):
        self._load()
        return self._metadata["task_type"]

    @property
    def data_source(self):
        self._load()
        return self._metadata["data_source"]

    @property
    def generator_model(self):
        self._load()
        return self._metadata["generator_model"]

    def attention(self):
        self._load()
        return self._attention

    def release_attention(self):
        self._attention = None


class FormalLabelStore:
    def __init__(self, dataset: FormalResearchDataset):
        self.dataset = dataset

    def response_labels(self, sample: FormalResearchSample):
        if sample.dataset is not self.dataset:
            raise ValueError("labels belong to a different dataset")
        return self.dataset._label_cache[sample.sample_id]
