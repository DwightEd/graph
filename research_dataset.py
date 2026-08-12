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
