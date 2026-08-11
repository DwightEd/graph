"""Unified sample-level access for research experiments."""

import json
from pathlib import Path

import torch

from cache import AttentionDataset, load_attention_sample, sha256
from features import load_hidden_features, load_node_features, load_token_stats


class ResearchDataset:
    """Join canonical features and one or more graph caches by sample_id."""

    def __init__(self, split_root, graph_roots=None, device="cpu", verify_hashes=False):
        self.root = Path(split_root)
        self.device = device
        self.verify_hashes = verify_hashes
        self.attention_dataset = AttentionDataset(self.root, device=device, verify_hashes=verify_hashes)
        self.manifest = self.attention_dataset.manifest
        self.rows = {str(row["sample_id"]): row for row in self.attention_dataset.rows}
        self.graph_roots = {name: Path(path) for name, path in (graph_roots or {}).items()}
        self.graph_rows = {name: self._graph_index(root) for name, root in self.graph_roots.items()}

        self.source_to_ids = {}
        for sample_id, row in self.rows.items():
            self.source_to_ids.setdefault(str(row["source_id"]), []).append(sample_id)

    @staticmethod
    def _graph_index(root):
        with (root / "index.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return {str(row["sample_id"]): row for row in rows}

    def __len__(self):
        return len(self.rows)

    def __contains__(self, sample_id):
        return str(sample_id) in self.rows

    def __getitem__(self, sample_id):
        sample_id = str(sample_id)
        if sample_id not in self.rows:
            raise KeyError(sample_id)
        return ResearchSample(self, sample_id)

    def __iter__(self):
        for sample_id in self.rows:
            yield self[sample_id]

    @property
    def sample_ids(self):
        return list(self.rows)

    @property
    def source_ids(self):
        return list(self.source_to_ids)

    def samples_from_source(self, source_id):
        return [self[sample_id] for sample_id in self.source_to_ids.get(str(source_id), [])]

    def filter(self, **metadata):
        """Return samples whose index metadata exactly matches all requested values."""
        return [
            self[sample_id]
            for sample_id, row in self.rows.items()
            if all(row.get(key) == value for key, value in metadata.items())
        ]


class ResearchSample:
    def __init__(self, dataset: ResearchDataset, sample_id: str):
        self.dataset = dataset
        self.sample_id = sample_id
        self.row = dataset.rows[sample_id]

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
        """Original task corpus, e.g. CNN/DM, MARCO or Yelp; not a model name."""
        return self.row.get("data_source")

    @property
    def generator_model(self):
        """Model that generated the RAGTruth response."""
        return self.row.get("generator_model", self.dataset.manifest.get("generator_model"))

    @property
    def observer_model(self):
        """White-box model whose internal states/attention were extracted."""
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
        path = self.dataset.root / self.row["path"]
        if self.dataset.verify_hashes and sha256(path) != self.row["sha256"]:
            raise ValueError("attention sample SHA256 does not match index")
        return load_attention_sample(
            path,
            sample_id=self.sample_id,
            source_id=self.source_id,
            attention_floor=self.dataset.attention_dataset.attention_floor,
            device=self.dataset.device,
        )

    def hidden(self):
        return load_hidden_features(
            self.dataset.root / "hidden" / f"{self.sample_id}.npz",
            device=self.dataset.device,
        )

    def stats(self):
        return load_token_stats(
            self.dataset.root / "token_stats" / f"{self.sample_id}.npz",
            device=self.dataset.device,
        )

    def node_features(self, mode="attention"):
        return load_node_features(self.dataset.root, self.attention(), mode=mode)

    def graph(self, name):
        root = self.dataset.graph_roots[name]
        row = self.dataset.graph_rows[name][self.sample_id]
        path = root / row["path"]
        if self.dataset.verify_hashes and "sha256" in row and sha256(path) != row["sha256"]:
            raise ValueError("graph SHA256 does not match index")
        graph = torch.load(path, map_location=self.dataset.device, weights_only=True)
        if int(graph["num_nodes"]) != self.attention().num_tokens:
            raise ValueError("graph and attention token counts do not match")
        return graph

    def attention_edges(self):
        """Decode CSR to human-readable layer/head/source/target/weight vectors."""
        sample = self.attention()
        R = sample.num_response_tokens
        counts = sample.response_row_ptr[1:] - sample.response_row_ptr[:-1]
        row = torch.repeat_interleave(
            torch.arange(counts.numel(), device=counts.device), counts
        )
        channel = row // R
        return {
            "layer": channel // sample.num_heads,
            "head": channel % sample.num_heads,
            "source": sample.response_column_indices.long(),
            "target": sample.response_idx + row % R,
            "weight": sample.response_values,
        }

    @property
    def response_slice(self):
        sample = self.attention()
        return slice(sample.response_idx, sample.num_tokens)


class LabelStore:
    """Optional evaluation-only access to token labels."""

    def __init__(self, path):
        with Path(path).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.rows = {str(row["sample_id"]): row for row in rows}

    def positive_runs(self, sample_id):
        """Return response-relative hallucination token intervals [start, end)."""
        return self.rows[str(sample_id)]["positive_runs"]

    def response_labels(self, sample: ResearchSample):
        attention = sample.attention()
        labels = torch.zeros(attention.num_response_tokens, dtype=torch.long)
        for start, end in self.positive_runs(sample.sample_id):
            labels[start:end] = 1
        return labels

    def token_labels(self, sample: ResearchSample):
        attention = sample.attention()
        labels = torch.zeros(attention.num_tokens, dtype=torch.long)
        labels[attention.response_idx:] = self.response_labels(sample)
        return labels
