"""Unified sample-level access for research experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cache import AttentionDataset, load_attention_sample, sha256
from features import load_hidden_features, load_node_features, load_token_stats


STRUCTURAL_FEATURE_NAMES = (
    "incoming_mass",
    "prompt_mass_share",
    "normalized_entropy",
    "history_lag",
    "in_degree",
    "prompt_degree",
    "history_degree",
    "in_density",
    "prompt_density",
    "history_density",
    "history_edge_share",
    "channel_edge_density",
)


def aggregate_attention_relations(attention, edges):
    """Merge channel-level CSR entries into unique source->target relations.

    The returned ``weight`` is the sum of retained layer/head weights divided by
    the total number of attention channels. Missing channels therefore contribute
    zero, matching the reconstructed sparse-attention semantics.
    """
    device = edges["weight"].device
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    empty_float = torch.empty(0, dtype=torch.float32, device=device)
    if edges["weight"].numel() == 0:
        return {
            "source": empty_long,
            "target": empty_long,
            "weight": empty_float,
            "channel_count": empty_long,
            "edge_type": empty_long,
        }

    source = edges["source"].long()
    target = edges["target"].long()
    weight = edges["weight"].float()
    if bool(((source < 0) | (source >= target) | (target >= attention.num_tokens)).any()):
        raise ValueError("attention relations must point from earlier valid tokens")

    pair_key = target * attention.num_tokens + source
    unique_key, inverse = torch.unique(pair_key, sorted=True, return_inverse=True)

    relation_weight = torch.zeros(
        unique_key.numel(), dtype=torch.float32, device=device
    )
    relation_weight.index_add_(0, inverse, weight)
    relation_weight /= float(attention.num_channels)

    channel_count = torch.bincount(
        inverse, minlength=unique_key.numel()
    ).to(torch.long)
    relation_target = torch.div(
        unique_key, attention.num_tokens, rounding_mode="floor"
    ).long()
    relation_source = unique_key.remainder(attention.num_tokens).long()
    edge_type = (relation_source >= attention.response_idx).long()

    return {
        "source": relation_source,
        "target": relation_target,
        "weight": relation_weight,
        "channel_count": channel_count,
        "edge_type": edge_type,
    }


def structural_features_from_relations(attention, relations):
    """Return [response_tokens, 12] structural graph features."""
    response_idx = attention.response_idx
    response_count = attention.num_response_tokens
    device = relations["weight"].device
    features = torch.zeros(
        (response_count, len(STRUCTURAL_FEATURE_NAMES)),
        dtype=torch.float32,
        device=device,
    )
    if relations["weight"].numel() == 0:
        return features

    source = relations["source"].long()
    target = relations["target"].long()
    weight = relations["weight"].float()
    channel_count = relations["channel_count"].float()
    rows = target - response_idx

    if bool(((rows < 0) | (rows >= response_count)).any()):
        raise ValueError("relation targets must be response tokens")

    prompt = source < response_idx
    history = ~prompt

    total_mass = torch.zeros(response_count, dtype=torch.float32, device=device)
    total_mass.index_add_(0, rows, weight)

    prompt_mass = torch.zeros_like(total_mass)
    prompt_mass.index_add_(0, rows[prompt], weight[prompt])
    prompt_share = torch.zeros_like(total_mass)
    nonempty = total_mass > 0
    prompt_share[nonempty] = prompt_mass[nonempty] / total_mass[nonempty]

    in_degree = torch.bincount(rows, minlength=response_count).float()
    prompt_degree = torch.bincount(rows[prompt], minlength=response_count).float()
    history_degree = torch.bincount(rows[history], minlength=response_count).float()

    probabilities = weight / total_mass[rows]
    entropy = torch.zeros_like(total_mass)
    entropy.index_add_(0, rows, -probabilities * probabilities.log())
    normalized_entropy = torch.zeros_like(total_mass)
    multiple = in_degree > 1
    normalized_entropy[multiple] = entropy[multiple] / in_degree[multiple].log()

    history_mass = torch.zeros_like(total_mass)
    history_mass.index_add_(0, rows[history], weight[history])
    history_lag_mass = torch.zeros_like(total_mass)
    if bool(history.any()):
        lag = (target[history] - source[history]).float()
        history_lag_mass.index_add_(
            0,
            rows[history],
            weight[history] * lag / max(response_count - 1, 1),
        )
    history_lag = torch.zeros_like(total_mass)
    has_history_mass = history_mass > 0
    history_lag[has_history_mass] = (
        history_lag_mass[has_history_mass] / history_mass[has_history_mass]
    )

    response_position = torch.arange(
        response_count, dtype=torch.float32, device=device
    )
    absolute_target = response_idx + response_position
    in_density = in_degree / absolute_target.clamp_min(1.0)
    prompt_density = prompt_degree / float(max(response_idx, 1))

    history_density = torch.zeros_like(total_mass)
    has_history = response_position > 0
    history_density[has_history] = (
        history_degree[has_history] / response_position[has_history]
    )

    history_edge_share = torch.zeros_like(total_mass)
    has_edges = in_degree > 0
    history_edge_share[has_edges] = history_degree[has_edges] / in_degree[has_edges]

    channel_degree = torch.zeros_like(total_mass)
    channel_degree.index_add_(0, rows, channel_count)
    channel_edge_density = channel_degree / (
        float(attention.num_channels) * absolute_target.clamp_min(1.0)
    )

    features = torch.stack(
        (
            total_mass,
            prompt_share,
            normalized_entropy,
            history_lag,
            in_degree,
            prompt_degree,
            history_degree,
            in_density,
            prompt_density,
            history_density,
            history_edge_share,
            channel_edge_density,
        ),
        dim=1,
    )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("structural graph features must be finite")
    return features


def structural_features_from_edges(attention, edges):
    """Decode channel edges into unique relations and structural node features."""
    relations = aggregate_attention_relations(attention, edges)
    return structural_features_from_relations(attention, relations)


class ResearchDataset:
    """Join canonical features and optional graph caches by sample_id."""

    def __init__(self, split_root, graph_roots=None, device="cpu", verify_hashes=False):
        self.root = Path(split_root)
        self.device = device
        self.verify_hashes = verify_hashes
        self.attention_dataset = AttentionDataset(
            self.root, device=device, verify_hashes=verify_hashes
        )
        self.manifest = self.attention_dataset.manifest
        self.rows = {
            str(row["sample_id"]): row for row in self.attention_dataset.rows
        }
        self.graph_roots = {
            name: Path(path) for name, path in (graph_roots or {}).items()
        }
        self.graph_rows = {
            name: self._graph_index(root) for name, root in self.graph_roots.items()
        }

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
        return [
            self[sample_id]
            for sample_id in self.source_to_ids.get(str(source_id), [])
        ]

    def filter(self, **metadata):
        """Return samples whose index metadata exactly matches all requested values."""
        return [
            self[sample_id]
            for sample_id, row in self.rows.items()
            if all(row.get(key) == value for key, value in metadata.items())
        ]

    def label_store(self, path=None):
        """Open the split's evaluation labels, or an explicitly supplied sidecar."""
        return LabelStore(self.root / "labels.jsonl" if path is None else path)


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

    def node_features(self, mode="attention", attention=None):
        attention = self.attention() if attention is None else attention
        return load_node_features(self.dataset.root, attention, mode=mode)

    def graph(self, name):
        root = self.dataset.graph_roots[name]
        row = self.dataset.graph_rows[name][self.sample_id]
        path = root / row["path"]
        if (
            self.dataset.verify_hashes
            and "sha256" in row
            and sha256(path) != row["sha256"]
        ):
            raise ValueError("graph SHA256 does not match index")
        graph = torch.load(path, map_location=self.dataset.device, weights_only=True)
        if int(graph["num_nodes"]) != self.attention().num_tokens:
            raise ValueError("graph and attention token counts do not match")
        return graph

    def attention_edges(self, attention=None):
        """Decode canonical CSR to layer/head/source/target/weight vectors."""
        sample = self.attention() if attention is None else attention
        response_count = sample.num_response_tokens
        counts = sample.response_row_ptr[1:] - sample.response_row_ptr[:-1]
        row = torch.repeat_interleave(
            torch.arange(counts.numel(), device=counts.device), counts
        )
        channel = row // response_count
        return {
            "layer": channel // sample.num_heads,
            "head": channel % sample.num_heads,
            "source": sample.response_column_indices.long(),
            "target": sample.response_idx + row % response_count,
            "weight": sample.response_values,
        }

    def relation_edges(self, attention=None, edges=None):
        """Return unique source->target relations aggregated across layer/head channels."""
        attention = self.attention() if attention is None else attention
        edges = self.attention_edges(attention=attention) if edges is None else edges
        return aggregate_attention_relations(attention, edges)

    def structural_features(self, attention=None, relations=None):
        """Return the 12-D structural state of every response token."""
        attention = self.attention() if attention is None else attention
        if relations is None:
            relations = self.relation_edges(attention=attention)
        return structural_features_from_relations(attention, relations)

    def graph_view(self, labels=None):
        """Load one self-contained token-graph view for analysis/visualization."""
        attention = self.attention()
        raw_edges = self.attention_edges(attention=attention)
        relations = aggregate_attention_relations(attention, raw_edges)
        features = structural_features_from_relations(attention, relations)
        positive_runs = []
        response_labels = torch.zeros(
            attention.num_response_tokens, dtype=torch.long
        )
        if labels is not None:
            positive_runs = labels.positive_runs(self.sample_id)
            response_labels = labels.response_labels(self, attention=attention).cpu()
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "metadata": self.metadata,
            "token_ids": attention.token_ids.detach().cpu(),
            "response_idx": attention.response_idx,
            "num_tokens": attention.num_tokens,
            "num_response_tokens": attention.num_response_tokens,
            "num_channels": attention.num_channels,
            "raw_edges": {key: value.detach().cpu() for key, value in raw_edges.items()},
            "relations": {key: value.detach().cpu() for key, value in relations.items()},
            "structural_feature_names": STRUCTURAL_FEATURE_NAMES,
            "response_features": features.detach().cpu(),
            "positive_runs": positive_runs,
            "response_labels": response_labels,
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
        return self.rows[str(sample_id)]["positive_runs"]

    def response_labels(self, sample: ResearchSample, attention=None):
        attention = sample.attention() if attention is None else attention
        labels = torch.zeros(attention.num_response_tokens, dtype=torch.long)
        for start, end in self.positive_runs(sample.sample_id):
            labels[start:end] = 1
        return labels

    def token_labels(self, sample: ResearchSample, attention=None):
        attention = sample.attention() if attention is None else attention
        labels = torch.zeros(attention.num_tokens, dtype=torch.long)
        labels[attention.response_idx:] = self.response_labels(
            sample, attention=attention
        )
        return labels
