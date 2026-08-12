"""Paper-style before/after projection of frozen attention-GNN node states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .graph import build_attention_graph
from .score import load_checkpoint


class _Reservoir:
    def __init__(self, capacity: int, embedding_dim: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.before = np.empty((capacity, embedding_dim), dtype=np.float32)
        self.after = np.empty((capacity, embedding_dim), dtype=np.float32)
        self.sample_id = np.empty(capacity, dtype=object)
        self.source_id = np.empty(capacity, dtype=object)
        self.token_index = np.empty(capacity, dtype=np.int32)
        self.response_count = np.empty(capacity, dtype=np.int32)
        self.count = 0

    def add(self, before, after, sample_id, source_id, response_count):
        for index in range(len(before)):
            slot = self.count if self.count < self.capacity else int(self.rng.integers(self.count + 1))
            self.count += 1
            if slot >= self.capacity:
                continue
            self.before[slot] = before[index]
            self.after[slot] = after[index]
            self.sample_id[slot] = sample_id
            self.source_id[slot] = source_id
            self.token_index[slot] = index
            self.response_count[slot] = response_count

    def take(self, count):
        if self.count < count:
            raise ValueError("domain contains fewer response tokens than requested")
        size = min(self.count, self.capacity)
        selected = np.arange(size) if size == count else self.rng.choice(size, count, replace=False)
        return {
            "before": self.before[selected], "after": self.after[selected],
            "sample_id": self.sample_id[selected], "source_id": self.source_id[selected],
            "token_index": self.token_index[selected], "response_count": self.response_count[selected],
        }


def _joint_projection(before: np.ndarray, after: np.ndarray, *, perplexity: float, seed: int):
    embeddings = np.concatenate((before, after), axis=0)
    if len(embeddings) < 3:
        raise ValueError("at least three paired response nodes are required")
    scaled = StandardScaler().fit_transform(embeddings)
    pca_dim = min(50, scaled.shape[1], len(scaled) - 1)
    working = PCA(n_components=pca_dim, random_state=seed).fit_transform(scaled)
    actual_perplexity = min(float(perplexity), len(working) - 1.0)
    coordinates = TSNE(
        n_components=2, perplexity=actual_perplexity, init="pca",
        learning_rate="auto", max_iter=1500, random_state=seed,
    ).fit_transform(working)
    return coordinates[:len(before)], coordinates[len(before):], actual_perplexity


def _token_labels(dataset, sample_id, token_index, response_count):
    labels = dataset.labels()
    output = np.zeros(len(sample_id), dtype=np.int8)
    runs_by_sample = {}
    for row, (identifier, index, count) in enumerate(zip(sample_id, token_index, response_count)):
        if identifier not in runs_by_sample:
            runs_by_sample[identifier] = labels.positive_runs(
                identifier, response_count=int(count)
            )
        for start, end in runs_by_sample[identifier]:
            if start <= int(index) < end:
                output[row] = 1
                break
    return output


def _plot(before, after, domain, label, source_domain, target_domain, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {(0, 0): "forestgreen", (0, 1): "red", (1, 0): "royalblue", (1, 1): "orange"}
    markers = {0: "o", 1: "x"}
    names = {
        (0, 0): "Source normal", (0, 1): "Source anomaly",
        (1, 0): "Target normal", (1, 1): "Target anomaly",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.80, wspace=0.20)
    limits = np.concatenate((before, after), axis=0)
    pad = 0.04 * np.maximum(np.ptp(limits, axis=0), 1.0)
    xlim = (limits[:, 0].min() - pad[0], limits[:, 0].max() + pad[0])
    ylim = (limits[:, 1].min() - pad[1], limits[:, 1].max() + pad[1])
    for axis, coordinates, title in (
        (axes[0], before, "(a) Before message passing ($h_0$)"),
        (axes[1], after, "(b) After RP/RR message passing ($h_K$)"),
    ):
        for key in ((0, 0), (1, 0), (0, 1), (1, 1)):
            mask = (domain == key[0]) & (label == key[1])
            if mask.any():
                axis.scatter(
                    coordinates[mask, 0], coordinates[mask, 1], marker=markers[key[0]],
                    c=colors[key], s=13 if key[1] == 0 else 25,
                    alpha=0.52 if key[1] == 0 else 0.9, linewidths=1.0,
                    rasterized=True,
                )
        axis.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2", xlim=xlim, ylim=ylim)
        axis.text(0.5, -0.18, f"{source_domain} = source, {target_domain} = target",
                  transform=axis.transAxes, ha="center", fontsize=9)
    legend = [
        Line2D([], [], marker=markers[key[0]], color=colors[key], linestyle="None",
               markersize=7, label=names[key])
        for key in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]
    figure.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.98),
        ncol=4, frameon=False,
    )
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)


class EmbeddingShiftVisualizer:
    """Project the same frozen graph-encoder nodes before and after RP/RR aggregation."""

    def __init__(
        self,
        dataset,
        *,
        checkpoint,
        domain_field,
        source_domain,
        target_domain,
        output_dir,
        device="cuda",
        max_nodes_per_domain=5000,
        perplexity=30.0,
        seed=0,
    ):
        self.dataset = dataset
        self.checkpoint = checkpoint
        self.domain_field = domain_field
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.output_dir = Path(output_dir)
        self.device = device
        self.max_nodes_per_domain = int(max_nodes_per_domain)
        self.perplexity = float(perplexity)
        self.seed = int(seed)
        if self.source_domain == self.target_domain:
            raise ValueError("source and target domains must be different")
        if self.max_nodes_per_domain < 2 or self.perplexity <= 0:
            raise ValueError("max nodes must be at least two and perplexity must be positive")

    def _collect(self, model, graph_config):
        reservoirs = {
            self.source_domain: _Reservoir(self.max_nodes_per_domain, model.embedding_dim, np.random.default_rng(self.seed)),
            self.target_domain: _Reservoir(self.max_nodes_per_domain, model.embedding_dim, np.random.default_rng(self.seed + 1)),
        }
        selected = [
            sample for sample in self.dataset
            if str(getattr(sample, self.domain_field)) in reservoirs
        ]
        for sample in tqdm(selected, desc="encode RP/RR node stages", unit="sample"):
            graph = build_attention_graph(sample.attention(), graph_config).to(self.device)
            with torch.no_grad():
                before, after = model.encode_stages(graph)
            response_count = graph.num_nodes - graph.response_idx
            reservoirs[str(getattr(sample, self.domain_field))].add(
                before[graph.response_idx:].float().cpu().numpy(),
                after[graph.response_idx:].float().cpu().numpy(),
                sample.sample_id, sample.source_id, response_count,
            )
            sample.release_attention()
        count = min(min(reservoir.count, reservoir.capacity) for reservoir in reservoirs.values())
        if count < 2:
            raise ValueError("each domain needs at least two response nodes")
        source = reservoirs[self.source_domain].take(count)
        target = reservoirs[self.target_domain].take(count)
        return source, target

    def run(self):
        model, _calibrator, graph_config, checkpoint = load_checkpoint(self.checkpoint, device=self.device)
        if not model.encoder.layers:
            raise ValueError("message_steps must be positive for before/after visualization")
        geometry = checkpoint.get("attention_geometry")
        if geometry is None:
            raise ValueError("checkpoint lacks exact attention geometry; retrain it with the current code")
        expected_geometry = {
            "num_layers": int(self.dataset.manifest["num_layers"]),
            "num_heads": int(self.dataset.manifest["num_heads"]),
            "alignment": self.dataset.manifest["alignment"],
            "attention_floor": float(self.dataset.manifest["attention_floor"]),
            "observer_model": self.dataset.manifest.get("observer_model"),
            "generator_model": self.dataset.manifest.get("generator_model"),
        }
        if geometry != expected_geometry:
            raise ValueError("checkpoint and canonical split have different attention geometry")
        source, target = self._collect(model, graph_config)
        before = np.concatenate((source["before"], target["before"]), axis=0)
        after = np.concatenate((source["after"], target["after"]), axis=0)
        domain = np.concatenate((np.zeros(len(source["before"]), dtype=np.int8), np.ones(len(target["before"]), dtype=np.int8)))
        sample_id = np.concatenate((source["sample_id"], target["sample_id"])).astype(str)
        source_id = np.concatenate((source["source_id"], target["source_id"])).astype(str)
        token_index = np.concatenate((source["token_index"], target["token_index"]))
        response_count = np.concatenate((source["response_count"], target["response_count"]))
        coordinates_before, coordinates_after, actual_perplexity = _joint_projection(
            before, after, perplexity=self.perplexity, seed=self.seed
        )
        label = _token_labels(self.dataset, sample_id, token_index, response_count)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure = self.output_dir / "embedding_shift_tsne.png"
        data = self.output_dir / "embedding_shift.npz"
        summary_path = self.output_dir / "summary.json"
        names = np.where(domain == 0, self.source_domain, self.target_domain)
        _plot(coordinates_before, coordinates_after, domain, label, self.source_domain, self.target_domain, figure)
        np.savez_compressed(
            data,
            schema=np.asarray("attention-graph-embedding-shift-v1"),
            coordinates_before=coordinates_before, coordinates_after=coordinates_after,
            embedding_before=before, embedding_after=after, label=label, domain=domain,
            domain_name=names, sample_id=sample_id, source_id=source_id, token_index=token_index,
            domain_field=np.asarray(self.domain_field), source_domain=np.asarray(self.source_domain),
            target_domain=np.asarray(self.target_domain), labels_read_during=np.asarray("coloring_only"),
            perplexity=np.asarray(actual_perplexity), seed=np.asarray(self.seed),
            claim_scope=np.asarray("message-passing representation shift, not domain alignment"),
        )
        summary = {
            "schema": "attention-graph-embedding-shift-v1", "figure": str(figure), "data": str(data),
            "checkpoint": str(self.checkpoint), "domain_field": self.domain_field,
            "source_domain": self.source_domain, "target_domain": self.target_domain,
            "source_nodes": int(len(source["before"])), "target_nodes": int(len(target["before"])),
            "source_samples": int(len(set(source["sample_id"]))),
            "target_samples": int(len(set(target["sample_id"]))),
            "source_groups": int(len(set(source["source_id"]))),
            "target_groups": int(len(set(target["source_id"]))),
            "source_prevalence": float(label[domain == 0].mean()), "target_prevalence": float(label[domain == 1].mean()),
            "representation": {"before": "encoder_node_initialization_h0", "after": "RP_RR_message_passing_hK"},
            "joint_projection": True, "labels_read_during": "coloring_only", "perplexity": actual_perplexity,
            "seed": self.seed, "checkpoint_best_epoch": checkpoint.get("best_epoch"),
            "claim_scope": "message-passing representation shift, not domain alignment",
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return {**summary, "summary": str(summary_path)}
