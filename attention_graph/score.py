"""Frozen encoder inference and unsupervised token anomaly scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .graph import GraphBuildConfig, build_attention_graph
from .model import MaskedAttentionAutoencoder, full_view, reconstruction_energy_by_node, target_masked_view

SCORE_COMPONENTS = (
    "support_rp",
    "support_rr",
    "weight_rp",
    "weight_rr",
    "distribution",
    "node",
)


@dataclass(frozen=True)
class RobustResidualCalibrator:
    center: tuple[float, ...]
    scale: tuple[float, ...]
    components: tuple[str, ...] = SCORE_COMPONENTS

    @classmethod
    def fit(cls, matrix: np.ndarray):
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(SCORE_COMPONENTS) or not len(values):
            raise ValueError("calibration matrix has the wrong shape")
        if not np.isfinite(values).all():
            raise ValueError("calibration residuals must be finite")
        center = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - center), axis=0)
        std = values.std(axis=0)
        scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
        return cls(tuple(center.tolist()), tuple(scale.tolist()))

    def transform(self, matrix: np.ndarray):
        values = np.asarray(matrix, dtype=np.float64)
        center = np.asarray(self.center)
        scale = np.asarray(self.scale)
        if values.ndim != 2 or values.shape[1] != len(center):
            raise ValueError("residual matrix has the wrong shape")
        z = (values - center) / scale
        score = np.maximum(z, 0.0).mean(axis=1)
        return z.astype(np.float32), score.astype(np.float32)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(
            center=tuple(map(float, value["center"])),
            scale=tuple(map(float, value["scale"])),
            components=tuple(value.get("components", SCORE_COMPONENTS)),
        )


def score_graph_raw(model, graph, *, target_block_size=1, seed=0):
    """Return full embeddings and leave-target-out reconstruction residuals."""
    if target_block_size < 1:
        raise ValueError("target_block_size must be positive")
    model.eval()
    response_nodes = torch.nonzero(graph.response_mask, as_tuple=False).flatten()
    with torch.no_grad():
        hidden = model.encode(graph, full_view(graph))
        embeddings = hidden[response_nodes].detach().float().cpu().numpy()

    residuals = np.zeros((len(response_nodes), len(SCORE_COMPONENTS)), dtype=np.float32)
    node_to_row = torch.full(
        (graph.num_nodes,), -1, dtype=torch.long, device=graph.node_attr.device
    )
    node_to_row[response_nodes] = torch.arange(len(response_nodes), device=graph.node_attr.device)
    rng = torch.Generator(device=graph.node_attr.device)

    for block_index, start in enumerate(range(0, len(response_nodes), target_block_size)):
        targets = response_nodes[start : start + target_block_size]
        view = target_masked_view(graph, targets)
        rng.manual_seed(int(seed + block_index * 1_000_003))
        with torch.no_grad():
            energy = reconstruction_energy_by_node(
                model,
                graph,
                view,
                max_support_edges=None,
                max_weight_traces=None,
                max_distribution_groups=None,
                generator=rng,
            )
        rows = node_to_row[targets].cpu().numpy()
        residuals[rows] = np.stack(
            [energy[name][targets].detach().float().cpu().numpy() for name in SCORE_COMPONENTS],
            axis=1,
        )
    return embeddings, residuals


def score_dataset(
    dataset,
    model,
    calibrator,
    *,
    graph_config: GraphBuildConfig,
    target_block_size=1,
    seed=0,
):
    """Score every response token without opening evaluation labels."""
    records = []
    for sample_index, sample_id in enumerate(dataset.sample_ids):
        sample = dataset[sample_id]
        graph = build_attention_graph(sample.attention(), graph_config)
        embeddings, residuals = score_graph_raw(
            model, graph, target_block_size=target_block_size, seed=seed + sample_index * 1009
        )
        z, scores = calibrator.transform(residuals)
        for token_index in range(len(scores)):
            record = {
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "token_index": token_index,
                "score": float(scores[token_index]),
                "embedding": embeddings[token_index].tolist(),
            }
            for column, name in enumerate(SCORE_COMPONENTS):
                record[f"residual_{name}"] = float(residuals[token_index, column])
                record[f"z_{name}"] = float(z[token_index, column])
            for name in ("task_type", "data_source", "generator_model"):
                record[name] = getattr(sample, name, None)
            records.append(record)
        sample.release_attention()
    return records


def load_checkpoint(path, *, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("schema") != "attention-graph-unsupervised-v1":
        raise ValueError("unsupported checkpoint schema")
    model_config = checkpoint["model_config"]
    model = MaskedAttentionAutoencoder(**model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    calibrator = RobustResidualCalibrator.from_dict(checkpoint["calibrator"])
    graph_config = GraphBuildConfig(**checkpoint["graph_config"])
    return model, calibrator, graph_config, checkpoint


def save_score_records(records, path):
    """Persist frozen embeddings/scores without evaluation labels."""
    if not records:
        raise ValueError("score records are empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "representation": np.asarray("learned_attention_gnn_node_embedding"),
        "embedding": np.asarray([row["embedding"] for row in records], dtype=np.float32),
        "score": np.asarray([row["score"] for row in records], dtype=np.float32),
        "sample_id": np.asarray([row["sample_id"] for row in records], dtype=str),
        "source_id": np.asarray([row["source_id"] for row in records], dtype=str),
        "token_index": np.asarray([row["token_index"] for row in records], dtype=np.int32),
        "task_type": np.asarray([str(row.get("task_type")) for row in records], dtype=str),
        "data_source": np.asarray([str(row.get("data_source")) for row in records], dtype=str),
        "generator_model": np.asarray([str(row.get("generator_model")) for row in records], dtype=str),
    }
    for name in SCORE_COMPONENTS:
        arrays[f"residual_{name}"] = np.asarray(
            [row[f"residual_{name}"] for row in records], dtype=np.float32
        )
        arrays[f"z_{name}"] = np.asarray(
            [row[f"z_{name}"] for row in records], dtype=np.float32
        )
    np.savez_compressed(path, **arrays)
    return str(path)


def load_score_records(path):
    with np.load(Path(path), allow_pickle=False) as arrays:
        required = {"embedding", "score", "sample_id", "source_id", "token_index"}
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"score artifact is missing {sorted(missing)}")
        count = len(arrays["score"])
        representation = (
            str(arrays["representation"].item())
            if "representation" in arrays.files
            else "unspecified_embedding"
        )
        records = []
        for row in range(count):
            record = {
                "embedding": arrays["embedding"][row].astype(np.float32, copy=False),
                "score": float(arrays["score"][row]),
                "sample_id": str(arrays["sample_id"][row]),
                "source_id": str(arrays["source_id"][row]),
                "token_index": int(arrays["token_index"][row]),
                "representation": representation,
            }
            for field in ("task_type", "data_source", "generator_model"):
                if field in arrays.files:
                    record[field] = str(arrays[field][row])
            for name in SCORE_COMPONENTS:
                for prefix in ("residual", "z"):
                    field = f"{prefix}_{name}"
                    if field in arrays.files:
                        record[field] = float(arrays[field][row])
            records.append(record)
    return records
