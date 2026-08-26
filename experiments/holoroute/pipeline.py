"""Fit a normal node-feature subspace, score tokens and export graph data."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
import torch
from tqdm.auto import tqdm

from .artifacts import (
    CHECKPOINT_SCHEMA,
    GRAPH_SCHEMA,
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    load_npz,
    save_checkpoint,
    save_npz,
    sha256,
)
from .config import DetectionConfig, FeatureConfig, GraphConfig, MethodConfig
from .detection import (
    CONDITION_NAMES,
    RESIDUAL_NAMES,
    Reservoir,
    SubspaceReference,
    token_conditions,
)
from .features import RoutingFeatures, build_node_features
from .graph import AttentionGraph, build_graph

MODEL_TYPE = "routing_fingerprint"


@dataclass(frozen=True)
class ScoreRows:
    sample_id: np.ndarray
    source_id: np.ndarray
    task_type: np.ndarray
    token_index: np.ndarray
    response_length: np.ndarray
    response_token_id: np.ndarray
    score: np.ndarray
    residual: np.ndarray
    standardized: np.ndarray
    coverage: np.ndarray
    condition: np.ndarray


def require_split(dataset, expected: str) -> None:
    actual = str(dataset.manifest.get("split"))
    if actual != expected:
        raise ValueError(f"expected {expected!r} data, found {actual!r}")


def select_samples(dataset, task: str, limit: int | None) -> tuple[str, ...]:
    selected: list[str] = []
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        try:
            if task.casefold() == "all" or str(sample.task_type).casefold() == task.casefold():
                selected.append(str(sample_id))
        finally:
            sample.release_attention()
    return tuple(selected[:limit] if limit is not None else selected)


def calibration_groups(
    dataset,
    sample_ids: tuple[str, ...],
    fraction: float,
    seed: int,
) -> set[str]:
    groups: set[str] = set()
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            groups.add(str(sample.source_id or sample_id))
        finally:
            sample.release_attention()
    ordered = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest(),
    )
    if len(ordered) < 2:
        raise ValueError("reference fitting needs at least two source groups")
    count = min(max(int(round(len(ordered) * fraction)), 1), len(ordered) - 1)
    return set(ordered[:count])


def config_from_dict(payload: dict[str, object]) -> MethodConfig:
    return MethodConfig(
        graph=GraphConfig(**payload["graph"]),
        feature=FeatureConfig(**payload["feature"]),
        detection=DetectionConfig(**payload["detection"]),
    )


def save_reference(path, reference, checkpoint_path, dataset) -> None:
    save_npz(
        path,
        schema=np.asarray(REFERENCE_SCHEMA),
        model_type=np.asarray(MODEL_TYPE),
        labels_included=np.asarray(False),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        train_manifest_sha256=np.asarray(sha256(Path(dataset.root) / "manifest.json")),
        residual_names=np.asarray(RESIDUAL_NAMES),
        condition_names=np.asarray(CONDITION_NAMES),
        **reference.arrays(),
    )


def load_method(checkpoint_path, reference_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["schema"] != CHECKPOINT_SCHEMA or checkpoint["model_type"] != MODEL_TYPE:
        raise ValueError("unsupported routing-fingerprint checkpoint")
    config = config_from_dict(checkpoint["config"])
    arrays = load_npz(reference_path)
    if str(arrays["schema"].item()) != REFERENCE_SCHEMA:
        raise ValueError("unsupported routing-fingerprint reference")
    if sha256(checkpoint_path) != str(arrays["checkpoint_sha256"].item()):
        raise ValueError("reference was fitted for a different checkpoint")
    return config, SubspaceReference.from_arrays(arrays), checkpoint


def add_reference_rows(
    reservoir: Reservoir,
    graph: AttentionGraph,
    features: RoutingFeatures,
) -> None:
    reservoir.add(
        feature=features.node.cpu().numpy().astype(np.float32),
        condition=token_conditions(graph),
        task=np.repeat(graph.task_type, graph.response_count),
    )


def fit_reference(
    dataset,
    checkpoint_path,
    reference_path,
    config: MethodConfig,
    task: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    require_split(dataset, "train")
    sample_ids = select_samples(dataset, task, limit)
    fit_rows = Reservoir(config.detection.reservoir_rows, config.detection.seed)
    calibration_rows = Reservoir(
        config.detection.reservoir_rows,
        config.detection.seed + 1,
    )
    feature_dim = None
    held_out_groups = calibration_groups(
        dataset,
        sample_ids,
        config.detection.calibration_fraction,
        config.detection.seed,
    )

    for sample_id in tqdm(sample_ids, desc="build train node features", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, config.graph)
            features = build_node_features(graph, config.feature)
            feature_dim = features.node.shape[1]
            group = graph.source_id or graph.sample_id
            destination = calibration_rows if group in held_out_groups else fit_rows
            add_reference_rows(destination, graph, features)
        finally:
            sample.release_attention()

    fit = fit_rows.values()
    calibration = calibration_rows.values()
    reference = SubspaceReference.fit(
        fit["feature"],
        fit["condition"],
        fit["task"],
        config.detection,
    ).calibrate(
        calibration["feature"],
        calibration["condition"],
        calibration["task"],
    )
    save_checkpoint(
        checkpoint_path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "model_type": MODEL_TYPE,
            "config": config.as_dict(),
            "feature_dim": int(feature_dim),
            "labels_read": False,
        },
    )
    save_reference(reference_path, reference, checkpoint_path, dataset)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "reference": str(Path(reference_path).resolve()),
        "samples": len(sample_ids),
        "fit_tokens": int(len(fit["feature"])),
        "calibration_tokens": int(len(calibration["feature"])),
        "feature_dim": int(feature_dim),
        "labels_read": False,
    }


def safe_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id) or "sample"


def export_graph(path, graph: AttentionGraph, features: RoutingFeatures) -> None:
    save_npz(
        path,
        schema=np.asarray(GRAPH_SCHEMA),
        sample_id=np.asarray(graph.sample_id),
        source_id=np.asarray(graph.source_id),
        task_type=np.asarray(graph.task_type),
        response_start=np.asarray(graph.response_start, dtype=np.int32),
        token_count=np.asarray(graph.token_count, dtype=np.int32),
        response_token_id=graph.response_token_ids.cpu().numpy().astype(np.int64),
        edge_source=graph.edges.source.cpu().numpy().astype(np.int32),
        edge_target=graph.edges.target.cpu().numpy().astype(np.int32),
        edge_layer=graph.edges.layer.cpu().numpy().astype(np.int16),
        edge_head=graph.edges.head.cpu().numpy().astype(np.int16),
        edge_weight=graph.edges.weight.cpu().numpy().astype(np.float32),
        diagonal=graph.diagonal.cpu().numpy().astype(np.float16),
        unresolved=graph.unresolved.cpu().numpy().astype(np.float16),
        token_layer_feature=features.token_layer.cpu().numpy().astype(np.float16),
        node_feature=features.node.cpu().numpy().astype(np.float16),
    )


def make_score_rows(
    graph: AttentionGraph,
    features: RoutingFeatures,
    reference: SubspaceReference,
) -> ScoreRows:
    condition = token_conditions(graph)
    task = np.repeat(graph.task_type, graph.response_count)
    score, energy = reference.transform(
        features.node.cpu().numpy().astype(np.float32),
        condition,
        task,
    )
    tokens = graph.response_count
    return ScoreRows(
        sample_id=np.repeat(graph.sample_id, tokens),
        source_id=np.repeat(graph.source_id, tokens),
        task_type=task,
        token_index=np.arange(tokens, dtype=np.int32),
        response_length=np.full(tokens, tokens, dtype=np.int32),
        response_token_id=graph.response_token_ids.cpu().numpy().astype(np.int64),
        score=score,
        residual=energy,
        standardized=energy,
        coverage=np.ones((tokens, 1), dtype=np.float32),
        condition=condition,
    )


def merge_rows(rows: list[ScoreRows]) -> dict[str, np.ndarray]:
    if not rows:
        raise RuntimeError("scoring produced no token rows")
    return {
        field: np.concatenate([getattr(row, field) for row in rows])
        for field in ScoreRows.__dataclass_fields__
    }


def score_dataset(
    dataset,
    checkpoint_path,
    reference_path,
    output_path,
    task: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    require_split(dataset, "test")
    config, reference, checkpoint = load_method(checkpoint_path, reference_path)
    rows: list[ScoreRows] = []
    graph_dir = Path(output_path).parent / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = select_samples(dataset, task, limit)

    for sample_id in tqdm(sample_ids, desc="score node features", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, config.graph)
            features = build_node_features(graph, config.feature)
            rows.append(make_score_rows(graph, features, reference))
            export_graph(graph_dir / f"{safe_name(sample_id)}.npz", graph, features)
        finally:
            sample.release_attention()

    arrays = merge_rows(rows)
    save_npz(
        output_path,
        schema=np.asarray(SCORE_SCHEMA),
        model_type=np.asarray(MODEL_TYPE),
        labels_included=np.asarray(False),
        checkpoint_path=np.asarray(str(Path(checkpoint_path).resolve())),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        reference_path=np.asarray(str(Path(reference_path).resolve())),
        reference_sha256=np.asarray(sha256(reference_path)),
        dataset_manifest_sha256=np.asarray(sha256(Path(dataset.root) / "manifest.json")),
        residual_names=np.asarray(RESIDUAL_NAMES),
        condition_names=np.asarray(CONDITION_NAMES),
        **arrays,
    )
    return {
        "scores": str(Path(output_path).resolve()),
        "graphs": str(graph_dir.resolve()),
        "samples": len(sample_ids),
        "tokens": len(arrays["score"]),
        "labels_read": False,
        "config": checkpoint["config"],
    }
