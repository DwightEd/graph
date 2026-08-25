"""Dataset-level reference fitting, scoring and graph-embedding export."""

from dataclasses import dataclass
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
from .config import DetectionConfig, GraphConfig, MethodConfig, PCutConfig
from .detection import (
    CONDITION_NAMES,
    RESIDUAL_NAMES,
    ConditionalReference,
    Reservoir,
    token_conditions,
)
from .graph import AttentionGraph, build_graph
from .pcut import PCutResult, compute_pcut

MODEL_TYPE = "pcut"


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
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("no samples match the requested task")
    return tuple(selected)


def config_from_dict(payload: dict[str, object]) -> MethodConfig:
    return MethodConfig(
        graph=GraphConfig(**payload["graph"]),
        pcut=PCutConfig(**payload["pcut"]),
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
        raise ValueError("unsupported P-Cut checkpoint")
    config = config_from_dict(checkpoint["config"])
    arrays = load_npz(reference_path)
    if str(arrays["schema"].item()) != REFERENCE_SCHEMA:
        raise ValueError("unsupported P-Cut reference")
    if sha256(checkpoint_path) != str(arrays["checkpoint_sha256"].item()):
        raise ValueError("reference was fitted for a different checkpoint")
    return config, ConditionalReference.from_arrays(arrays), checkpoint


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
    save_checkpoint(
        checkpoint_path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "model_type": MODEL_TYPE,
            "config": config.as_dict(),
            "labels_read": False,
        },
    )

    reservoir = Reservoir(config.detection.reservoir_rows, config.pcut.seed)
    for sample_id in tqdm(sample_ids, desc="fit P-Cut reference", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, config.graph)
            result = compute_pcut(graph, config.pcut)
            reservoir.add(
                residual=result.closure.cpu().numpy()[:, None],
                condition=token_conditions(graph, result),
                task=np.repeat(str(sample.task_type or ""), graph.response_count),
            )
        finally:
            sample.release_attention()

    rows = reservoir.values()
    reference = ConditionalReference.fit(
        rows["residual"],
        rows["condition"],
        rows["task"],
        config.detection,
    )
    save_reference(reference_path, reference, checkpoint_path, dataset)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "reference": str(Path(reference_path).resolve()),
        "samples": len(sample_ids),
        "tokens": int(len(rows["residual"])),
        "labels_read": False,
    }


def safe_name(sample_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    return name or "sample"


def export_graph(path, graph: AttentionGraph, result: PCutResult) -> None:
    parts = result.edge_parts
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
        edge_prompt_rooted=parts.prompt_rooted.cpu().numpy().astype(np.float32),
        edge_response_closed=parts.response_closed.cpu().numpy().astype(np.float32),
        edge_uncertain=parts.uncertain.cpu().numpy().astype(np.float32),
        diagonal=graph.diagonal.cpu().numpy().astype(np.float16),
        unresolved=graph.unresolved.cpu().numpy().astype(np.float16),
        prompt_origin_lower=result.prompt_origin_lower.cpu().numpy().astype(np.float16),
        prompt_origin_upper=result.prompt_origin_upper.cpu().numpy().astype(np.float16),
        token_layer_embedding=result.token_layer_embedding.cpu().numpy().astype(np.float16),
        token_embedding=result.token_embedding.cpu().numpy().astype(np.float16),
        no_prompt_embedding=result.no_prompt_embedding.cpu().numpy().astype(np.float16),
        no_closed_embedding=result.no_closed_embedding.cpu().numpy().astype(np.float16),
        prompt_necessity=result.prompt_necessity.cpu().numpy().astype(np.float32),
        response_closed_necessity=result.response_closed_necessity.cpu().numpy().astype(np.float32),
        closure=result.closure.cpu().numpy().astype(np.float32),
    )


def make_score_rows(
    graph: AttentionGraph,
    result: PCutResult,
    reference: ConditionalReference,
) -> ScoreRows:
    condition = token_conditions(graph, result)
    task = np.repeat(graph.task_type, graph.response_count)
    residual = result.closure.cpu().numpy().astype(np.float32)[:, None]
    score, standardized = reference.transform(residual, condition, task)
    tokens = graph.response_count
    return ScoreRows(
        sample_id=np.repeat(graph.sample_id, tokens),
        source_id=np.repeat(graph.source_id, tokens),
        task_type=task,
        token_index=np.arange(tokens, dtype=np.int32),
        response_length=np.full(tokens, tokens, dtype=np.int32),
        response_token_id=graph.response_token_ids.cpu().numpy().astype(np.int64),
        score=score,
        residual=residual,
        standardized=standardized,
        coverage=result.coverage.cpu().numpy().astype(np.float32)[:, None],
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

    for sample_id in tqdm(sample_ids, desc="score P-Cut", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, config.graph)
            result = compute_pcut(graph, config.pcut)
            rows.append(make_score_rows(graph, result, reference))
            export_graph(graph_dir / f"{safe_name(sample_id)}.npz", graph, result)
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
