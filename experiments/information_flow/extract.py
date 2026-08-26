"""Extract label-free information-flow node embeddings from an attention split."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from experiment_protocol import partition_source_groups
from research_dataset import open_research_dataset
from experiments.grounded_route.config import GraphConfig
from experiments.grounded_route.graph import build_graph

from .config import FlowConfig, VIEW_NAMES
from .transport import encode_views


def select_samples(dataset, task: str, limit: int | None) -> tuple[str, ...]:
    selected = []
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        if task.casefold() == "all" or str(sample.task_type).casefold() == task.casefold():
            selected.append(str(sample_id))
    if limit is not None:
        selected = selected[: int(limit)]
    return tuple(selected)


def select_scope(dataset, sample_ids, scope: str, config: FlowConfig):
    if scope == "all":
        return sample_ids
    split = partition_source_groups(
        dataset,
        sample_ids,
        calibration_fraction=config.calibration_fraction,
        seed=config.seed,
    )
    return split["calibration_sample_ids"]


def append_rows(storage, graph, embeddings):
    count = graph.response_count
    common = {
        "sample_id": np.repeat(graph.sample_id, count),
        "source_id": np.repeat(graph.source_id, count),
        "token_index": np.arange(count, dtype=np.int32),
        "response_length": np.full(count, count, dtype=np.int32),
        "response_token_id": graph.response_token_ids.cpu().numpy().astype(np.int64),
    }
    for name, embedding in embeddings.items():
        for field, value in common.items():
            storage[name][field].append(value)
        storage[name]["embedding"].append(
            embedding.detach().cpu().numpy().astype(np.float32)
        )


def save_indices(output_dir: Path, storage) -> dict[str, str]:
    paths = {}
    for name in VIEW_NAMES:
        arrays = {
            field: np.concatenate(blocks)
            for field, blocks in storage[name].items()
        }
        path = output_dir / f"index_{name}.npz"
        np.savez_compressed(path, **arrays)
        paths[name] = str(path.resolve())
    return paths


def save_graph(path: Path, graph, views) -> None:
    arrays = {
        "sample_id": np.asarray(graph.sample_id),
        "source_id": np.asarray(graph.source_id),
        "task_type": np.asarray(graph.task_type),
        "response_start": np.asarray(graph.response_start, dtype=np.int32),
        "token_ids": graph.token_ids.cpu().numpy().astype(np.int32),
        "edge_source": graph.edges.source.cpu().numpy().astype(np.int32),
        "edge_target": graph.edges.target.cpu().numpy().astype(np.int32),
        "edge_layer": graph.edges.layer.cpu().numpy().astype(np.int16),
        "edge_head": graph.edges.head.cpu().numpy().astype(np.int16),
        "edge_weight": graph.edges.weight.cpu().numpy().astype(np.float16),
        "diagonal": graph.diagonal.detach().cpu().numpy().astype(np.float16),
        "unresolved": graph.unresolved.detach().cpu().numpy().astype(np.float16),
        "trajectory": views.trajectory.detach().cpu().numpy().astype(np.float16),
    }
    for name, embedding in views.embeddings().items():
        arrays[f"embedding_{name}"] = (
            embedding.detach().cpu().numpy().astype(np.float16)
        )
    np.savez_compressed(path, **arrays)


def extract(
    data_root,
    output_dir,
    *,
    task: str = "QA",
    scope: str = "all",
    limit: int | None = None,
    device: str = "cpu",
    config: FlowConfig | None = None,
) -> dict[str, object]:
    config = FlowConfig() if config is None else config
    dataset = open_research_dataset(data_root, device="cpu")
    sample_ids = select_scope(
        dataset,
        select_samples(dataset, task, limit),
        scope,
        config,
    )

    output_dir = Path(output_dir)
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    storage = {
        name: {
            "sample_id": [],
            "source_id": [],
            "token_index": [],
            "response_length": [],
            "response_token_id": [],
            "embedding": [],
        }
        for name in VIEW_NAMES
    }

    tokens = 0
    edges = 0
    for number, sample_id in enumerate(
        tqdm(sample_ids, desc=f"information flow {scope}", unit="sample")
    ):
        sample = dataset[sample_id]
        try:
            graph = build_graph(sample, GraphConfig())
        finally:
            sample.release_attention()

        graph = graph.to(device)
        views = encode_views(graph, config)
        append_rows(storage, graph, views.embeddings())
        save_graph(graph_dir / f"{number:08d}.npz", graph, views)
        tokens += graph.response_count
        edges += graph.edge_count

    indices = save_indices(output_dir, storage)
    report = {
        "method": "attention_only_information_flow",
        "scope": scope,
        "samples": len(sample_ids),
        "tokens": tokens,
        "edges": edges,
        "labels_read": False,
        "config": asdict(config),
        "indices": indices,
    }
    (output_dir / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
