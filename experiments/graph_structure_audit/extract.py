"""Label-free extraction of per-sample graphs, motifs, and recovery profiles."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from experiments.source_reuse_contrast.data import (
    collect_source_reuse_graph,
    select_sample_ids,
)
from .artifacts import AUDIT_SCHEMA, save_npz, write_json
from .config import GraphAuditConfig
from .structures import RECOVERY_METRICS, STRUCTURAL_METRICS, audit_graph


def _open_dataset(split_root):
    from research_dataset import open_research_dataset

    return open_research_dataset(split_root, device="cpu")


def _safe_name(sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]
    return f"sample_{digest}.npz"


def extract_graph_audit(
    *,
    split_root,
    output_dir,
    config: GraphAuditConfig | None = None,
    task_type: str | None = None,
    limit: int | None = None,
    save_raw_graphs: bool = True,
) -> Path:
    config = GraphAuditConfig() if config is None else config
    config.validate()
    dataset = _open_dataset(split_root)
    sample_ids = select_sample_ids(dataset, task_type=task_type, limit=limit)
    output_dir = Path(output_dir)
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list] = {
        "sample_id": [],
        "source_id": [],
        "task_type": [],
        "token_index": [],
        "response_length": [],
        "structural": [],
        "recovery": [],
        "valid_recovery": [],
    }
    graph_index: list[dict[str, object]] = []

    iterator = tqdm(
        sample_ids,
        desc="graph structure audit",
        unit="sample",
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    for sample_id in iterator:
        sample = dataset[sample_id]
        graph = collect_source_reuse_graph(sample, block_rows=config.block_rows)
        result = audit_graph(graph, config)
        response_count = graph.num_response_tokens

        rows["sample_id"].extend([graph.sample_id] * response_count)
        rows["source_id"].extend([graph.source_id] * response_count)
        rows["task_type"].extend([graph.task_type] * response_count)
        rows["token_index"].extend(range(response_count))
        rows["response_length"].extend([response_count] * response_count)
        rows["structural"].append(result.structural)
        rows["recovery"].append(result.recovery)
        rows["valid_recovery"].append(result.valid_recovery)

        file_name = _safe_name(graph.sample_id)
        graph_path = graph_dir / file_name
        sample_artifact = {
            "schema": np.asarray(AUDIT_SCHEMA),
            "labels_included": np.asarray(False),
            "sample_id": np.asarray(graph.sample_id),
            "source_id": np.asarray(graph.source_id),
            "task_type": np.asarray(graph.task_type),
            "response_idx": np.asarray(graph.response_idx, dtype=np.int32),
            "num_response_tokens": np.asarray(response_count, dtype=np.int32),
            "num_layers": np.asarray(graph.num_layers, dtype=np.int16),
            "num_heads": np.asarray(graph.num_heads, dtype=np.int16),
            "attention_floor": np.asarray(graph.attention_floor, dtype=np.float32),
            "structural_names": np.asarray(STRUCTURAL_METRICS, dtype=str),
            "recovery_names": np.asarray(RECOVERY_METRICS, dtype=str),
            "structural": result.structural,
            "recovery": result.recovery,
            "valid_recovery": result.valid_recovery,
        }
        if save_raw_graphs:
            sample_artifact.update(
                {
                    "layer": graph.layer.cpu().numpy().astype(np.int16),
                    "head": graph.head.cpu().numpy().astype(np.int16),
                    "query": graph.query.cpu().numpy().astype(np.int32),
                    "source": graph.source.cpu().numpy().astype(np.int32),
                    "weight": graph.weight.cpu().numpy().astype(np.float32),
                    "query_ptr": graph.query_ptr.cpu().numpy().astype(np.int64),
                    "diagonal": graph.diagonal.cpu().numpy().astype(np.float16),
                }
            )
        save_npz(graph_path, **sample_artifact)
        graph_index.append(
            {
                "sample_id": graph.sample_id,
                "source_id": graph.source_id,
                "task_type": graph.task_type,
                "tokens": response_count,
                "edges": graph.num_edges,
                "file": str(Path("graphs") / file_name),
            }
        )
        sample.release_attention()
        iterator.set_postfix(tokens=len(rows["sample_id"]), graphs=len(graph_index))

    structural = np.concatenate(rows["structural"], axis=0)
    recovery = np.concatenate(rows["recovery"], axis=0)
    valid_recovery = np.concatenate(rows["valid_recovery"], axis=0)
    token_path = output_dir / "tokens.npz"
    save_npz(
        token_path,
        schema=np.asarray(AUDIT_SCHEMA),
        labels_included=np.asarray(False),
        sample_id=np.asarray(rows["sample_id"], dtype=str),
        source_id=np.asarray(rows["source_id"], dtype=str),
        task_type=np.asarray(rows["task_type"], dtype=str),
        token_index=np.asarray(rows["token_index"], dtype=np.int32),
        response_length=np.asarray(rows["response_length"], dtype=np.int32),
        structural_names=np.asarray(STRUCTURAL_METRICS, dtype=str),
        recovery_names=np.asarray(RECOVERY_METRICS, dtype=str),
        structural=structural,
        recovery=recovery,
        valid_recovery=valid_recovery,
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema": AUDIT_SCHEMA,
            "labels_read": False,
            "split_root": str(Path(split_root).resolve()),
            "task_type": task_type,
            "samples": len(graph_index),
            "tokens": int(len(rows["sample_id"])),
            "config": asdict(config),
            "structural_metrics": list(STRUCTURAL_METRICS),
            "recovery_metrics": list(RECOVERY_METRICS),
            "raw_graphs_saved": save_raw_graphs,
            "token_file": "tokens.npz",
            "graphs": graph_index,
        },
    )
    return token_path
