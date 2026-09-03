"""Dataset traversal and serialization for functional message graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from research_dataset import open_research_dataset

from .capture import FunctionalMessageReplay
from .data import evidence_mask, load_sources, task_name

STATE_DIRECTORY = "functional_message_graph"


def _read_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def capture_split(
    split_root: str | Path,
    source_info: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    *,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    predictor_batch: int = 1,
    prefix_chunk: int = 128,
    edge_cover: float = 0.95,
    edge_budget: int = 64,
    limit: int | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    dataset = open_research_dataset(
        split_root, device="cpu", retain_embedded_labels=False
    )
    output = Path(output_root)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    manifest_path = output / "manifest.json"
    rows = _read_index(index_path)
    identity = {
        "schema": "functional-message-graph-v2",
        "split_root": str(Path(split_root).resolve()),
        "source_info": str(Path(source_info).resolve()),
        "model": str(Path(model_path).resolve()),
        "dtype": str(dtype),
        "edge_cover": float(edge_cover),
        "edge_budget": int(edge_budget),
        "labels_used": False,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("output directory belongs to a different graph capture")
        if previous.get("complete") and limit is None:
            return previous
    else:
        manifest_path.write_text(
            json.dumps({**identity, "samples": len(rows), "complete": False}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    completed = {row["sample_id"] for row in rows}
    selected = {task: 0 for task in ("QA", "Summary", "Data2txt")}
    for row in rows:
        selected[task_name(row["task_type"])] += 1

    sources = load_sources(source_info)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    replay = FunctionalMessageReplay.from_pretrained(
        model_path, device=device, dtype=dtype
    )
    masks: dict[str, torch.Tensor] = {}
    for sample_id in dataset.sample_ids:
        if limit is not None and all(count >= limit for count in selected.values()):
            break
        sample_id = str(sample_id)
        if sample_id in completed:
            continue
        sample = dataset[sample_id]
        current_task = task_name(sample.task_type)
        if limit is not None and selected[current_task] >= limit:
            continue
        selected[current_task] += 1
        attention = sample.attention()
        source_id = str(sample.source_id)
        mask = masks.get(source_id)
        if mask is None:
            mask = torch.from_numpy(
                evidence_mask(
                    sources[source_id],
                    tokenizer,
                    attention.token_ids,
                    int(attention.response_idx),
                )
            )
            masks[source_id] = mask

        print(f"build {sample.task_type}: {sample_id}", flush=True)
        graph = replay.capture(
            attention.token_ids,
            int(attention.response_idx),
            mask,
            predictor_batch=predictor_batch,
            prefix_chunk=prefix_chunk,
            edge_cover=edge_cover,
            edge_budget=edge_budget,
        )
        sample.release_attention()
        graph.update(
            sample_id=sample_id,
            source_id=source_id,
            task_type=current_task,
            generator_model=sample.generator_model,
            predictor_batch=int(predictor_batch),
            prefix_chunk=int(prefix_chunk),
            edge_cover=float(edge_cover),
            edge_budget=int(edge_budget),
            labels_used=False,
        )
        path = sample_dir / f"{sample_id}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(graph, temporary)
        temporary.replace(path)
        row = {
            "sample_id": sample_id,
            "source_id": source_id,
            "task_type": graph["task_type"],
            "generator_model": sample.generator_model,
            "path": str(path.relative_to(output)),
            "response_tokens": int(graph["node_embedding"].shape[0]),
            "embedding_dim": int(graph["node_embedding"].shape[1]),
            "explicit_attention_edges": int(graph["edge_index"].shape[1]),
        }
        with index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        rows.append(row)

    complete = limit is None and len(rows) == len(dataset.sample_ids)
    manifest = {
        **identity,
        "predictor_batch": int(predictor_batch),
        "prefix_chunk": int(prefix_chunk),
        "samples": len(rows),
        "complete": complete,
        "labels_used": False,
        "index": index_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def capture_all(
    cache_root: str | Path,
    source_info: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    reports = []
    for split in ("train", "test"):
        reports.append(
            capture_split(
                Path(cache_root) / split,
                source_info,
                model_path,
                Path(output_root) / STATE_DIRECTORY / split,
                **kwargs,
            )
        )
    return reports