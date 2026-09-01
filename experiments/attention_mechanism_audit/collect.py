"""Collect resumable frozen-model mechanism states for every RAGTruth task.

This module only owns dataset traversal, provenance, serialization, and resume.
Mechanism extraction lives in :mod:`capture`; detector fitting lives in the
evaluation path.  In particular, no hallucination label is opened here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from research_dataset import open_research_dataset

from .capture import FunctionalTraceReplay
from .data import (
    TASK_TYPES,
    build_evidence_mask,
    canonical_task_type,
    load_source_info,
)

SCHEMA = "ragtruth-mechanism-state"
VERSION = 6


def _file_identity(path: str | Path) -> str:
    """Return a relocation-stable identity for an input file or fixture."""

    path = Path(path)
    if not path.is_file():
        return path.name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_identity(path: str | Path) -> str:
    path = Path(path)
    if not path.is_dir():
        return path.name
    digest = hashlib.sha256()
    digest.update(path.name.encode())
    for candidate in sorted(path.iterdir(), key=lambda item: item.name):
        if candidate.name == "config.json" or candidate.name.endswith(
            (".safetensors", ".bin", ".index.json")
        ):
            digest.update(candidate.name.encode())
            digest.update(b"\0")
            digest.update(str(candidate.stat().st_size).encode())
            digest.update(b"\0")
            if candidate.name.endswith(".json"):
                digest.update(_file_identity(candidate).encode())
    return digest.hexdigest()


def _split_identity(dataset: Any) -> str:
    """Identify the physical shard by split name and ordered sample IDs."""

    digest = hashlib.sha256()
    digest.update(str(dataset.manifest.get("split", "")).encode())
    for sample_id in dataset.sample_ids:
        digest.update(b"\0")
        digest.update(str(sample_id).encode())
    return digest.hexdigest()


def _validate_alignment(capture: dict[str, Any]) -> None:
    """Reject a trace whose token axis no longer matches teacher forcing."""

    tokens = len(capture["token_ids"])
    response_start = int(capture["response_start"])
    if not 0 < response_start < tokens:
        raise ValueError("response_start is outside token_ids")
    response_tokens = tokens - response_start
    for name, value in capture["score_inputs"].items():
        if len(value) != response_tokens:
            raise ValueError(f"score input {name} is not response-token aligned")
    for name, value in capture["trace"].items():
        if value.ndim >= 2 and value.shape[1] != response_tokens:
            raise ValueError(f"mechanism trace {name} is not response-token aligned")


def _save(path: Path, capture: dict[str, Any]) -> dict[str, Any]:
    _validate_alignment(capture)
    peak_memory = capture.pop("peak_cuda_reserved_bytes")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(capture, temporary)
    temporary.replace(path)
    return {
        "path": path.name,
        "tokens": len(capture["token_ids"]),
        "response_tokens": len(capture["token_ids"]) - capture["response_start"],
        "bytes": path.stat().st_size,
        "peak_cuda_reserved_bytes": int(peak_memory),
    }


def capture_split(
    *,
    split_root: str | Path,
    source_info: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    limit: int | None = None,
    predictor_chunk: int = 128,
    top_k: int = 8,
    logit_chunk: int = 64,
    intervention_batch: int = 3,
) -> dict[str, Any]:
    """Collect one physical shard, journaling progress for exact resume."""

    from transformers import AutoTokenizer

    dataset = open_research_dataset(
        split_root, device="cpu", retain_embedded_labels=False
    )
    output = Path(output_root)
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    manifest_path = output / "manifest.json"
    if index_path.is_file() and not manifest_path.is_file():
        raise ValueError(f"index exists without a v{VERSION} manifest; use a new output")
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    expected = {
        "schema": SCHEMA,
        "version": VERSION,
        "split_identity": _split_identity(dataset),
        "source_identity": _file_identity(source_info),
        "observer_identity": _model_identity(model_path),
        "model_dtype": str(dtype),
        "top_k_message_sources": int(top_k),
        "task_types": list(TASK_TYPES),
    }
    if previous_manifest:
        changed = [
            name
            for name, value in expected.items()
            if previous_manifest.get(name) != value
        ]
        if changed:
            raise ValueError(
                "cannot resume mechanism states with changed identity: "
                + ", ".join(changed)
            )
        if previous_manifest.get("complete") and limit is None:
            return previous_manifest
    else:
        manifest_path.write_text(
            json.dumps(
                {**expected, "samples": 0, "complete": False, "labels_used": False},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    sources = load_source_info(source_info)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    replay = FunctionalTraceReplay.from_pretrained(
        model_path, device=device, dtype=dtype
    )
    rows = load_index(output) if index_path.is_file() else []
    indexed = {str(row["sample_id"]): row["task_type"] for row in rows}
    resumed = len(rows)
    new_samples = 0
    evidence_cache: dict[str, torch.Tensor] = {}
    eligible = 0
    selected = {task: 0 for task in TASK_TYPES}

    def record(saved: dict[str, Any]) -> None:
        nonlocal new_samples
        rows.append(saved)
        with index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(saved, sort_keys=True) + "\n")
        new_samples += 1

    for sample_id in dataset.sample_ids:
        if limit is not None and all(count >= limit for count in selected.values()):
            break
        saved_task = indexed.get(str(sample_id))
        if saved_task is not None:
            task_type = canonical_task_type(saved_task)
            if limit is not None and selected[task_type] >= limit:
                continue
            selected[task_type] += 1
            eligible += 1
            continue

        sample = dataset[sample_id]
        task_type = canonical_task_type(sample.task_type)
        if limit is not None and selected[task_type] >= limit:
            continue
        selected[task_type] += 1
        eligible += 1
        attention = sample.attention()
        source_id = str(sample.source_id)
        generator_model = sample.generator_model
        print(f"collect {new_samples + 1} ({task_type}): {sample_id}", flush=True)
        evidence_mask = evidence_cache.get(source_id)
        if evidence_mask is None:
            evidence_mask = torch.from_numpy(
                build_evidence_mask(
                    sources[source_id],
                    tokenizer,
                    attention.token_ids,
                    int(attention.response_idx),
                )
            )
            evidence_cache[source_id] = evidence_mask
        capture = replay.capture(
            attention.token_ids,
            int(attention.response_idx),
            evidence_mask,
            predictor_chunk=predictor_chunk,
            top_k=top_k,
            logit_chunk=logit_chunk,
            intervention_batch=intervention_batch,
        )
        sample.release_attention()
        saved = _save(samples / f"{sample_id}.pt", capture)
        record(
            {
                "sample_id": str(sample_id),
                "source_id": source_id,
                "task_type": task_type,
                "generator_model": generator_model,
                **saved,
            }
        )

    complete = limit is None or bool(previous_manifest.get("complete", False))
    eligible_total = (
        eligible if limit is None else previous_manifest.get("eligible_samples")
    )
    manifest = {
        **expected,
        "samples": len(rows),
        "dataset_candidates": len(dataset.sample_ids),
        "eligible_samples": eligible_total,
        "selected_samples_seen": eligible,
        "resumed_samples": resumed,
        "new_samples": new_samples,
        "complete": complete,
        "split": dataset.manifest.get("split"),
        "task_types": list(TASK_TYPES),
        "generator_models": sorted({str(row["generator_model"]) for row in rows}),
        "labels_used": False,
        "model_dtype": str(dtype),
        "predictor_chunk": int(predictor_chunk),
        "intervention_batch": int(intervention_batch),
        "top_k_message_sources": int(top_k),
        "max_cuda_reserved_bytes": max(
            (row["peak_cuda_reserved_bytes"] for row in rows), default=0
        ),
        "index": index_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def capture_all(
    *,
    split_roots: Sequence[str | Path],
    source_info: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    limit: int | None = None,
) -> dict[str, list[tuple[Path, Path]]]:
    """Collect both physical shards once and return task-filtered inputs."""

    pairs = []
    for split_root in map(Path, split_roots):
        state_root = Path(output_root) / "routing_state" / split_root.name
        capture_split(
            split_root=split_root,
            source_info=source_info,
            model_path=model_path,
            output_root=state_root,
            device=device,
            dtype=dtype,
            limit=limit,
        )
        pairs.append((state_root, split_root))
    return {task: list(pairs) for task in TASK_TYPES}


def load_index(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / "index.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
