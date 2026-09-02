"""Collect resumable frozen-model mechanism states for every RAGTruth task.

This module only owns dataset traversal, provenance, serialization, and resume.
Mechanism extraction lives in :mod:`capture`; detector fitting lives in the
evaluation path.  In particular, no hallucination label is opened here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from research_dataset import open_research_dataset

from .capture import (
    BRANCH_NAMES,
    REGISTER_NAMES,
    REGISTER_STAGE_NAMES,
    ROLE_NAMES,
    FunctionalTraceReplay,
)
from .data import (
    TASK_TYPES,
    build_evidence_mask,
    canonical_task_type,
    load_source_info,
)

SCHEMA = "ragtruth-mechanism-state"
VERSION = 8
STATE_DIRECTORY = "dual_register_state"
_MODEL_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".pt",
    ".pth",
    ".safetensors",
    ".tiktoken",
    ".txt",
}


def file_stamp(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stamp: dict[str, Any] = {"path": str(path.resolve())}
    if path.is_file():
        stat = path.stat()
        stamp.update(size=stat.st_size, modified_ns=stat.st_mtime_ns)
    return stamp


def model_stamp(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_dir():
        return {"path": str(path.resolve()), "files": []}
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.casefold() in _MODEL_SUFFIXES
    )
    return {
        "path": str(path.resolve()),
        "files": [
            [
                candidate.relative_to(path).as_posix(),
                candidate.stat().st_size,
                candidate.stat().st_mtime_ns,
            ]
            for candidate in files
        ],
    }


def _capture_spec(top_k: int, route_cover_mass: float) -> dict[str, Any]:
    """Describe only the scientific state saved by capture, not chunking knobs."""

    return {
        "branches": list(BRANCH_NAMES),
        "registers": list(REGISTER_NAMES),
        "register_stages": list(REGISTER_STAGE_NAMES),
        "source_roles": list(ROLE_NAMES),
        "route_cover_mass": float(route_cover_mass),
        "top_k": int(top_k),
    }


def target_token_ids(token_ids: Any, response_start: int) -> list[int]:
    return torch.as_tensor(token_ids, dtype=torch.long, device="cpu")[
        response_start:
    ].tolist()


def split_stamp(dataset: Any, path: str | Path) -> dict[str, Any]:
    manifest = json.loads(json.dumps(dataset.manifest, default=str))
    return {
        "path": str(Path(path).resolve()),
        "manifest": manifest,
        "samples": len(dataset.sample_ids),
    }


def _validate_alignment(capture: dict[str, Any]) -> None:
    """Reject a trace whose token axis no longer matches teacher forcing."""

    tokens = len(capture["token_ids"])
    response_start = int(capture["response_start"])
    if not 0 < response_start < tokens:
        raise ValueError("response_start is outside token_ids")
    evidence_mask = torch.as_tensor(capture["evidence_mask"])
    if evidence_mask.dtype != torch.bool or evidence_mask.shape != (response_start,):
        raise ValueError("evidence_mask is not prompt-token aligned")
    response_tokens = tokens - response_start
    for name, value in capture["score_inputs"].items():
        if len(value) != response_tokens:
            raise ValueError(f"score input {name} is not response-token aligned")
    for name, value in capture["trace"].items():
        if value.ndim >= 2 and value.shape[1] != response_tokens:
            raise ValueError(f"mechanism trace {name} is not response-token aligned")


def _save(
    path: Path,
    capture: dict[str, Any],
    *,
    artifact_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_alignment(capture)
    if artifact_contract is not None:
        capture["artifact_contract"] = artifact_contract
    peak_memory = capture.pop("peak_cuda_reserved_bytes")
    response_start = int(capture["response_start"])
    evidence_mask = torch.as_tensor(capture["evidence_mask"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(capture, temporary)
    temporary.replace(path)
    saved = {
        "path": path.name,
        "tokens": len(capture["token_ids"]),
        "prompt_tokens": response_start,
        "evidence_tokens": int(evidence_mask.sum()),
        "response_tokens": len(capture["token_ids"]) - response_start,
        "target_token_ids": target_token_ids(capture["token_ids"], response_start),
        "bytes": path.stat().st_size,
        "peak_cuda_reserved_bytes": int(peak_memory),
    }
    if artifact_contract is not None:
        saved["artifact_contract"] = artifact_contract
    return saved


def validate_saved_artifact(
    artifact: dict[str, Any] | Any, record: dict[str, Any] | Any
) -> None:
    """Check a loaded artifact against its journal row before using values."""

    expected_contract = record.get("artifact_contract")
    if (
        expected_contract is not None
        and artifact.get("artifact_contract") != expected_contract
    ):
        raise ValueError("artifact contract does not match its index row")
    expected_tokens = record.get("target_token_ids")
    actual_tokens = target_token_ids(
        artifact["token_ids"], int(artifact["response_start"])
    )
    if expected_tokens is not None and actual_tokens != expected_tokens:
        raise ValueError("artifact target tokens do not match its index row")


def _validate_resume_index(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Validate the journal boundary before any indexed sample is skipped."""

    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("resume index contains duplicate sample IDs")
    missing = []
    changed = []
    for row in rows:
        path = root / "samples" / str(row["path"])
        if not path.is_file():
            missing.append(str(row["sample_id"]))
        elif row.get("bytes") is not None and path.stat().st_size != int(row["bytes"]):
            changed.append(str(row["sample_id"]))
    if missing:
        raise ValueError("resume index references a missing sample artifact")
    if changed:
        raise ValueError("resume index references a size-changed sample artifact")


def capture_split(
    *,
    split_root: str | Path,
    source_info: str | Path,
    model_path: str | Path,
    output_root: str | Path,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    limit: int | None = None,
    predictor_chunk: int = 64,
    top_k: int = 32,
    logit_chunk: int = 64,
    route_cover_mass: float = 0.8,
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
        raise ValueError(
            f"index exists without a v{VERSION} manifest; use a new output"
        )
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    rows = load_index(output) if index_path.is_file() else []
    expected = {
        "schema": SCHEMA,
        "version": VERSION,
        "split_identity": split_stamp(dataset, split_root),
        "source_identity": file_stamp(source_info),
        "observer_identity": model_stamp(model_path),
        "model_dtype": str(dtype),
        "capture_spec": _capture_spec(top_k, route_cover_mass),
        "task_types": list(TASK_TYPES),
    }
    artifact_contract = {
        "schema": SCHEMA,
        "version": VERSION,
        "capture_spec": expected["capture_spec"],
    }
    if previous_manifest:
        changed = [
            name
            for name, value in expected.items()
            if previous_manifest.get(name) != value
        ]
        if changed:
            if rows or int(previous_manifest.get("samples", 0)):
                raise ValueError(
                    "cannot resume mechanism states with changed identity: "
                    + ", ".join(changed)
                )
            previous_manifest = {}
    if not previous_manifest:
        manifest_path.write_text(
            json.dumps(
                {**expected, "samples": 0, "complete": False, "labels_used": False},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    _validate_resume_index(output, rows)
    recorded_samples = int(previous_manifest.get("samples", 0))
    if len(rows) < recorded_samples or (
        previous_manifest.get("complete") and len(rows) != recorded_samples
    ):
        raise ValueError("resume manifest and index sample counts disagree")
    if previous_manifest.get("complete") and limit is None:
        return previous_manifest

    sources = load_source_info(source_info)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    replay = FunctionalTraceReplay.from_pretrained(
        model_path, device=device, dtype=dtype
    )
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
            route_cover_mass=route_cover_mass,
        )
        sample.release_attention()
        saved = _save(
            samples / f"{sample_id}.pt",
            capture,
            artifact_contract=artifact_contract,
        )
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
        "generator_models": sorted({str(row["generator_model"]) for row in rows}),
        "labels_used": False,
        "predictor_chunk": int(predictor_chunk),
        "logit_chunk": int(logit_chunk),
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
        state_root = Path(output_root) / STATE_DIRECTORY / split_root.name
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
