"""Collect label-free shortcut-route artifacts from RAGTruth caches.

Collection owns only dataset traversal, input identity, atomic persistence, and
resume. The native model observer and route construction remain in their
dedicated modules; hallucination labels are never retained, exposed, or
consulted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from research_dataset import open_research_dataset

from .data import (
    TASK_TYPES,
    build_evidence_mask,
    canonical_task_type,
    load_source_info,
)
from .route_artifact import SCHEMA as ARTIFACT_SCHEMA
from .route_artifact import save_route_artifact
from .route_capture import NativeRouteObserver
from .route_pipeline import build_route_artifact

SCHEMA = "shortcut-route-collection-v1"
VERSION = 1
STATE_DIRECTORY = "shortcut_route_state"
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
_IDENTITY_FIELDS = (
    "schema",
    "version",
    "artifact_schema",
    "dataset_identity",
    "source_identity",
    "observer_identity",
    "model_dtype",
    "top_k",
    "cover_mass",
    "task_types",
    "index",
    "labels_used",
)
_MODEL_STAMP_CACHE: dict[
    tuple[str, tuple[tuple[str, int, int], ...]], dict[str, Any]
] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_stamp(path: str | Path) -> dict[str, Any]:
    """Return a readable identity stamp for one collection input."""

    path = Path(path)
    stamp: dict[str, Any] = {"path": str(path.resolve())}
    if path.is_file():
        stat = path.stat()
        stamp.update(size=stat.st_size, sha256=_sha256(path))
    return stamp


def model_stamp(path: str | Path) -> dict[str, Any]:
    """Bind a collection to the local observer checkpoint files."""

    path = Path(path)
    if not path.is_dir():
        return {"path": str(path.resolve()), "files": []}
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.casefold() in _MODEL_SUFFIXES
    )
    metadata = tuple(
        (
            candidate.relative_to(path).as_posix(),
            candidate.stat().st_size,
            candidate.stat().st_mtime_ns,
        )
        for candidate in files
    )
    cache_key = (str(path.resolve()), metadata)
    cached = _MODEL_STAMP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    entries = [
        [relative, size, _sha256(candidate)]
        for candidate, (relative, size, _modified_ns) in zip(
            files, metadata, strict=True
        )
    ]
    digest = hashlib.sha256()
    for relative, size, file_digest in entries:
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(str(file_digest).encode())
        digest.update(b"\0")
    stamp = {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "files": entries,
    }
    _MODEL_STAMP_CACHE[cache_key] = stamp
    return stamp


def split_stamp(dataset: Any, path: str | Path) -> dict[str, Any]:
    """Bind a collection to the opened label-free dataset shard."""

    root = Path(path)
    return {
        "path": str(root.resolve()),
        "manifest_file": file_stamp(root / "manifest.json"),
        "manifest": json.loads(json.dumps(dataset.manifest, default=str)),
        "samples": len(dataset.sample_ids),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write_index(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _artifact_name(sample_id: object) -> str:
    value = str(sample_id)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("sample_id must be a safe filename component")
    return f"{value}.npz"


def _atomic_save_artifact(path: Path, artifact: Any) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        save_route_artifact(temporary, artifact)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_index(root: str | Path) -> list[dict[str, Any]]:
    """Load the atomic sample index, or an empty index before first capture."""

    path = Path(root) / "index.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_resume_index(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Check that every committed journal row still names its artifact."""

    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("resume index contains duplicate sample IDs")
    for row in rows:
        artifact_name = str(row["path"])
        if artifact_name != _artifact_name(row["sample_id"]):
            raise ValueError("resume index artifact path does not match its sample ID")
        path = root / "samples" / artifact_name
        if not path.is_file():
            raise ValueError("resume index references a missing sample artifact")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError("resume index references a size-changed sample artifact")
        if _sha256(path) != str(row["sha256"]):
            raise ValueError("resume index references a digest-changed sample artifact")


def _expected_manifest(
    *,
    dataset: Any,
    split_root: str | Path,
    source_info: str | Path,
    model_path: str | Path,
    dtype: torch.dtype,
    top_k: int,
    cover_mass: float,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "artifact_schema": ARTIFACT_SCHEMA,
        "dataset_identity": split_stamp(dataset, split_root),
        "source_identity": file_stamp(source_info),
        "observer_identity": model_stamp(model_path),
        "model_dtype": str(dtype),
        "top_k": int(top_k),
        "cover_mass": float(cover_mass),
        "task_types": list(TASK_TYPES),
        "index": "index.jsonl",
        "dataset_candidates": len(dataset.sample_ids),
        "samples": 0,
        "complete": False,
        "labels_used": False,
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
    top_k: int = 64,
    cover_mass: float = 0.95,
) -> dict[str, Any]:
    """Capture one shard with a full-sequence observer and exact resume."""

    from transformers import AutoTokenizer

    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
    ):
        raise ValueError("limit must be a nonnegative integer or None")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if not math.isfinite(cover_mass) or not 0 < cover_mass <= 1:
        raise ValueError("cover_mass must be in (0, 1]")

    dataset = open_research_dataset(
        split_root,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=False,
    )
    output = Path(output_root)
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    manifest_path = output / "manifest.json"
    if index_path.is_file() and not manifest_path.is_file():
        raise ValueError("index exists without its collection manifest")

    rows = load_index(output)
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    expected = _expected_manifest(
        dataset=dataset,
        split_root=split_root,
        source_info=source_info,
        model_path=model_path,
        dtype=dtype,
        top_k=top_k,
        cover_mass=cover_mass,
    )
    if previous:
        changed = [
            field
            for field in _IDENTITY_FIELDS
            if previous.get(field) != expected[field]
        ]
        if changed and (rows or int(previous.get("samples", 0))):
            raise ValueError(
                "cannot resume shortcut routes with changed identity: "
                + ", ".join(changed)
            )
        if changed:
            previous = {}
    if not previous:
        _atomic_write_json(manifest_path, expected)
        previous = expected

    _validate_resume_index(output, rows)
    recorded = int(previous.get("samples", 0))
    complete = previous.get("complete")
    if not isinstance(complete, bool):
        raise TypeError("resume manifest complete must be boolean")
    if len(rows) < recorded or (complete and len(rows) != recorded):
        raise ValueError("resume manifest and index sample counts disagree")
    dataset_ids = [str(sample_id) for sample_id in dataset.sample_ids]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("canonical dataset sample IDs must be unique")
    dataset_order = {sample_id: index for index, sample_id in enumerate(dataset_ids)}
    indexed_ids = [str(row["sample_id"]) for row in rows]
    if any(sample_id not in dataset_order for sample_id in indexed_ids) or [
        dataset_order[sample_id] for sample_id in indexed_ids
    ] != sorted(dataset_order[sample_id] for sample_id in indexed_ids):
        raise ValueError("resume index is not a canonical dataset subsequence")
    if complete:
        if (
            recorded != len(dataset_ids)
            or int(previous.get("dataset_candidates", -1)) != len(dataset_ids)
            or indexed_ids != dataset_ids
        ):
            raise ValueError(
                "complete manifest does not cover the canonical dataset in order"
            )
        return previous

    sources = load_source_info(source_info)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    observer = NativeRouteObserver.from_pretrained(
        model_path, device=device, dtype=dtype
    )
    indexed = {
        str(row["sample_id"]): canonical_task_type(row["task_type"]) for row in rows
    }
    selected = {task: 0 for task in TASK_TYPES}
    evidence_cache: dict[str, torch.Tensor] = {}

    for sample_id in dataset.sample_ids:
        if limit is not None and all(count >= limit for count in selected.values()):
            break
        key = str(sample_id)
        if key in indexed:
            task_type = indexed[key]
            if limit is not None and selected[task_type] >= limit:
                continue
            selected[task_type] += 1
            continue

        sample = dataset[sample_id]
        task_type = canonical_task_type(sample.task_type)
        if limit is not None and selected[task_type] >= limit:
            continue
        selected[task_type] += 1
        source_id = str(sample.source_id)
        generator_model = (
            "" if sample.generator_model is None else str(sample.generator_model)
        )
        attention = sample.attention()
        try:
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
            print(f"collect {len(rows) + 1} ({task_type}): {key}", flush=True)
            captured = observer.capture(
                attention.token_ids,
                int(attention.response_idx),
                evidence_mask,
            )
            try:
                artifact = build_route_artifact(
                    captured,
                    top_k=top_k,
                    cover_mass=cover_mass,
                )
            finally:
                del captured
        finally:
            sample.release_attention()

        try:
            artifact_path = samples / _artifact_name(key)
            _atomic_save_artifact(artifact_path, artifact)
            row = {
                "sample_id": key,
                "source_id": source_id,
                "task_type": task_type,
                "generator_model": generator_model,
                "path": artifact_path.name,
                "bytes": artifact_path.stat().st_size,
                "sha256": _sha256(artifact_path),
                "events": len(artifact.events.query_position),
                "response_start": int(artifact.response_start),
            }
            rows.append(row)
            rows.sort(key=lambda current: dataset_order[str(current["sample_id"])])
            try:
                _atomic_write_index(index_path, rows)
            except BaseException:
                rows.remove(row)
                raise
        finally:
            del artifact

    indexed_ids = [str(row["sample_id"]) for row in rows]
    if limit is None and indexed_ids != dataset_ids:
        raise ValueError("completed capture does not cover the canonical dataset")
    manifest = {
        **expected,
        "samples": len(rows),
        "complete": limit is None,
    }
    _atomic_write_json(manifest_path, manifest)
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
    top_k: int = 64,
    cover_mass: float = 0.95,
) -> dict[str, list[tuple[Path, Path]]]:
    """Collect physical shards once and expose the pairs for each task view."""

    pairs: list[tuple[Path, Path]] = []
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
            top_k=top_k,
            cover_mass=cover_mass,
        )
        pairs.append((state_root, split_root))
    return {task: list(pairs) for task in TASK_TYPES}
