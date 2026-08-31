"""Capture compact causal-route traces for real QA samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from research_dataset import open_research_dataset

from .capture import FunctionalTraceReplay
from .data import build_prompt_role_ids, load_source_info

SCHEMA = "ragtruth-functional-message-audit"
VERSION = 3


def _save(path: Path, capture: dict[str, Any]) -> dict[str, Any]:
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
    predictor_chunk: int = 64,
    top_k: int = 8,
    logit_chunk: int = 64,
    intervention_batch: int = 3,
) -> dict[str, Any]:
    """Capture every eligible sample, journaling progress for exact resume."""

    from transformers import AutoTokenizer

    observer_checkpoint = str(Path(model_path).resolve())
    output = Path(output_root)
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    manifest_path = output / "manifest.json"
    if index_path.is_file() and not manifest_path.is_file():
        raise ValueError("index exists without a v3 manifest; use a new output")
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    expected = {
        "schema": SCHEMA,
        "version": VERSION,
        "split_root": str(Path(split_root).resolve()),
        "source_info": str(Path(source_info).resolve()),
        "observer_checkpoint": observer_checkpoint,
        "model_dtype": str(dtype),
        "top_k_message_sources": int(top_k),
    }
    if previous_manifest:
        changed = [
            name
            for name, value in expected.items()
            if previous_manifest.get(name) != value
        ]
        if changed:
            raise ValueError(
                "cannot resume traces with changed provenance: " + ", ".join(changed)
            )
    else:
        initial = {
            **expected,
            "samples": 0,
            "complete": False,
            "labels_used": False,
        }
        manifest_path.write_text(
            json.dumps(initial, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    dataset = open_research_dataset(
        split_root, device="cpu", retain_embedded_labels=False
    )
    sources = load_source_info(source_info)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    replay = FunctionalTraceReplay.from_pretrained(
        model_path, device=device, dtype=dtype
    )
    rows = load_index(output) if index_path.is_file() else []
    indexed = {str(row["sample_id"]) for row in rows}
    resumed = len(rows)
    new_samples = 0
    role_cache: dict[str, torch.Tensor] = {}
    eligible = 0

    def record(saved: dict[str, Any]) -> None:
        nonlocal new_samples
        rows.append(saved)
        with index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(saved, sort_keys=True) + "\n")
        new_samples += 1

    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        if str(sample.task_type).casefold() != "qa":
            sample.release_attention()
            continue
        if limit is not None and eligible >= int(limit):
            sample.release_attention()
            break
        eligible += 1
        if str(sample_id) in indexed:
            sample.release_attention()
            continue

        attention = sample.attention()
        source_id = str(sample.source_id)
        print(f"capture {new_samples + 1} (eligible {eligible}): {sample_id}", flush=True)
        prompt_roles = role_cache.get(source_id)
        if prompt_roles is None:
            prompt_roles = torch.from_numpy(
                build_prompt_role_ids(
                    sources[source_id],
                    tokenizer,
                    attention.token_ids,
                    int(attention.response_idx),
                )
            )
            role_cache[source_id] = prompt_roles
        capture = replay.capture(
            attention.token_ids,
            int(attention.response_idx),
            prompt_roles,
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
                "generator_model": sample.generator_model,
                **saved,
            }
        )

    complete = limit is None or bool(previous_manifest.get("complete", False))
    eligible_total = (
        eligible
        if limit is None
        else previous_manifest.get("eligible_qa")
    )
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "samples": len(rows),
        "dataset_candidates": len(dataset.sample_ids),
        "eligible_qa": eligible_total,
        "selected_qa_seen": eligible,
        "resumed_samples": resumed,
        "new_samples": new_samples,
        "complete": complete,
        "split": dataset.manifest.get("split"),
        "split_root": str(Path(split_root).resolve()),
        "source_info": str(Path(source_info).resolve()),
        "observer_checkpoint": observer_checkpoint,
        "generator_models": sorted(
            {str(row["generator_model"]) for row in rows}
        ),
        "labels_used": False,
        "model_dtype": str(dtype),
        "predictor_chunk": int(predictor_chunk),
        "intervention_batch": int(intervention_batch),
        "top_k_message_sources": int(top_k),
        "max_cuda_reserved_bytes": max(
            (row["peak_cuda_reserved_bytes"] for row in rows), default=0
        ),
        "index": str(index_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def load_index(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / "index.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
