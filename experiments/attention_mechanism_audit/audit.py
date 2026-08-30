"""Stream real QA samples through the frozen functional-flow audit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import torch

from research_dataset import open_research_dataset

from .capture import FunctionalTraceReplay
from .data import build_prompt_role_ids, load_source_info


SCHEMA = "ragtruth-functional-message-audit"
VERSION = 2


def mechanism_effects(scores: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Factorial same-sample effects on observed-token log probability."""

    full = scores["full"]["target_logprob"]
    no_evidence = scores["evidence_removed"]["target_logprob"]
    no_response = scores["response_removed"]["target_logprob"]
    no_context = scores["evidence_response_removed"]["target_logprob"]
    return {
        "evidence_message_effect": full - no_evidence,
        "response_message_effect": full - no_response,
        "evidence_message_effect_without_response": no_response - no_context,
        "response_message_effect_without_evidence": no_evidence - no_context,
        "evidence_response_message_interaction": (
            full - no_evidence - no_response + no_context
        ),
        "evidence_response_removed_logprob": no_context,
        "evidence_response_removed_margin": scores["evidence_response_removed"][
            "target_margin"
        ],
        "full_margin": scores["full"]["target_margin"],
    }


def _save(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    torch.save(artifact, path)
    return {
        "sample_id": artifact["sample_id"],
        "source_id": artifact["source_id"],
        "task_type": artifact["task_type"],
        "split": artifact.get("split"),
        "generator_model": artifact["generator_model"],
        "trace_level": artifact.get("trace_level", "raw"),
        "path": path.name,
        "tokens": int(len(artifact["token_ids"])),
        "response_tokens": int(len(artifact["target_ids"])),
        "bytes": path.stat().st_size,
        "peak_cuda_reserved_bytes": int(artifact["peak_cuda_reserved_bytes"]),
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
    trace_level: str = "mechanism",
) -> dict[str, Any]:
    """Capture every eligible sample, journaling progress for exact resume."""

    from transformers import AutoTokenizer

    dataset = open_research_dataset(
        split_root, device="cpu", retain_embedded_labels=False
    )
    sources = load_source_info(source_info)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    replay = FunctionalTraceReplay.from_pretrained(
        model_path, device=device, dtype=dtype
    )

    output = Path(output_root)
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.jsonl"
    manifest_path = output / "manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    if previous_manifest:
        expected = {
            "schema": SCHEMA,
            "version": VERSION,
            "split_root": str(Path(split_root).resolve()),
            "source_info": str(Path(source_info).resolve()),
            "observer_checkpoint": replay.checkpoint,
            "model_dtype": str(dtype),
            "top_k_message_sources": int(top_k),
        }
        changed = [
            name
            for name, value in expected.items()
            if previous_manifest.get(name, value) != value
        ]
        if changed:
            raise ValueError(
                "cannot resume traces with changed provenance: " + ", ".join(changed)
            )
    rows = load_index(output) if index_path.is_file() else []
    existing_levels = {
        row.get("trace_level", previous_manifest.get("trace_level", "raw"))
        for row in rows
    }
    if trace_level == "raw" and "mechanism" in existing_levels:
        raise ValueError("raw capture requires a new output when compact traces exist")
    indexed = {str(row["sample_id"]) for row in rows}
    resumed = len(rows)
    new_samples = 0
    role_cache: dict[str, torch.Tensor] = {}
    pending = None
    eligible = 0

    def record(saved: dict[str, Any]) -> None:
        nonlocal new_samples
        rows.append(saved)
        indexed.add(str(saved["sample_id"]))
        with index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(saved, sort_keys=True) + "\n")
        new_samples += 1

    with ThreadPoolExecutor(max_workers=1) as writer:
        for sample_id in dataset.sample_ids:
            sample = dataset[sample_id]
            if str(sample.task_type).casefold() != "qa":
                sample.release_attention()
                continue
            if limit is not None and eligible >= int(limit):
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
                retain_raw=trace_level == "raw",
            )
            artifact = {
                "schema": SCHEMA,
                "version": VERSION,
                "sample_id": str(sample_id),
                "source_id": source_id,
                "task_type": str(sample.task_type),
                "generator_model": sample.generator_model,
                "observer_checkpoint": replay.checkpoint,
                "split": dataset.manifest.get("split"),
                "trace_level": trace_level,
                "labels_used": False,
                **capture,
            }
            artifact["mechanism"] = mechanism_effects(artifact["scores"])
            destination = samples / f"{sample_id}.pt"
            sample.release_attention()
            if pending is not None:
                record(pending.result())
            pending = writer.submit(_save, destination, artifact)
        if pending is not None:
            record(pending.result())

    complete = limit is None or bool(previous_manifest.get("complete", False))
    eligible_total = (
        eligible
        if limit is None
        else previous_manifest.get(
            "eligible_qa", previous_manifest.get("eligible_qa_seen")
        )
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
        "observer_checkpoint": replay.checkpoint,
        "generator_models": sorted(
            {str(row["generator_model"]) for row in rows}
        ),
        "labels_used": False,
        "model_dtype": str(dtype),
        "predictor_chunk": int(predictor_chunk),
        "intervention_batch": int(intervention_batch),
        "top_k_message_sources": int(top_k),
        "trace_levels": sorted(
            existing_levels | ({trace_level} if new_samples else set())
        ),
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


__all__ = [
    "SCHEMA",
    "VERSION",
    "capture_split",
    "load_index",
    "mechanism_effects",
]
