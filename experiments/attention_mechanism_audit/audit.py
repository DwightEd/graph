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
        "generator_model": artifact["generator_model"],
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
) -> dict[str, Any]:
    """Capture label-free raw dynamics, then persist each sample immediately."""

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
    rows: list[dict[str, Any]] = []
    role_cache: dict[str, torch.Tensor] = {}
    pending = None
    captured = 0

    with ThreadPoolExecutor(max_workers=1) as writer:
        for sample_id in dataset.sample_ids:
            if limit is not None and captured >= int(limit):
                break
            sample = dataset[sample_id]
            attention = sample.attention()
            if str(sample.task_type).casefold() != "qa":
                sample.release_attention()
                continue
            source_id = str(sample.source_id)
            captured += 1
            print(f"capture {captured}: {sample_id}", flush=True)
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
            artifact = {
                "schema": SCHEMA,
                "version": VERSION,
                "sample_id": str(sample_id),
                "source_id": source_id,
                "task_type": str(sample.task_type),
                "generator_model": sample.generator_model,
                "observer_checkpoint": replay.checkpoint,
                "labels_used": False,
                **capture,
            }
            artifact["mechanism"] = mechanism_effects(artifact["scores"])
            destination = samples / f"{sample_id}.pt"
            sample.release_attention()
            if pending is not None:
                rows.append(pending.result())
            pending = writer.submit(_save, destination, artifact)
        if pending is not None:
            rows.append(pending.result())

    index_path = output / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "samples": len(rows),
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
        "max_cuda_reserved_bytes": max(
            (row["peak_cuda_reserved_bytes"] for row in rows), default=0
        ),
        "index": str(index_path),
    }
    (output / "manifest.json").write_text(
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
