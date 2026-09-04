"""Label-free traversal for one-model re-anchor flow discovery."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from research_dataset import open_research_dataset
from experiments.common.ragtruth_alignment import (
    TASK_TYPES,
    build_evidence_mask,
    canonical_task_type,
    load_source_info,
)

from .artifacts import save_result
from .capture import capture_sample
from .visualize import save_sample_figure

MANIFEST = "run_manifest.json"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def same_model(recorded: str, requested: str) -> bool:
    return not recorded or recorded == requested or Path(recorded).name == Path(requested).name


def open_manifest(output: Path, config: dict) -> dict:
    path = output / MANIFEST
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError("output already belongs to a different experiment")
        manifest["analysis_complete"] = False
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    return {"config": config, "analysis_complete": False, "samples": []}


def analyze_split(
    model,
    tokenizer,
    dataset_root: str | Path,
    source_info: str | Path,
    output_root: str | Path,
    *,
    model_path: str,
    model_id: str,
    limit: int | None = None,
    audit_limit: int = 0,
    plot_limit: int = 3,
    max_events: int | None = None,
    query_chunk: int = 128,
    min_claim_tokens: int = 2,
    max_claim_tokens: int = 96,
    anchor_width: int = 3,
    reread_window: int = 5,
    backbone_cover: float = 0.8,
    backbone_edges: int = 32,
) -> dict[str, dict[str, int]]:
    dataset = open_research_dataset(
        dataset_root,
        device="cpu",
        retain_embedded_labels=False,
    )
    cache_model = str(getattr(dataset, "spec", {}).get("model_path", ""))
    if not same_model(cache_model, model_path):
        raise ValueError("cached observer and graph/intervention model differ")

    sources = load_source_info(source_info)
    output = Path(output_root)
    config = {
        "model": str(Path(model_path).resolve()),
        "model_id": model_id,
        "dataset_root": str(Path(dataset_root).resolve()),
        "source_info": str(Path(source_info).resolve()),
        "limit_per_task": limit,
        "audit_limit_per_task": audit_limit,
        "plot_limit_per_task": plot_limit,
        "max_events": max_events,
        "query_chunk": query_chunk,
        "min_claim_tokens": min_claim_tokens,
        "max_claim_tokens": max_claim_tokens,
        "anchor_width": anchor_width,
        "reread_window": reread_window,
        "backbone_cover": backbone_cover,
        "backbone_edges": backbone_edges,
    }
    manifest = open_manifest(output, config)
    completed = {str(row["sample_id"]) for row in manifest["samples"]}
    audited_sources = {task: set() for task in TASK_TYPES}
    for row in manifest["samples"]:
        if row.get("audited"):
            audited_sources[canonical_task_type(row["task_type"])].add(
                str(row["source_id"])
            )
    counts = {
        task: {"selected": 0, "saved": 0, "skipped": 0, "audited": 0}
        for task in TASK_TYPES
    }

    for dataset_sample_id in dataset.sample_ids:
        if limit is not None and all(
            counts[task]["selected"] >= limit for task in TASK_TYPES
        ):
            break
        sample = dataset[dataset_sample_id]
        sample_id = str(dataset_sample_id)
        task = canonical_task_type(sample.task_type)
        if limit is not None and counts[task]["selected"] >= limit:
            sample.release_attention()
            continue
        ordinal = counts[task]["selected"]
        counts[task]["selected"] += 1

        cached = sample.attention()
        token_ids = cached.token_ids.detach().cpu().clone()
        response_start = int(cached.response_idx)
        source_id = str(sample.source_id)
        generator = getattr(sample, "generator_model", None)
        observer = getattr(sample, "observer_model", None)
        generator = "" if generator is None else str(generator)
        observer = cache_model or ("" if observer is None else str(observer))
        del cached
        sample.release_attention()
        if not same_model(observer, model_path):
            raise ValueError(f"sample observer differs from current model: {sample_id}")
        if max_events is not None:
            token_ids = token_ids[: response_start + max_events]

        audit = (
            source_id not in audited_sources[task]
            and len(audited_sources[task]) < audit_limit
        )
        if audit:
            audited_sources[task].add(source_id)
        result_path = output / "results" / task / f"{sample_id}.npz"
        if sample_id in completed and result_path.is_file():
            counts[task]["skipped"] += 1
            counts[task]["audited"] += int(audit)
            continue

        evidence_mask = build_evidence_mask(
            sources[source_id], tokenizer, token_ids, response_start
        )
        print(f"reanchor {task} {ordinal + 1}: {sample_id}", flush=True)
        captured = capture_sample(
            model,
            tokenizer,
            token_ids,
            response_start,
            evidence_mask,
            sample_id=sample_id,
            source_id=source_id,
            task_type=task,
            model_id=model_id,
            audit=audit,
            query_chunk=query_chunk,
            min_claim_tokens=min_claim_tokens,
            max_claim_tokens=max_claim_tokens,
            anchor_width=anchor_width,
            reread_window=reread_window,
            backbone_cover=backbone_cover,
            backbone_edges=backbone_edges,
        )
        captured.arrays["generator_model"] = generator
        captured.arrays["observer_model"] = observer
        save_result(result_path, captured.arrays)

        if ordinal < plot_limit:
            convert = getattr(tokenizer, "convert_ids_to_tokens", None)
            labels = None
            if callable(convert):
                labels = [
                    str(token)
                    for token in convert(token_ids[response_start:].tolist())
                ]
            save_sample_figure(
                output / "figures" / task / f"{sample_id}.png",
                captured,
                response_labels=labels,
                title=f"{task}: {sample_id}",
            )

        manifest["samples"].append(
            {
                "sample_id": sample_id,
                "source_id": source_id,
                "task_type": task,
                "result": result_path.relative_to(output).as_posix(),
                "events": len(token_ids) - response_start,
                "audited": audit,
                "generator_model": generator,
                "observer_model": observer,
            }
        )
        save_json(output / MANIFEST, manifest)
        completed.add(sample_id)
        counts[task]["saved"] += 1
        counts[task]["audited"] += int(audit)
        del captured, evidence_mask, token_ids, sample
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["analysis_complete"] = True
    manifest["selected_samples"] = {
        task: counts[task]["selected"] for task in TASK_TYPES
    }
    save_json(output / MANIFEST, manifest)
    return counts
