"""Stream one-pass routing-rhythm captures to disk."""

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
from .capture import CAPTURE_SCHEMA, capture_sample

MANIFEST = "run_manifest.json"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def same_model(recorded: str, requested: str) -> bool:
    return (
        not recorded
        or recorded == requested
        or Path(recorded).name == Path(requested).name
    )


def open_manifest(output: Path, config: dict) -> dict:
    path = output / MANIFEST
    if not path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        return {"config": config, "analysis_complete": False, "samples": []}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("config") != config:
        raise ValueError("output contains another configuration; choose a new --output")
    manifest["analysis_complete"] = False
    return manifest


def analyze_split(
    model,
    tokenizer,
    dataset_root: str | Path,
    source_info: str | Path,
    output_root: str | Path,
    *,
    model_path: str,
    model_id: str,
    dtype: str,
    limit: int | None = None,
    max_events: int | None = None,
    query_chunk: int = 64,
    route_window: int = 4,
    future_horizon: int = 16,
    distance_scale: int = 16,
    peak_quantile: float = 0.9,
    max_lag: int = 3,
    plot_limit: int = 1,
    plot_sample_id: str | None = None,
) -> dict[str, int]:
    dataset = open_research_dataset(
        dataset_root,
        device="cpu",
        retain_embedded_labels=False,
    )
    cache_model = str(getattr(dataset, "spec", {}).get("model_path", ""))
    if not same_model(cache_model, model_path):
        raise ValueError("cached observer and current model differ")

    output = Path(output_root)
    config = {
        "capture_schema": CAPTURE_SCHEMA,
        "model": str(Path(model_path).resolve()),
        "model_id": model_id,
        "dtype": dtype,
        "dataset_root": str(Path(dataset_root).resolve()),
        "source_info": str(Path(source_info).resolve()),
        "limit_per_task": limit,
        "max_events": max_events,
        "query_chunk": query_chunk,
        "route_window": route_window,
        "future_horizon": future_horizon,
        "distance_scale": distance_scale,
        "peak_quantile": peak_quantile,
        "max_lag": max_lag,
        "plot_limit": plot_limit,
        "plot_sample_id": plot_sample_id,
    }
    manifest = open_manifest(output, config)
    manifest["samples"] = [
        row for row in manifest["samples"] if (output / row["result"]).is_file()
    ]
    completed = {str(row["sample_id"]) for row in manifest["samples"]}
    detailed = sum(bool(row.get("detail")) for row in manifest["samples"])
    counts = {task: 0 for task in TASK_TYPES}
    sources = load_source_info(source_info)

    for dataset_sample_id in dataset.sample_ids:
        if limit is not None and all(counts[task] >= limit for task in TASK_TYPES):
            break
        sample = dataset[dataset_sample_id]
        sample_id = str(dataset_sample_id)
        if plot_sample_id is not None and sample_id != plot_sample_id:
            sample.release_attention()
            continue
        task = canonical_task_type(sample.task_type)
        if limit is not None and counts[task] >= limit:
            sample.release_attention()
            continue
        counts[task] += 1
        result_path = output / "results" / task / f"{sample_id}.npz"
        if sample_id in completed and result_path.is_file():
            sample.release_attention()
            if plot_sample_id is not None:
                break
            continue

        cached = sample.attention()
        token_ids = cached.token_ids.detach().cpu().clone()
        response_start = int(cached.response_idx)
        source_id = str(sample.source_id)
        generator = str(getattr(sample, "generator_model", "") or "")
        cached_observer = cache_model or str(
            getattr(sample, "observer_model", "") or ""
        )
        sample.release_attention()
        if max_events is not None:
            token_ids = token_ids[: response_start + max_events]

        evidence_mask = build_evidence_mask(
            sources[source_id], tokenizer, token_ids, response_start
        )
        detail = (
            sample_id == plot_sample_id
            if plot_sample_id is not None
            else detailed < plot_limit
        )
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
            query_chunk=query_chunk,
            route_window=route_window,
            future_horizon=future_horizon,
            distance_scale=distance_scale,
            peak_quantile=peak_quantile,
            max_lag=max_lag,
            detail=detail,
        )
        captured.arrays.update(
            generator_model=generator,
            observer_model=str(Path(model_path).resolve()),
            cached_observer_model=cached_observer,
            dtype=dtype,
        )
        save_result(result_path, captured.arrays)
        manifest["samples"].append(
            {
                "sample_id": sample_id,
                "source_id": source_id,
                "task_type": task,
                "result": result_path.relative_to(output).as_posix(),
                "detail": detail,
            }
        )
        save_json(output / MANIFEST, manifest)
        completed.add(sample_id)
        detailed += int(detail)
        del captured, evidence_mask, token_ids, sample
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if plot_sample_id is not None:
            break

    if plot_sample_id is not None and not any(
        str(row["sample_id"]) == plot_sample_id for row in manifest["samples"]
    ):
        raise ValueError(f"sample id not found: {plot_sample_id}")
    manifest["analysis_complete"] = True
    manifest["selected_samples"] = counts
    save_json(output / MANIFEST, manifest)
    return counts
