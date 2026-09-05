"""Stream routing, functional context and optional grouped captures to disk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def _hash_rank(*parts: str) -> bytes:
    joined = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(joined).digest()


def _select_mechanism_records(
    records: list[tuple[str, str, str]], limit: int
) -> set[str]:
    """Select a deterministic, source-diverse mechanism subset per task."""

    selected: set[str] = set()
    for task in TASK_TYPES:
        task_records = [record for record in records if record[0] == task]
        by_source: dict[str, list[str]] = {}
        for _, source_id, sample_id in task_records:
            by_source.setdefault(source_id, []).append(sample_id)
        primary = []
        for source_id, sample_ids in by_source.items():
            sample_id = min(
                sample_ids,
                key=lambda item: _hash_rank(task, source_id, item),
            )
            primary.append((source_id, sample_id))
        primary.sort(key=lambda item: _hash_rank(task, item[0]))
        chosen = [sample_id for _, sample_id in primary[:limit]]
        if len(chosen) < limit:
            remaining = sorted(
                (
                    (source_id, sample_id)
                    for _, source_id, sample_id in task_records
                    if sample_id not in chosen
                ),
                key=lambda item: _hash_rank(task, item[0], item[1]),
            )
            chosen.extend(sample_id for _, sample_id in remaining[: limit - len(chosen)])
        selected.update(chosen)
    return selected


def _mechanism_sample_ids(dataset, limit: int, capture_limit: int | None) -> set[str]:
    records: list[tuple[str, str, str]] = []
    counts = {task: 0 for task in TASK_TYPES}
    for dataset_sample_id in dataset.sample_ids:
        sample = dataset[dataset_sample_id]
        task = canonical_task_type(sample.task_type)
        if capture_limit is None or counts[task] < capture_limit:
            counts[task] += 1
            records.append((task, str(sample.source_id), str(dataset_sample_id)))
        sample.release_attention()
    return _select_mechanism_records(records, limit)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def same_model(recorded: str, requested: str) -> bool:
    return not recorded or recorded == requested or Path(recorded).name == Path(requested).name


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
    mechanism_limit: int = 0,
) -> dict[str, int]:
    dataset = open_research_dataset(
        dataset_root, device="cpu", retain_embedded_labels=False
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
        "functional_pass": "all_selected",
        "mechanism_limit_per_task": mechanism_limit,
        "mechanism_sampling": "source_hash_v1",
    }
    manifest = open_manifest(output, config)
    manifest["samples"] = [
        row for row in manifest["samples"] if (output / row["result"]).is_file()
    ]
    completed = {str(row["sample_id"]) for row in manifest["samples"]}
    detailed = sum(bool(row.get("detail")) for row in manifest["samples"])
    counts = {task: 0 for task in TASK_TYPES}
    mechanism_counts = {
        task: sum(
            bool(row.get("mechanism")) and row.get("task_type") == task
            for row in manifest["samples"]
        )
        for task in TASK_TYPES
    }
    sources = load_source_info(source_info)
    mechanism_selection = (
        _mechanism_sample_ids(dataset, mechanism_limit, limit)
        if mechanism_limit > 0 and plot_sample_id is None
        else set()
    )

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
        del cached
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
        mechanism = (
            plot_sample_id is not None
            or mechanism_limit < 0
            or sample_id in mechanism_selection
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
            mechanism=mechanism,
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
                "functional": True,
                "mechanism": mechanism,
            }
        )
        save_json(output / MANIFEST, manifest)
        completed.add(sample_id)
        detailed += int(detail)
        mechanism_counts[task] += int(mechanism)
        del captured, evidence_mask, token_ids, sample
        if plot_sample_id is not None:
            break

    if plot_sample_id is not None and plot_sample_id not in completed:
        raise ValueError(f"sample id not found: {plot_sample_id}")
    manifest["analysis_complete"] = True
    manifest["selected_samples"] = counts
    manifest["functional_samples"] = counts
    manifest["mechanism_samples"] = mechanism_counts
    save_json(output / MANIFEST, manifest)
    return counts
