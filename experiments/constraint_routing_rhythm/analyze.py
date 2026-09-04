"""Label-free, one-sample-at-a-time constraint-rhythm analysis."""

from __future__ import annotations

import gc
import json
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from research_dataset import open_research_dataset

from .artifacts import load_result, save_result
from .capture import capture_sample
from .data import (
    TASK_TYPES,
    build_evidence_mask,
    canonical_task_type,
    load_source_info,
)
from .visualize import save_sample_figure

MANIFEST_NAME = "run_manifest.json"


def observer_matches_run(
    cache_observer_model: str,
    run_config: Mapping[str, object],
    model_id: str,
) -> bool:
    """Check that cached observations and current interventions use one model."""

    if not cache_observer_model:
        return True
    observed = Path(cache_observer_model)
    requested = Path(str(run_config["model"]))
    if observed.is_absolute():
        return observed.resolve() == requested.resolve()
    return cache_observer_model == model_id or observed.name == Path(model_id).name


def observer_identity(dataset, sample_observer_model: str) -> str:
    """Prefer the formal cache's full model path over its basename view."""

    model_path = getattr(dataset, "spec", {}).get("model_path")
    return str(model_path) if model_path else sample_observer_model


def save_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def open_manifest(
    output: Path,
    run_config: Mapping[str, object],
    audit_source_ids: Mapping[str, list[str]],
) -> dict[str, object]:
    """Start or resume exactly one JSON-declared analysis run."""

    path = output / MANIFEST_NAME
    config = json.loads(json.dumps(dict(run_config)))
    existing = {
        result.relative_to(output).as_posix()
        for result in (output / "results").rglob("*.npz")
        if not (result.name.startswith(".") and result.name.endswith(".tmp.npz"))
    }
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("config") != config:
            raise ValueError("output belongs to a different run configuration")
        if manifest.get("audit_source_ids") != audit_source_ids:
            raise ValueError("audit source selection changed for this output")
    else:
        if existing:
            raise ValueError("result files exist without a run manifest")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "config": config,
            "analysis_complete": False,
            "selected_samples": {},
            "audit_source_ids": dict(audit_source_ids),
            "audit_sample_ids": {task: [] for task in TASK_TYPES},
            "samples": [],
        }

    listed = {str(sample["result"]) for sample in manifest["samples"]}
    unexpected = existing - listed
    if unexpected:
        raise ValueError(f"result is not listed in run manifest: {min(unexpected)}")
    manifest["analysis_complete"] = False
    save_manifest(path, manifest)
    return manifest


def select_audit_sources(
    dataset,
    audit_limit: int,
    audit_seed: int,
    *,
    limit: int | None = None,
) -> dict[str, list[str]]:
    """Choose distinct label-free sources from the intended sample scope."""

    selected = {task: [] for task in TASK_TYPES}
    if not audit_limit:
        return selected

    source_tasks: dict[str, str] = {}
    counts = {task: 0 for task in TASK_TYPES}
    for sample_id in dataset.sample_ids:
        if limit is not None and all(count >= limit for count in counts.values()):
            break
        sample = dataset[sample_id]
        try:
            source_id = str(sample.source_id)
            task = canonical_task_type(sample.task_type)
        finally:
            sample.release_attention()
        if limit is not None and counts[task] >= limit:
            continue
        counts[task] += 1
        previous = source_tasks.setdefault(source_id, task)
        if previous != task:
            raise ValueError(f"source occurs in two tasks: {source_id}")

    randomizer = random.Random(audit_seed)
    for task in TASK_TYPES:
        candidates = [
            source_id
            for source_id, source_task in source_tasks.items()
            if source_task == task
        ]
        randomizer.shuffle(candidates)
        selected[task] = candidates[:audit_limit]
    return selected


def saved_result_matches(
    path: Path,
    entry: Mapping[str, object],
    *,
    sample_id: str,
    source_id: str,
    task_type: str,
    model_id: str,
    response_start: int,
    event_count: int,
    full_response_events: int,
    token_ids,
    evidence_positions: list[int],
    audit_requested: bool,
    plot_requested: bool,
    generator_model: str,
    observer_model: str,
) -> bool:
    """Confirm that a resume marker is the result declared by its manifest."""

    if not path.is_file() or entry.get("complete") is not True:
        return False
    result = load_result(path)
    prediction = np.asarray(result["prediction_position"], dtype=np.int64)
    expected = response_start + np.arange(event_count)
    identities_match = (
        str(result["sample_id"].item()) == sample_id
        and str(result["source_id"].item()) == source_id
        and str(result["task_type"].item()) == task_type
        and str(result["model_id"].item()) == model_id
        and str(result["generator_model"].item()) == generator_model
        and str(result["observer_model"].item()) == observer_model
        and str(entry["generator_model"]) == generator_model
        and str(entry["observer_model"]) == observer_model
        and int(entry["events"]) == event_count
        and int(entry["full_response_events"]) == full_response_events
        and list(entry["evidence_positions"]) == evidence_positions
        and int(result["evidence_tokens"].item()) == len(evidence_positions)
        and bool(entry["audit_requested"]) == audit_requested
        and bool(entry["plot_requested"]) == plot_requested
        and np.array_equal(prediction, expected)
    )
    target = np.asarray(token_ids, dtype=np.int64)[prediction]
    if not identities_match or not np.array_equal(result["target_token_id"], target):
        raise ValueError(f"saved result does not match its run manifest: {sample_id}")
    return bool(result["control_audited"].item()) == audit_requested


def analyze_split(
    model,
    tokenizer,
    split_root: str | Path,
    source_info: str | Path,
    output_root: str | Path,
    *,
    model_id: str,
    limit: int | None = None,
    audit_limit: int = 0,
    plot_limit: int = 0,
    max_events: int | None = None,
    head_quantile: float = 0.3,
    query_chunk: int = 128,
    window: int = 10,
    horizon_low: int = 10,
    horizon_high: int = 100,
    carrier_quantile: float = 0.75,
    mass_floor: float = 1e-6,
    max_carriers: int = 8,
    split_layer: int | None = None,
    audit_seed: int = 2026,
    run_config: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    """Analyze a label-free split in its fixed per-task data order.

    ``audit_limit`` chooses a seeded, source-disjoint subset from the split.
    ``plot_limit`` selects the first samples of each task. Both assignments are
    persisted so resuming cannot silently move diagnostics to later samples.
    """

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    if audit_limit < 0 or plot_limit < 0:
        raise ValueError("audit_limit and plot_limit must be nonnegative")
    if max_events is not None and max_events < 1:
        raise ValueError("max_events must be positive or None")

    dataset = open_research_dataset(
        split_root,
        device="cpu",
        retain_embedded_labels=False,
    )
    dataset_observer = observer_identity(dataset, "")
    if not observer_matches_run(dataset_observer, run_config, model_id):
        raise ValueError("cache observer differs from intervention model")
    sources = load_source_info(source_info)
    output = Path(output_root)
    manifest_path = output / MANIFEST_NAME
    audit_source_ids = select_audit_sources(
        dataset, audit_limit, audit_seed, limit=limit
    )
    manifest = open_manifest(output, run_config, audit_source_ids)
    manifest_samples = {str(entry["sample_id"]): entry for entry in manifest["samples"]}
    audit_samples = manifest["audit_sample_ids"]
    assigned_audit_sources = {
        str(entry["source_id"]): str(entry["sample_id"])
        for entry in manifest["samples"]
        if entry.get("audit_requested")
    }
    counts = {
        task: {
            "selected": 0,
            "saved": 0,
            "skipped": 0,
            "audited": 0,
            "plotted": 0,
        }
        for task in TASK_TYPES
    }
    selected_sample_ids = []

    for dataset_sample_id in dataset.sample_ids:
        if limit is not None and all(
            task_counts["selected"] >= limit for task_counts in counts.values()
        ):
            break
        sample = dataset[dataset_sample_id]
        sample_id = str(dataset_sample_id)
        task = canonical_task_type(sample.task_type)
        task_counts = counts[task]
        if limit is not None and task_counts["selected"] >= limit:
            sample.release_attention()
            continue

        ordinal = task_counts["selected"]
        task_counts["selected"] += 1
        selected_sample_ids.append(sample_id)
        make_plot = ordinal < plot_limit
        result_path = output / "results" / task / f"{sample_id}.npz"
        result_name = result_path.relative_to(output).as_posix()

        cached = sample.attention()
        response_start = int(cached.response_idx)
        source_id = str(sample.source_id)
        generator_model = getattr(sample, "generator_model", None)
        generator_model = "" if generator_model is None else str(generator_model)
        sample_observer_model = getattr(sample, "observer_model", None)
        sample_observer_model = (
            "" if sample_observer_model is None else str(sample_observer_model)
        )
        observer_model = observer_identity(dataset, sample_observer_model)
        if not observer_matches_run(observer_model, run_config, model_id):
            raise ValueError(
                f"cache observer differs from intervention model: {sample_id}"
            )
        audit_relay = assigned_audit_sources.get(source_id) == sample_id
        if (
            source_id in audit_source_ids[task]
            and source_id not in assigned_audit_sources
        ):
            assigned_audit_sources[source_id] = sample_id
            audit_samples[task].append(sample_id)
            audit_relay = True
        full_response_events = len(cached.token_ids) - response_start
        event_count = min(full_response_events, max_events or full_response_events)
        token_ids = cached.token_ids.detach().cpu().clone()
        del cached
        sample.release_attention()
        evidence_mask = build_evidence_mask(
            sources[source_id],
            tokenizer,
            token_ids,
            response_start,
        )
        evidence_positions = np.flatnonzero(evidence_mask).tolist()
        entry = manifest_samples.get(sample_id)
        if entry is not None and str(entry["result"]) != result_name:
            raise ValueError(f"sample has two result paths in manifest: {sample_id}")
        result_matches = entry is not None and saved_result_matches(
            result_path,
            entry,
            sample_id=sample_id,
            source_id=source_id,
            task_type=task,
            model_id=model_id,
            response_start=response_start,
            event_count=event_count,
            full_response_events=full_response_events,
            token_ids=token_ids,
            evidence_positions=evidence_positions,
            audit_requested=audit_relay,
            plot_requested=make_plot,
            generator_model=generator_model,
            observer_model=observer_model,
        )
        figure_exists = (output / "figures" / task / f"{sample_id}.png").is_file()
        if result_matches and (not make_plot or figure_exists):
            task_counts["skipped"] += 1
            task_counts["audited"] += int(audit_relay)
            task_counts["plotted"] += int(make_plot)
            continue

        pending = {
            "sample_id": sample_id,
            "source_id": source_id,
            "task_type": task,
            "generator_model": generator_model,
            "observer_model": observer_model,
            "result": result_name,
            "events": event_count,
            "full_response_events": full_response_events,
            "evidence_positions": evidence_positions,
            "audit_requested": audit_relay,
            "plot_requested": make_plot,
            "complete": False,
        }
        if entry is None:
            manifest["samples"].append(pending)
            manifest_samples[sample_id] = pending
        else:
            entry.clear()
            entry.update(pending)
            pending = entry
        save_manifest(manifest_path, manifest)

        if max_events is not None:
            token_ids = token_ids[: response_start + max_events]
        print(f"analyze {task} {ordinal + 1}: {sample_id}", flush=True)
        captured = capture_sample(
            model,
            token_ids,
            response_start,
            evidence_mask,
            sample_id=sample_id,
            source_id=source_id,
            task_type=task,
            model_id=model_id,
            audit_relay=audit_relay,
            head_quantile=head_quantile,
            query_chunk=query_chunk,
            window=window,
            horizon_low=horizon_low,
            horizon_high=horizon_high,
            carrier_quantile=carrier_quantile,
            mass_floor=mass_floor,
            max_carriers=max_carriers,
            split_layer=split_layer,
        )
        captured.arrays["generator_model"] = generator_model
        captured.arrays["observer_model"] = observer_model

        if make_plot:
            plot_evidence = torch.zeros(
                captured.routes.all_map.shape[1], dtype=torch.bool
            )
            plot_evidence[:response_start] = torch.as_tensor(evidence_mask)
            convert_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
            token_labels = None
            source_token_labels = None
            if callable(convert_tokens):
                response_ids = token_ids[response_start:].tolist()
                token_labels = [str(token) for token in convert_tokens(response_ids)]
                source_ids = token_ids[:-1].tolist()
                source_token_labels = [
                    str(token) for token in convert_tokens(source_ids)
                ]
            save_sample_figure(
                output / "figures" / task / f"{sample_id}.png",
                local_route=captured.routes.local_map,
                global_route=captured.routes.global_map,
                functional_reach=captured.rhythm.functional_reach,
                relay_capacity=captured.rhythm.relay_capacity,
                constraint_deficit=captured.arrays["constraint_deficit"],
                response_positions=captured.rhythm.prediction_position,
                token_labels=token_labels,
                source_token_labels=source_token_labels,
                response_start=response_start,
                evidence_mask=plot_evidence,
                carrier_mask=captured.rhythm.carrier_mask,
                title=f"{task}: {sample_id}",
            )
            task_counts["plotted"] += 1

        # The result is the resume marker, so it is written after its optional
        # figure and contains only the small arrays supplied by capture_sample.
        save_result(result_path, captured.arrays)
        pending["complete"] = True
        save_manifest(manifest_path, manifest)
        task_counts["saved"] += 1
        task_counts["audited"] += int(audit_relay)

        del captured, evidence_mask, token_ids, sample
        gc.collect()
        torch.cuda.empty_cache()

    if [
        str(entry["sample_id"]) for entry in manifest["samples"]
    ] != selected_sample_ids:
        raise ValueError("dataset sample scope changed for this run manifest")
    manifest["selected_samples"] = {
        task: task_counts["selected"] for task, task_counts in counts.items()
    }
    manifest["model_roles"] = {
        "intervention_model": model_id,
        "response_generator_models": sorted(
            {
                str(entry["generator_model"])
                for entry in manifest["samples"]
                if entry["generator_model"]
            }
        ),
        "observer_models": sorted(
            {
                str(entry["observer_model"])
                for entry in manifest["samples"]
                if entry["observer_model"]
            }
        ),
    }
    manifest["analysis_complete"] = True
    save_manifest(manifest_path, manifest)
    return counts
