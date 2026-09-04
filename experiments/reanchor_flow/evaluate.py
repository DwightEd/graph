"""Open labels after capture and write concise mechanism reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from research_dataset import open_research_dataset
from experiments.common.ragtruth_alignment import TASK_TYPES, canonical_task_type

from .artifacts import load_result
from .capture import CAPTURE_SCHEMA
from .report import task_summary
from .visualize import (
    save_mechanism_figure,
    save_population_figure,
    save_sample_figure,
    save_sample_mechanism_figure,
)

BASE_FIELDS = (
    "prompt_delta",
    "evidence_delta",
    "nonlocal_delta",
    "route_change",
    "future_influence",
    "prompt_lift",
    "history_lift",
    "transition_peak",
    "prompt_peak",
    "review_peak",
    "anchor_peak",
    "prompt_paired_anchor",
    "review_paired_anchor",
    "prompt_coupling_rate",
    "prompt_coupling_null_rate",
    "prompt_median_anchor_lag",
    "review_coupling_rate",
    "review_coupling_null_rate",
    "review_median_anchor_lag",
    "evidence_share_layer",
)
MECHANISM_FIELDS = (
    "evidence_state_presence",
    "evidence_state_control",
    "evidence_readout_gain",
    "evidence_effect",
    "other_prompt_effect",
    "prompt_effect",
    "evidence_prompt_interaction",
    "history_effect",
    "evidence_peak_control",
    "evidence_late_control_loss",
)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_result(result: dict, token_ids, response_start: int) -> int:
    if int(np.asarray(result["capture_schema"]).item()) != CAPTURE_SCHEMA:
        raise ValueError("stale re-anchor artifact")
    prediction = np.asarray(result["prediction_position"], dtype=np.int64)
    query = np.asarray(result["query_position"], dtype=np.int64)
    target = np.asarray(result["target_token_id"], dtype=np.int64)
    if not np.array_equal(prediction, response_start + np.arange(len(prediction))):
        raise ValueError("prediction positions are not token aligned")
    if not np.array_equal(query + 1, prediction):
        raise ValueError("query/target relation is not q=p-1")
    if not np.array_equal(target, np.asarray(token_ids)[prediction]):
        raise ValueError("target token ids changed")
    return len(prediction)


def evaluate_results(
    output_root: str | Path,
    dataset_root: str | Path,
    *,
    bootstrap: int = 1000,
    seed: int = 2026,
    curve_radius: int = 6,
) -> dict[str, dict]:
    output = Path(output_root)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("analysis_complete"):
        raise ValueError("analysis is incomplete")
    if manifest.get("config", {}).get("capture_schema") != CAPTURE_SCHEMA:
        raise ValueError("output belongs to another capture schema")

    dataset = open_research_dataset(dataset_root, device="cpu", retain_embedded_labels=True)
    labels = dataset.prepare_evaluation_labels([])
    by_task = {task: [] for task in TASK_TYPES}
    for entry in manifest["samples"]:
        result = load_result(output / entry["result"])
        sample = dataset[str(entry["sample_id"])]
        cached = sample.attention()
        response_start = int(cached.response_idx)
        token_ids = cached.token_ids.detach().cpu().numpy()
        count = validate_result(result, token_ids, response_start)
        label = np.asarray(labels.response_labels(sample), dtype=bool)[:count]
        sample.release_attention()
        task = canonical_task_type(sample.task_type)
        row = {
            "source_id": str(entry["source_id"]),
            "label": label,
            "target_token_id": np.asarray(result["target_token_id"], dtype=np.int64),
            "mechanism": bool(int(np.asarray(result.get("mechanism", 0)).item())),
            **{name: np.asarray(result[name]) for name in BASE_FIELDS},
        }
        if row["mechanism"]:
            row.update({name: np.asarray(result[name]) for name in MECHANISM_FIELDS})
        by_task[task].append(row)

        if int(np.asarray(result.get("detail", 0)).item()):
            stem = output / "figures" / f"sample_{task}_{entry['sample_id']}"
            save_sample_figure(
                stem.with_suffix(".png"), result, label, title=f"{task} {entry['sample_id']}"
            )
            if row["mechanism"]:
                save_sample_mechanism_figure(
                    stem.with_name(stem.name + "_mechanism").with_suffix(".png"),
                    result,
                    label,
                    title=f"{task} {entry['sample_id']}",
                )

    reports = {}
    for task_index, task in enumerate(TASK_TYPES):
        if not by_task[task]:
            continue
        report = json_ready(
            task_summary(
                by_task[task],
                bootstrap=bootstrap,
                seed=seed + 100 * task_index,
                radius=curve_radius,
            )
        )
        report["scope"] = "teacher-forced observer; grouped cuts affect response-query rows"
        path = output / "reports" / task.casefold() / "mechanism_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        save_population_figure(path.parent / "rhythm_summary.png", report)
        if report["mechanism"]["samples"]:
            save_mechanism_figure(path.parent / "mechanism_atlas.png", report["mechanism"])
        reports[task] = report
    return reports
