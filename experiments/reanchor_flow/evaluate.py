"""Stream captured artifacts into the re-anchor phenomenon reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from research_dataset import open_research_dataset
from experiments.common.ragtruth_alignment import TASK_TYPES, canonical_task_type

from .artifacts import load_result
from .capture import CAPTURE_SCHEMA
from .hypotheses import task_report
from .signals import compact_row
from .visualize import save_population_figure


def json_ready(value):
    """Convert NumPy objects and non-finite floats to strict JSON values."""

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


def evaluate_results(
    output_root: str | Path,
    dataset_root: str | Path,
    *,
    bootstrap: int = 1000,
    seed: int = 2026,
    pre: int = 5,
    post: int = 3,
    curve_low: int = -5,
    curve_high: int = 10,
) -> dict[str, dict]:
    output = Path(output_root)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("analysis_complete"):
        raise ValueError("analysis is incomplete")
    if manifest.get("config", {}).get("capture_schema") != CAPTURE_SCHEMA:
        raise ValueError(
            "output uses a stale audit schema; rerun analyze in a new --output directory"
        )
    config = manifest["config"]
    if str(Path(dataset_root).resolve()) != str(Path(config["dataset_root"]).resolve()):
        raise ValueError("evaluation dataset differs from the capture manifest")
    sample_ids = [str(entry["sample_id"]) for entry in manifest["samples"]]
    result_paths = [str(entry["result"]) for entry in manifest["samples"]]
    if len(sample_ids) != len(set(sample_ids)) or len(result_paths) != len(set(result_paths)):
        raise ValueError("run manifest contains duplicate samples or result paths")

    dataset = open_research_dataset(
        dataset_root, device="cpu", retain_embedded_labels=True
    )
    label_store = dataset.prepare_evaluation_labels([])
    by_task: dict[str, list[dict]] = {task: [] for task in TASK_TYPES}
    for entry in manifest["samples"]:
        result = load_result(output / entry["result"])
        sample = dataset[str(entry["sample_id"])]
        try:
            row = compact_row(entry, result, sample, label_store, config)
            by_task[canonical_task_type(sample.task_type)].append(row)
        finally:
            sample.release_attention()
        del result

    reports = {}
    for number, task in enumerate(TASK_TYPES):
        if not by_task[task]:
            continue
        report = task_report(
            task,
            by_task[task],
            bootstrap=bootstrap,
            seed=seed + 100 * number,
            pre=pre,
            post=post,
            curve_low=curve_low,
            curve_high=curve_high,
        )
        report = json_ready(report)
        report_path = output / "reports" / task.casefold() / "phenomenon_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        save_population_figure(
            report_path.parent / "phenomenon_audit.png", report["event_curves"]
        )
        reports[task] = report
    return reports
