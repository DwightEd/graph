"""Post-hoc claim-level evaluation of frozen re-anchor flow graphs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_dataset import open_research_dataset
from experiments.constraint_routing_rhythm.data import TASK_TYPES, canonical_task_type

from .artifacts import load_result
from .metrics import metric, paired_bootstrap, paired_effect
from .visualize import save_population_figure

PRIMARY = "functional_global_reanchor"
CONTROLS = {
    "attention_global_reanchor": "attention_evidence_reanchor_flow",
    "middle_functional_reanchor": "middle_evidence_reanchor_flow",
    "rewired_global_reanchor": "rewired_evidence_reanchor_flow",
    "direct_evidence_sink": "functional_direct_evidence_sink",
    "bag_evidence_claim": "functional_bag_evidence_claim",
    "boundary_reread": "functional_reread_pulse",
}


def safe_correlation(first, second) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def aligned_curves(rows: list[dict], low: int = -5, high: int = 10) -> dict[str, list]:
    offsets = np.arange(low, high + 1)
    groups = ("correct", "hallucinated", "control")
    channels = ("evidence", "history")
    sums = {(group, channel): np.zeros(len(offsets)) for group in groups for channel in channels}
    counts = {(group, channel): np.zeros(len(offsets), dtype=np.int64) for group in groups for channel in channels}

    def add(row: dict, center: int, group: str) -> None:
        for number, shift in enumerate(offsets):
            event = center - row["response_start"] + int(shift)
            if not 0 <= event < len(row["functional_evidence_inflow"]):
                continue
            for channel, field in (
                ("evidence", "functional_evidence_inflow"),
                ("history", "functional_history_inflow"),
            ):
                value = row[field][event]
                if np.isfinite(value):
                    sums[(group, channel)][number] += value
                    counts[(group, channel)][number] += 1

    for row in rows:
        for start, stop, hallucinated in zip(
            row["claim_start"],
            row["claim_stop"],
            row["claim_label"],
            strict=True,
        ):
            add(row, int(start), "hallucinated" if hallucinated else "correct")
            control = int(start) + max(1, (int(stop) - int(start)) // 2)
            if control < int(stop):
                add(row, control, "control")

    result: dict[str, list] = {"offset": offsets.tolist()}
    for group in groups:
        for channel in channels:
            result[f"{group}_{channel}"] = np.divide(
                sums[(group, channel)],
                counts[(group, channel)],
                out=np.full(len(offsets), np.nan),
                where=counts[(group, channel)] > 0,
            ).tolist()
            result[f"{group}_{channel}_count"] = counts[(group, channel)].tolist()
    return result


def task_report(task: str, rows: list[dict], bootstrap: int, seed: int) -> dict:
    label = np.concatenate([row["claim_label"] for row in rows])
    source = np.concatenate(
        [np.repeat(row["source_id"], len(row["claim_label"])) for row in rows]
    )
    raw = {
        PRIMARY: np.concatenate(
            [row["functional_evidence_reanchor_flow"] for row in rows]
        )
    }
    raw.update(
        {
            name: np.concatenate([row[field] for row in rows])
            for name, field in CONTROLS.items()
        }
    )
    # Lower evidence-seeded flow is the preregistered missed-re-anchor direction.
    score = {name: -value for name, value in raw.items()}

    graph_necessity = {
        f"{PRIMARY}_minus_{name}": paired_bootstrap(
            label,
            score[PRIMARY],
            score[name],
            source,
            repeats=bootstrap,
            seed=seed + number,
        )
        for number, name in enumerate(CONTROLS)
    }

    audit = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in (
            "audit_functional_backbone_delta",
            "audit_attention_backbone_delta",
            "audit_capacity_bag_delta",
            "audit_matched_endpoint_delta",
        )
    }
    path_cut = {
        "functional_vs_attention": paired_effect(
            audit["audit_functional_backbone_delta"],
            audit["audit_attention_backbone_delta"],
        ),
        "functional_vs_capacity_bag": paired_effect(
            audit["audit_functional_backbone_delta"],
            audit["audit_capacity_bag_delta"],
        ),
        "functional_vs_matched_endpoint": paired_effect(
            audit["audit_functional_backbone_delta"],
            audit["audit_matched_endpoint_delta"],
        ),
    }

    position = np.concatenate(
        [
            np.linspace(0.0, 1.0, len(row["functional_evidence_inflow"]))
            for row in rows
        ]
    )
    evidence = np.concatenate([row["functional_evidence_inflow"] for row in rows])
    history = np.concatenate([row["functional_history_inflow"] for row in rows])
    return {
        "task": task,
        "samples": len(rows),
        "sources": int(np.unique([row["source_id"] for row in rows]).size),
        "claims": int(len(label)),
        "hallucinated_claims": int(label.sum()),
        "prevalence": float(label.mean()) if len(label) else None,
        "primary": PRIMARY,
        "score_direction": (
            "higher means less evidence-seeded path mass reaches the claim sink "
            "through the first claim tokens"
        ),
        "metrics": {name: metric(label, value) for name, value in score.items()},
        "graph_necessity": graph_necessity,
        "path_cut_validation": path_cut,
        "normal_autoregressive_drift": {
            "evidence_inflow_vs_relative_position": safe_correlation(position, evidence),
            "history_inflow_vs_relative_position": safe_correlation(position, history),
        },
        "claim_aligned_curves": aligned_curves(rows),
        "interpretation_gate": (
            "Graph structure is supported only if the full functional flow beats "
            "attention, direct, bag, and role/lag-rewired controls on identical "
            "claims, and its selected backbone causes a larger absolute margin "
            "change than matched endpoint and capacity-bag cuts."
        ),
        "labels_used_during": "post-hoc evaluation only",
    }


def evaluate_results(
    output_root: str | Path,
    dataset_root: str | Path,
    *,
    bootstrap: int = 400,
    seed: int = 2026,
) -> dict[str, dict]:
    output = Path(output_root)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("analysis_complete"):
        raise ValueError("analysis is incomplete")
    frozen = [
        (entry, load_result(output / entry["result"]))
        for entry in manifest["samples"]
    ]

    dataset = open_research_dataset(
        dataset_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    label_store = dataset.prepare_evaluation_labels([])
    by_task: dict[str, list[dict]] = {task: [] for task in TASK_TYPES}
    for entry, result in frozen:
        sample_id = str(entry["sample_id"])
        sample = dataset[sample_id]
        try:
            cached = sample.attention()
            labels = np.asarray(label_store.response_labels(sample), dtype=bool)
            response_start = int(cached.response_idx)
            prediction = np.asarray(result["prediction_position"], dtype=np.int64)
            target = np.asarray(result["target_token_id"], dtype=np.int64)
            token_ids = np.asarray(cached.token_ids.detach().cpu(), dtype=np.int64)
            expected = response_start + np.arange(len(prediction))
            if not np.array_equal(prediction, expected):
                raise ValueError(f"prediction coordinates changed: {sample_id}")
            if not np.array_equal(target, token_ids[prediction]):
                raise ValueError(f"target tokens changed: {sample_id}")
            claim_start = np.asarray(result["claim_start"], dtype=np.int64)
            claim_stop = np.asarray(result["claim_stop"], dtype=np.int64)
            claim_label = np.asarray(
                [
                    labels[start - response_start : stop - response_start].any()
                    for start, stop in zip(claim_start, claim_stop, strict=True)
                ],
                dtype=bool,
            )
            row = {name: value for name, value in result.items()}
            row.update(
                sample_id=sample_id,
                source_id=str(sample.source_id),
                response_start=response_start,
                claim_start=claim_start,
                claim_stop=claim_stop,
                claim_label=claim_label,
            )
            by_task[canonical_task_type(sample.task_type)].append(row)
        finally:
            sample.release_attention()

    reports = {}
    for number, task in enumerate(TASK_TYPES):
        if not by_task[task]:
            continue
        report = task_report(task, by_task[task], bootstrap, seed + 100 * number)
        report_path = output / "reports" / task.casefold() / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        save_population_figure(
            report_path.parent / "claim_aligned_reanchor.png",
            report["claim_aligned_curves"],
        )
        reports[task] = report
    return reports
