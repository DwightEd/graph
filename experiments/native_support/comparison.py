"""Reuse native NPZ once, freeze routing controls, then open evaluation labels."""

import csv

from state_audit.storage import (
    read_arrays,
    read_json,
    write_arrays,
    write_csv,
    write_json,
)
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .calibration import PROTOCOLS, crossfit_collapse
from .comparison_evaluation import evaluate_comparison
from .pipeline import score_response
from .report import write_comparison_report
from .routes import ROUTE_SCORES
from .source_regions import compile_regions, region_mask

DIRECTORY = "route_comparison_v2"
METHODS = {name: name for name in (*ROUTE_SCORES, *PROTOCOLS, "entropy", "surprisal", "negative_margin")}
METHODS.update(support_graph="risk", direct_prompt="direct_risk")


def measure_responses(output, destination, settings, regions):
    rows, records = [], []
    for index, response in enumerate(tqdm(settings["responses"], desc="cached route comparison")):
        source = output / "responses" / f"{index:04d}"
        target = destination / "responses" / f"{index:04d}"
        rows.extend(score_response(response, source, destination=target, with_routes=True,
                                   evidence_mask=region_mask(regions, response)))
        arrays = read_arrays(target / "scores.npz")
        record = {"id": response["id"], "source_id": response["source_id"],
                  "prompt_length": response["prompt_length"], "tokens": len(arrays["target"])}
        record.update({field: arrays[field] for field, _ in PROTOCOLS.values()})
        records.append(record)
    return rows, records


def add_calibrated_scores(destination, records, rows):
    calibrated, protocol = crossfit_collapse(records)
    write_json(destination / "calibration.json", protocol)
    for row in rows:
        scores = calibrated[row["response_id"]]
        row.update({name: values[row["target"]] for name, values in scores.items()})
    for index, record in enumerate(records):
        directory = destination / "responses" / f"{index:04d}"
        arrays = read_arrays(directory / "scores.npz")
        arrays.update(calibrated[record["id"]])
        write_arrays(directory / "scores.npz", **arrays)
        selected = [row for row in rows if row["response_id"] == record["id"]]
        write_csv(directory / "tokens.csv", selected, list(selected[0]))
    return protocol


def evaluation_headlines(evaluation):
    if evaluation["status"] != "evaluated":
        return evaluation
    phases = ("all_error", "span_onset_vs_normal", "first_error_vs_normal", "continuation_vs_normal")
    methods = {}
    for method, metrics in evaluation["methods"].items():
        methods[method] = {}
        for phase in phases:
            keys = ("tokens", "positives", "auroc", "ap", "unscored_tokens")
            methods[method][phase] = {key: metrics[phase][key] for key in keys}
    return {
        "status": "evaluated", "threshold_calibrated": False,
        "methods": methods,
    }


def comparison_summary(settings, rows, regions, calibration, evaluation):
    primary = "routing_imbalance" if regions["status"] == "available" else "prompt_routing_imbalance"
    return {
        "purpose": "label_free_native_route_comparison_v2", "primary_method": primary,
        "responses": len(settings["responses"]), "scored_tokens": len(rows),
        "labels_used": False, "gradients": False, "interventions": False,
        "model_forward_during_comparison": False,
        "source_regions": regions["status"], "calibration": calibration["status"],
        "max_abs_ledger_error": max(abs(row["ledger_error"]) for row in rows),
        "cohort": settings.get("cohort", {"selection": "input_manifest"}),
        "historical_comparability": "same formulas where available; new prompts/population are not historical AUROC reproduction",
        "support_graph_role": "retained_v1_control_not_primary",
        "focus_scope": "source_blocks" if regions["status"] == "available" else "ordinary_prompt",
        "method_protocol": {
            "routing_imbalance": "079c33a: mean-layer (history-source) projected-message norm / all-source norm",
            "attention_displacement": "a2a40fd: mean layer/head (history-source) attention",
            "prompt_variants": "ordinary prompt/history numerator; original all-source denominator; not exact historical evidence scores",
            "collapse": "f7344e2 carrier geometry; ordinary prompt; source-disjoint causal position calibration",
            "collapse_offline": "historical full prompt and final-answer-length calibration; offline control only",
            "direction": "fixed higher-is-risk; no label-selected sign or coefficients",
        },
        "evaluation": evaluation_headlines(evaluation),
        "evaluation_performed_by_this_stage": evaluation["status"] == "evaluated",
    }


def run_comparison(output, settings, annotations=None):
    destination = output / DIRECTORY
    regions = compile_regions(settings, destination)
    with threadpool_limits(limits=1):
        rows, records = measure_responses(output, destination, settings, regions)
        calibration = add_calibrated_scores(destination, records, rows)
    write_csv(destination / "tokens.csv", rows, list(rows[0]))
    annotations = annotations or output / "annotations.json"
    evaluation = evaluate_comparison(output, destination, annotations, METHODS, rows)
    summary = comparison_summary(settings, rows, regions, calibration, evaluation)
    write_comparison_report(destination / "report.html", rows, summary["primary_method"], evaluation)
    write_json(destination / "summary.json", summary)
    return summary


def evaluate_existing(output, annotations=None):
    destination = output / DIRECTORY
    with (destination / "tokens.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["target"] = int(row["target"])
        for name in (*ROUTE_SCORES, "prompt_read", "focus_positive_write", "focus_negative_write",
                     "ffn_negative", "entropy", "risk"):
            row[name] = float(row[name])
    evaluation = evaluate_comparison(output, destination, annotations or output / "annotations.json", METHODS, rows)
    summary = read_json(destination / "summary.json")
    summary["evaluation"] = evaluation_headlines(evaluation)
    summary["evaluation_performed_by_this_stage"] = evaluation["status"] == "evaluated"
    write_comparison_report(destination / "report.html", rows, summary["primary_method"], evaluation)
    write_json(destination / "summary.json", summary)
    return summary["evaluation"]
