"""Post-score evaluation and approximation/nonadditivity diagnostics."""

import numpy as np
from state_audit.storage import read_arrays, read_json, write_csv, write_json

from ..comparison_evaluation import compare_metrics
from ..dual_state.report import attach_annotations
from ..evidence_contrast.unit_report import evaluate_units, load_records
from ..readout.report import evaluation_records
from .representation import COMPARISONS, METHODS, TOKEN_METHODS


def intervention_diagnostics(output, settings):
    rows = []
    for index, response in enumerate(settings["responses"]):
        directory = output / "responses" / f"{index:04d}"
        views = read_json(directory / "views.json")
        for unit in views["units"]:
            saved = read_arrays(directory / f"unit_{unit['start']:06d}" / "measurements.npz")
            effects = saved["keep"][:, None, :].astype(float) - saved["single"]
            joint = saved["keep"].astype(float) - saved["joint"]
            approximation = saved["approximation"].T
            row = dict(response_id=response["id"], start=unit["start"], stop=unit["stop"],
                selected_edges=len(saved["edges"]),
                sham_overlap=int(np.all(saved["edges"] == saved["sham_edges"], axis=1).sum()),
                reconstruction_max_error=float(saved["reconstruction_error"].max()),
                unit_mean_joint_with_source=float(joint[0].mean()),
                unit_mean_joint_without_source=float(joint[1].mean()),
                joint_minus_sum_single_max=float(np.abs(joint - effects.sum(1)).max()),
                gradient_vs_finite_mean_abs_error=float(np.abs(effects.mean(-1) - approximation).mean())
                    if len(saved["edges"]) else None)
            rows.append(row)
    write_csv(output / "intervention_diagnostics.csv", rows, list(rows[0]))


def evaluate(output, settings):
    intervention_diagnostics(output, settings)
    result = evaluate_units(output, settings, METHODS, TOKEN_METHODS, COMPARISONS)
    if result["status"] != "evaluated":
        return result
    records, predictions = load_records(output, settings, METHODS)
    attach_annotations(records, read_json(output / "annotations.json"))
    for index, record in enumerate(records):
        scores = read_arrays(output / "responses" / f"{index:04d}" / "scores.npz")
        record["valid"] &= scores["selected_count"] > 0
    evaluated = evaluation_records(records, predictions, METHODS)
    write_json(output / "intervened_tokens_evaluation.json", dict(
        subset="tokens_in_units_with_at_least_one_selected_history_edge",
        methods=compare_metrics(evaluated, METHODS), labels_used_for_selection=False))
    return result
