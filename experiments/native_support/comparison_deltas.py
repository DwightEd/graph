"""Matched ranking deltas and recovery ranks; never tune the scoring stage."""

import numpy as np
from state_audit.storage import write_csv, write_json

from .evaluate import evaluation_records


def difference(left, right):
    return left - right if left is not None and right is not None else None


def ranking_deltas(evaluation, pairs):
    rows = []
    for candidate, control in pairs:
        for phase, result in evaluation["methods"][candidate].items():
            reference = evaluation["methods"][control][phase]
            rows.append({"candidate": candidate, "control": control, "phase": phase,
                         "tokens": result["tokens"], "positives": result["positives"],
                         "delta_auroc": difference(result["auroc"], reference["auroc"]),
                         "delta_ap": difference(result["ap"], reference["ap"]),
                         "delta_source_auroc": difference(result["source_balanced"]["auroc"], reference["source_balanced"]["auroc"]),
                         "delta_within_auroc": difference(result["within_answer"]["pair_weighted_auroc"], reference["within_answer"]["pair_weighted_auroc"])})
    return rows


def recovery_rows(records, rows, methods):
    lookup = {(r["response_id"], int(r["target"])): r for r in rows}
    result = []
    for record in records:
        normal = record["labels"] == 0
        adjacent = np.r_[False, np.diff(record["target"]) == 1]
        after_error = normal & np.r_[False, record["labels"][:-1] == 1] & adjacent
        for index in np.flatnonzero(after_error):
            source = lookup[record["id"], int(record["target"][index])]
            for column, method in enumerate(methods):
                scores = record["scores"][:, column]
                reference = scores[normal]
                percentile = np.mean((reference < scores[index]) + 0.5 * (reference == scores[index]))
                result.append({"method": method, "within_answer_normal_percentile": float(percentile),
                               "score": float(scores[index]), **source})
    return result


def write_deltas(output, destination, annotations, evaluation, methods, rows, pairs):
    if evaluation["status"] != "evaluated":
        return {"status": "unavailable"}
    deltas = ranking_deltas(evaluation, pairs)
    write_csv(destination / "comparisons.csv", deltas, list(deltas[0]))
    result = {"status": "evaluated", "automatic_method_selection": False, "comparisons": deltas}
    write_json(destination / "comparisons.json", result)
    records = evaluation_records(output, annotations, destination, tuple(methods.values()))
    recoveries = recovery_rows(records, rows, methods)
    fields = list(recoveries[0]) if recoveries else ["method", "response_id", "target"]
    write_csv(destination / "recovery.csv", recoveries, fields)
    return result
