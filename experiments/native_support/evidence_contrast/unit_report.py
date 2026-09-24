"""Separate token ranking, text-unit discrimination, and within-unit localization."""

import numpy as np
from state_audit.storage import read_arrays, read_json, write_csv, write_json

from ..comparison_deltas import ranking_deltas
from ..comparison_evaluation import compare_metrics, group_metrics, within_answer
from ..dual_state.report import attach_annotations, prediction_rows
from ..readout.report import evaluation_records
from .aggregation import COMPARISONS, METHODS, TOKEN_METHODS
from .unit_budget import unit_budgets


def load_records(output, settings):
    records, predictions = [], []
    for index, response in enumerate(settings["responses"]):
        directory = output / "responses" / f"{index:04d}"
        saved = read_arrays(directory / "scores.npz")
        units = read_json(directory / "views.json")["units"]
        records.append(dict(id=response["id"], source_id=response["source_id"], response=response,
            target=saved["target"], response_length=len(saved["target"]), units=units, baselines={}))
        predictions.append({name: saved[name] for name in METHODS})
    return records, predictions


def unit_rows(records, predictions):
    rows = []
    for record, scores in zip(records, predictions):
        response = record["response"]
        pieces = response["token_text"][response["prompt_length"]:]
        for index, unit in enumerate(record["units"]):
            start, stop = unit["start"], unit["stop"]
            valid = record["valid"][start:stop]
            errors = int(record["labels"][start:stop][valid].sum())
            row = dict(response_id=record["id"], source_id=record["source_id"], unit_id=index,
                start=start, stop=stop, length=stop-start, valid_tokens=int(valid.sum()),
                fully_annotated=bool(valid.all()), error_tokens=errors,
                error_fraction=errors / valid.sum() if valid.any() else None,
                text="".join(pieces[start:stop]))
            row.update({name: float(scores[f"{name}_unit_mean"][start]) for name in TOKEN_METHODS})
            rows.append(row)
    return rows


def unit_metrics(rows):
    selected = [row for row in rows if row["fully_annotated"]]
    labels = np.asarray([row["error_tokens"] > 0 for row in selected], dtype=int)
    answers = np.asarray([row["response_id"] for row in selected])
    sources = np.asarray([row["source_id"] for row in selected])
    measured = {name: group_metrics(labels, np.asarray([row[name] for row in selected]), answers, sources)
                for name in TOKEN_METHODS}
    return dict(unit_of_analysis="saved_text_unit", positive="contains_any_annotated_error_token",
                excluded_incomplete_units=len(rows)-len(selected),
                note="Shared metric count fields named tokens count text units here", methods=measured)


def localization_metrics(records, predictions):
    """Only compare error/normal token pairs in the very same saved text unit."""
    labels, groups = [], []
    scores = {name: [] for name in METHODS}
    for answer, (record, predicted) in enumerate(zip(records, predictions)):
        for index, unit in enumerate(record["units"]):
            target = np.arange(unit["start"], unit["stop"])
            target = target[record["valid"][target]]
            labels.extend(record["labels"][target])
            groups.extend([f"{answer}:{index}"] * len(target))
            for name in METHODS:
                scores[name].extend(predicted[name][target])
    measured = {name: within_answer(np.asarray(labels), np.asarray(values), np.asarray(groups))
                for name, values in scores.items()}
    for value in measured.values():
        value["mixed_units"] = value.pop("mixed_answers")
    status = "evaluated" if measured[TOKEN_METHODS[0]]["mixed_units"] else "unavailable_no_mixed_units"
    return dict(status=status, comparison="error_vs_normal_in_same_text_unit", methods=measured,
                note="Constant unit scores have AUROC 0.5; no within-unit localization")


def annotation_alignment(rows):
    complete = [row for row in rows if row["fully_annotated"]]
    mixed = sum(0 < row["error_tokens"] < row["length"] for row in complete)
    return dict(units=len(rows), fully_annotated_units=len(complete),
        all_normal_units=sum(row["error_tokens"] == 0 for row in complete),
        all_error_units=sum(row["error_tokens"] == row["length"] for row in complete),
        mixed_units=mixed, within_unit_localization_identifiable=mixed > 0,
        labels_used_for_partition=False,
        interpretation="No mixed units means this cohort cannot test localization inside a unit")


def plot_comparison(output, evaluation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = TOKEN_METHODS
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, metric in zip(axes, ("auroc", "ap")):
        for offset, suffix, label in ((-.2, "", "Individual token"), (.2, "_unit_mean", "Same text-unit mean")):
            values = [evaluation["methods"][name + suffix]["all_error"][metric] for name in names]
            axis.barh(np.arange(len(names)) + offset, [np.nan if v is None else v for v in values],
                      height=.36, label=label)
        axis.set(yticks=np.arange(len(names)), yticklabels=names, xlabel=metric.upper(), xlim=(0, 1))
        axis.legend(fontsize=8)
    figure.suptitle("Same tokens and text units | exploratory cached evaluation")
    figure.tight_layout()
    figure.savefig(output / "aggregation.png", dpi=150)
    plt.close(figure)


def evaluate_units(output, settings):
    if not (output / "annotations.json").is_file():
        result = dict(status="unavailable", reason="missing_token_annotations")
        write_json(output / "evaluation.json", result)
        return result
    records, predictions = load_records(output, settings)
    attach_annotations(records, read_json(output / "annotations.json"))
    evaluated = evaluation_records(records, predictions, METHODS)
    result = dict(status="evaluated", methods=compare_metrics(evaluated, METHODS),
        by_answer={record["id"]: compare_metrics([record], METHODS) for record in evaluated},
        cohort=settings.get("cohort", {}), labels_used_for_scoring=False, automatic_model_selection=False)
    write_json(output / "evaluation.json", result)
    rows = unit_rows(records, predictions)
    budget, delays, alarms = unit_budgets(records, rows)
    metrics = [dict(method=name, phase=phase, **{key: value[key] for key in ("tokens", "positives", "auroc", "ap")})
               for name, phases in result["methods"].items() for phase, value in phases.items()]
    tables = dict(units=rows, predictions=prediction_rows(records, predictions), metrics=metrics,
                  comparisons=ranking_deltas(result, COMPARISONS), span_availability=delays, unit_alarms=alarms)
    for name, table in tables.items():
        write_csv(output / f"{name}.csv", table, list(table[0]) if table else ["method"])
    write_json(output / "unit_evaluation.json", unit_metrics(rows))
    write_json(output / "within_unit.json", localization_metrics(records, predictions))
    write_json(output / "annotation_alignment.json", annotation_alignment(rows))
    write_json(output / "unit_budget.json", budget)
    plot_comparison(output, result)
    return result
