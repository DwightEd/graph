"""Post-score token evaluation, onset/normal-run budgets, and aggregate plots."""

import numpy as np
from state_audit.storage import read_arrays, read_json, write_csv, write_json

from ..comparison_evaluation import compare_metrics, ranking_examples
from ..comparison_deltas import ranking_deltas
from ..dual_state.report import attach_annotations, budget_details, intervals, prediction_rows
from ..readout.report import evaluation_records, top_rank
from .bootstrap import source_bootstrap
from .scoring import CANDIDATES, CONTROLS, METHODS, PRIMARY


def load_records(output, settings):
    records, predictions = [], []
    controls = [name for name in METHODS if name not in CANDIDATES]
    for index, response in enumerate(settings["responses"]):
        saved = read_arrays(output / "responses" / f"{index:04d}" / "scores.npz")
        records.append(dict(id=response["id"], source_id=response["source_id"], response=response,
                            target=saved["target"], response_length=len(saved["target"]),
                            baselines={name: saved[name] for name in controls}))
        predictions.append({name: saved[name] for name in CANDIDATES})
    return records, predictions


def budget_summary(records, spans, normals, answers):
    def fraction(numerator, denominator):
        return numerator / denominator if denominator else None
    all_normal = {r["id"]: int(r["valid"].sum()) for r in records
                  if not (r["labels"][r["valid"]] == 1).any()}
    result = {}
    for method in METHODS:
        selected = [row for row in spans if row["method"] == method]
        runs = [row for row in normals if row["method"] == method]
        clean = [row for row in answers if row["method"] == method and row["response_id"] in all_normal]
        first = {}
        for row in selected:
            first.setdefault(row["response_id"], row)
        result[method] = dict(span_count=len(selected), error_answer_count=len(first),
            span_onset_recall=fraction(sum(r["onset_alarm"] for r in selected), len(selected)),
            answer_first_error_recall=fraction(sum(r["onset_alarm"] for r in first.values()), len(first)),
            normal_run_count=len(runs), normal_run_alarm_rate=fraction(sum(r["any_alarm"] for r in runs), len(runs)),
            normal_token_fpr=fraction(sum(r["alarm_tokens"] for r in runs), sum(r["length"] for r in runs)),
            all_normal_answer_count=len(clean),
            all_normal_answer_alarm_rate=fraction(sum(r["selected"] > 0 for r in clean), len(clean)),
            all_normal_token_fpr=fraction(sum(r["selected"] for r in clean), sum(all_normal.values())))
    return result


def write_tables(output, records, predictions, evaluated, result):
    rows = prediction_rows(records, predictions)
    write_csv(output / "predictions.csv", rows, list(rows[0]))
    onsets, high_normals = ranking_examples(evaluated, rows, METHODS)
    spans, runs, answers = budget_details(records, predictions, METHODS)
    for name, items in (("onsets", onsets), ("high_risk_normals", high_normals),
                        ("span_budget", spans), ("normal_run_budget", runs), ("budget_by_answer", answers)):
        write_csv(output / f"{name}.csv", items, list(items[0]) if items else ["method"])
    metrics = [dict(method=method, phase=phase, tokens=value["tokens"], positives=value["positives"],
                    auroc=value["auroc"], ap=value["ap"], source_balanced_ap=value["source_balanced"]["ap"],
                    within_answer_auroc=value["within_answer"]["pair_weighted_auroc"])
               for method, phases in result["methods"].items() for phase, value in phases.items()]
    write_csv(output / "metrics.csv", metrics, list(metrics[0]))
    write_json(output / "detection_budget.json", dict(budget_fraction=.1,
        tie_policy="stable_original_answer_and_token_order", deployment_threshold=False,
        methods=budget_summary(records, spans, runs, answers)))
    deltas = ranking_deltas(result, [(PRIMARY, control) for control in CONTROLS])
    write_csv(output / "comparisons.csv", deltas, list(deltas[0]))


def plot_results(output, records, predictions, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for index, (record, predicted) in enumerate(zip(records, predictions)):
        figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        groups = [(predicted, ("source_full", "source_local", PRIMARY, "source_pair_span"), "Risk score (nats/token)"),
                  (record["baselines"], CONTROLS, "Route score")]
        for axis, (values, names, label) in zip(axes, groups):
            for start, end in intervals(record, True):
                axis.axvspan(start - .5, end - .5, color="tomato", alpha=.18)
            for name in names:
                axis.plot(record["target"], values[name], label=name, linewidth=.9)
            axis.set_ylabel(label)
            axis.legend(fontsize=7)
        axes[0].set_title(f"{record['id']} | shaded: annotated error | scores are not probabilities")
        axes[1].set_xlabel("Original answer token")
        figure.tight_layout()
        figure.savefig(output / "responses" / f"{index:04d}" / "trajectory.png", dpi=130)
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric in zip(axes, ("auroc", "ap")):
        values = [result["methods"][name]["all_error"][metric] for name in METHODS]
        axis.barh(METHODS, [np.nan if value is None else value for value in values])
        axis.set_xlabel(metric.upper())
    figure.tight_layout()
    figure.savefig(output / "summary.png", dpi=140)
    plt.close(figure)


def evaluate(output, settings, annotations, bootstrap_repeats):
    if not annotations.is_file():
        result = dict(status="unavailable", reason="missing_token_annotations")
        write_json(output / "evaluation.json", result)
        return result
    records, predictions = load_records(output, settings)
    attach_annotations(records, read_json(annotations))
    evaluated = evaluation_records(records, predictions, METHODS)
    result = dict(status="evaluated", primary_candidate=PRIMARY, methods=compare_metrics(evaluated, METHODS),
        by_answer={r["id"]: compare_metrics([r], METHODS) for r in evaluated},
        top_decile=top_rank(evaluated, METHODS), labels_used_for_scoring=False, threshold_calibrated=False,
        cohort=settings.get("cohort", {}), automatic_model_selection=False)
    write_json(output / "evaluation.json", result)
    write_tables(output, records, predictions, evaluated, result)
    write_json(output / "source_bootstrap.json", source_bootstrap(evaluated, bootstrap_repeats))
    plot_results(output, records, predictions, result)
    return result
