"""Read annotations only after scoring; expose onset, continuation and normal-run costs."""

import numpy as np
from state_audit.storage import write_csv, write_json

from ..comparison_evaluation import compare_metrics, ranking_examples
from ..evaluate import annotation_targets
from ..readout.report import evaluation_records, top_rank


def attach_annotations(records, annotations):
    for record in records:
        annotation = annotations[record["id"]]
        response = record["response"]
        tokens = response["token_ids"][response["prompt_length"]:]
        if annotation["source_id"] != record["source_id"] or not np.array_equal(annotation["token_ids"], tokens):
            raise ValueError(f"{record['id']}: annotation identity differs")
        labels, onsets, firsts, valid = annotation_targets(annotation, len(tokens), record["id"])
        record.update(labels=labels, onsets=onsets, firsts=firsts, valid=valid)


def prediction_rows(records, predictions):
    rows = []
    for record, predicted in zip(records, predictions):
        response = record["response"]
        tokens = response["token_text"][response["prompt_length"]:]
        scores = {**record["baselines"], **predicted}
        for target, token in enumerate(tokens):
            row = dict(response_id=record["id"], source_id=record["source_id"], target=target,
                       token=token, label=int(record["labels"][target]), valid=bool(record["valid"][target]))
            row.update({name: float(value[target]) for name, value in scores.items()})
            rows.append(row)
    return rows


def intervals(record, positive):
    """Split at missing labels, changes in labels, and independently annotated onsets."""
    selected = record["valid"] & (record["labels"] == int(positive))
    start = None
    result = []
    for target in range(len(selected) + 1):
        inside = target < len(selected) and selected[target]
        restart = inside and positive and record["onsets"][target]
        if start is not None and (not inside or restart):
            result.append((start, target))
            start = None
        if inside and start is None:
            start = target
    return result


def budget_details(records, predictions, methods):
    """Global top 10% is a ranking budget, not a deployed or label-calibrated threshold."""
    spans, normals, by_answer = [], [], []
    locations = [(answer, target) for answer, record in enumerate(records)
                 for target in np.flatnonzero(record["valid"])]
    for method in methods:
        values = [{**r["baselines"], **p}[method] for r, p in zip(records, predictions)]
        joined = np.asarray([values[answer][target] for answer, target in locations])
        selected = np.argsort(-joined, kind="stable")[:int(np.ceil(len(joined) / 10))]
        alarms = [np.zeros(record["response_length"], dtype=bool) for record in records]
        for index in selected:
            answer, target = locations[index]
            alarms[answer][target] = True
        for record, alarm in zip(records, alarms):
            by_answer.append(dict(method=method, response_id=record["id"],
                                  selected=int(alarm.sum()), positives=int((alarm & (record["labels"] == 1)).sum()),
                                  negatives=int((alarm & (record["labels"] == 0)).sum())))
            for positive, output in ((True, spans), (False, normals)):
                for start, end in intervals(record, positive):
                    hits = np.flatnonzero(alarm[start:end])
                    output.append(dict(method=method, response_id=record["id"], start=start, end=end,
                        length=end-start, alarm_tokens=len(hits), coverage=len(hits)/(end-start),
                        any_alarm=bool(len(hits)), onset_alarm=bool(alarm[start]),
                        first_alarm_delay=int(hits[0]) if len(hits) else None))
    return spans, normals, by_answer


def plot_trajectories(destination, records, predictions, heads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for index, (record, predicted) in enumerate(zip(records, predictions)):
        scores = {**record["baselines"], **predicted}
        groups = [("Observable scalar", ["observable_route", "observable_route_causal_mean", "observable_route_causal_dual"])]
        if heads:
            groups.append(("Head calibrated", ["head_current", "head_causal_persistent", "head_causal_dual"]))
        figure, axes = plt.subplots(len(groups), 1, figsize=(11, 3 * len(groups)), squeeze=False)
        for axis, (title, methods) in zip(axes[:, 0], groups):
            for start, end in intervals(record, True):
                axis.axvspan(start - .5, end - .5, color="tomato", alpha=.18)
            for method in methods:
                axis.plot(record["target"], scores[method], label=method, linewidth=1)
            axis.set(title=f"{record['id']} | {title} | shaded: annotated error", xlabel="Target token", ylabel="Score")
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(destination / "responses" / f"{index:04d}" / "trajectory.png", dpi=140)
        plt.close(figure)


def evaluate(destination, dataset, predictions, protocol):
    records = dataset["records"]
    methods = list(records[0]["baselines"]) + list(predictions[0])
    evaluation_input = evaluation_records(records, predictions, methods)
    result = dict(methods=compare_metrics(evaluation_input, methods),
                  by_answer={r["id"]: compare_metrics([r], methods) for r in evaluation_input},
                  top_decile=top_rank(evaluation_input, methods),
                  scope="unlabelled_scoring; development_cohort; no_confirmatory_claim",
                  threshold_calibrated=False, labels_used_for_scoring=False)
    rows = prediction_rows(records, predictions)
    write_csv(destination / "predictions.csv", rows, list(rows[0]))
    onsets, normals = ranking_examples(evaluation_input, rows, methods)
    spans, runs, budget = budget_details(records, predictions, methods)
    for name, items in (("onsets", onsets), ("high_risk_normals", normals), ("span_budget", spans),
                        ("normal_run_budget", runs), ("budget_by_answer", budget)):
        write_csv(destination / f"{name}.csv", items, list(items[0]) if items else ["method"])
    metrics = [dict(method=method, phase=phase, tokens=value["tokens"], positives=value["positives"],
                    auroc=value["auroc"], ap=value["ap"], within_answer_auroc=value["within_answer"]["pair_weighted_auroc"])
               for method, phases in result["methods"].items() for phase, value in phases.items()]
    write_csv(destination / "metrics.csv", metrics, list(metrics[0]))
    write_json(destination / "evaluation.json", result)
    write_json(destination / "summary.json", dict(protocol=protocol, evaluation=result))
    plot_trajectories(destination, records, predictions, bool(dataset["schema"]))
    return result
