"""Comparable held-out token metrics and compact supervised review exports."""

from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from state_audit.storage import write_arrays, write_csv, write_json

from ..comparison_evaluation import compare_metrics


def evaluation_records(records, predictions, methods, common=False):
    result = []
    for record, predicted in zip(records, predictions):
        scores = {**record["baselines"], **predicted}
        matrix = np.column_stack([scores[name] for name in methods])
        valid = record["valid"].copy()
        if common:
            valid &= np.isfinite(matrix).all(1)
        result.append({**{key: record[key] for key in ("id", "source_id", "response_length")},
                       **{key: record[key][valid] for key in ("labels", "onsets", "firsts", "target")},
                       "scores": matrix[valid]})
    return result


def top_rank(records, methods):
    labels = np.concatenate([r["labels"] for r in records])
    scores = np.concatenate([r["scores"] for r in records])
    result = {}
    for column, method in enumerate(methods):
        finite = np.flatnonzero(np.isfinite(scores[:, column]))
        count = int(np.ceil(len(finite) / 10))
        highest = finite[np.argsort(-scores[finite, column], kind="stable")[:count]]
        positives = int(labels[highest].sum())
        result[method] = dict(count=count, positives=positives,
                             precision=positives / count if count else None,
                             recall=positives / int(labels.sum()) if labels.sum() else None)
    return result


def save_predictions(destination, records, predictions):
    rows = []
    for index, (record, predicted) in enumerate(zip(records, predictions)):
        response = record["response"]
        tokens = response["token_text"][response["prompt_length"]:]
        scores = {**record["baselines"], **predicted}
        write_arrays(destination / "responses" / f"{index:04d}" / "scores.npz",
                     target=record["target"], token_id=np.asarray(response["token_ids"][response["prompt_length"]:]),
                     labels=record["labels"], valid=record["valid"], **scores)
        for target, token in enumerate(tokens):
            row = dict(response_id=record["id"], source_id=record["source_id"], target=target,
                       token=token, label=int(record["labels"][target]), valid=bool(record["valid"][target]))
            row.update({name: float(value[target]) for name, value in scores.items()})
            rows.append(row)
    write_csv(destination / "predictions.csv", rows, list(rows[0]))


def evaluate(destination, dataset, predictions, folds, protocol):
    methods = list(dataset["records"][0]["baselines"]) + list(predictions[0])
    records = evaluation_records(dataset["records"], predictions, methods)
    common = evaluation_records(dataset["records"], predictions, methods, common=True)
    evaluation = dict(methods=compare_metrics(records, methods), common_tokens=compare_metrics(common, methods),
                      by_answer={r["id"]: compare_metrics([r], methods) for r in records},
                      top_decile=top_rank(common, methods), top_decile_scope="common_scored_valid_tokens",
                      scope="source_disjoint_supervised_diagnostic; not_fresh_confirmatory_test")
    write_json(destination / "evaluation.json", evaluation)
    write_json(destination / "folds.json", folds)
    write_json(destination / "feature_schema.json", dataset["schema"])
    save_predictions(destination, dataset["records"], predictions)
    write_json(destination / "summary.json", {**protocol, "evaluation": evaluation})
    rows = []
    for method, phases in evaluation["common_tokens"].items():
        for phase, metrics in phases.items():
            rows.append(dict(method=method, phase=phase, tokens=metrics["tokens"], positives=metrics["positives"],
                             auroc=metrics["auroc"], ap=metrics["ap"],
                             within_answer_auroc=metrics["within_answer"]["pair_weighted_auroc"]))
    write_csv(destination / "metrics.csv", rows, list(rows[0]))
    return evaluation


def pack(destination):
    archive = destination.with_name(destination.name + "_review_light.zip")
    temporary = archive.with_suffix(".partial.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as output:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and "models" not in path.relative_to(destination).parts:
                output.write(path, path.relative_to(destination))
    temporary.replace(archive)
    return str(archive)
