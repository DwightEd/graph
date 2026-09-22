"""Evaluate frozen methods on common tokens; expose within-answer and weighting effects."""

from collections import Counter

import numpy as np
from state_audit.storage import write_csv, write_json

from .evaluate import evaluation_records, ranking


def phase_masks(labels, onsets, firsts):
    normal = labels == 0
    continuation = labels.astype(bool) & ~onsets
    return {
        "all_error": (np.ones(len(labels), dtype=bool), labels),
        "span_onset_vs_normal": (normal | onsets, onsets),
        "first_error_vs_normal": (normal | firsts, firsts),
        "continuation_vs_normal": (normal | continuation, continuation),
    }


def within_answer(labels, scores, answer_ids):
    values, pairs = [], []
    for identity in np.unique(answer_ids):
        selected = answer_ids == identity
        measured = ranking(labels[selected], scores[selected])
        if measured["auroc"] is not None:
            values.append(measured["auroc"])
            pairs.append(measured["positives"] * measured["negatives"])
    return {
        "mixed_answers": len(values), "positive_negative_pairs": int(sum(pairs)),
        "macro_auroc": float(np.mean(values)) if values else None,
        "pair_weighted_auroc": float(np.average(values, weights=pairs)) if values else None,
    }


def group_metrics(labels, scores, answers, sources):
    finite = np.isfinite(scores)
    labels, scores, answers, sources = (x[finite] for x in (labels, scores, answers, sources))
    counts = Counter(sources)
    weights = np.asarray([1 / counts[source] for source in sources])
    return {
        **ranking(labels, scores), "unscored_tokens": int((~finite).sum()),
        "source_balanced": ranking(labels, scores, weights),
        "within_answer": within_answer(labels, scores, answers),
    }


def compare_metrics(records, methods):
    joined = {name: np.concatenate([r[name] for r in records]) for name in ("labels", "onsets", "firsts", "scores")}
    answers = np.concatenate([np.repeat(r["id"], len(r["labels"])) for r in records])
    sources = np.concatenate([np.repeat(r["source_id"], len(r["labels"])) for r in records])
    front = np.concatenate([r["target"] < r["response_length"] / 2 for r in records])
    groups = phase_masks(joined["labels"], joined["onsets"], joined["firsts"])
    groups.update(front_half=(front, joined["labels"]), back_half=(~front, joined["labels"]))
    result = {}
    for column, method in enumerate(methods):
        result[method] = {
            phase: group_metrics(target[mask], joined["scores"][mask, column], answers[mask], sources[mask])
            for phase, (mask, target) in groups.items()
        }
    return result


def ranking_examples(records, rows, methods):
    """No threshold: these are high-ranked normal tokens, not claimed false positives."""
    lookup = {(row["response_id"], row["target"]): row for row in rows}
    onsets, normals = [], []
    for record in records:
        for column, method in enumerate(methods):
            score = record["scores"][:, column]
            ordinary = np.flatnonzero((record["labels"] == 0) & np.isfinite(score))
            highest = ordinary[np.argsort(-score[ordinary], kind="stable")[:10]]
            starts = record["onsets"] | record["firsts"]
            selected = np.flatnonzero(starts & np.isfinite(score))
            for index in np.union1d(selected, highest):
                source = lookup[record["id"], int(record["target"][index])]
                reference = score[ordinary]
                percentile = np.mean((reference < score[index]) + 0.5 * (reference == score[index])) if len(reference) else None
                item = {"method": method, "label": int(record["labels"][index]),
                        "is_span_onset": bool(record["onsets"][index]),
                        "is_answer_first_error": bool(record["firsts"][index]), "score": float(score[index]),
                        "within_answer_normal_percentile": percentile, **source}
                (onsets if starts[index] else normals).append(item)
    return onsets, normals


def evaluate_comparison(output, destination, annotations, methods, rows):
    if not annotations.is_file():
        result = {"status": "unavailable", "reason": "missing_token_annotations", "annotations_path": str(annotations)}
        write_json(destination / "evaluation_status.json", result)
        return result
    records = evaluation_records(output, annotations, destination, tuple(methods.values()))
    result = {
        "status": "evaluated", "labels_used_for_scoring": False, "threshold_calibrated": False,
        "methods": compare_metrics(records, methods),
        "by_answer": {r["id"]: compare_metrics([r], methods) for r in records},
        "weighting": "pooled token metrics plus equal-source and within-answer comparisons",
        "annotations_path": str(annotations),
    }
    write_json(destination / "evaluation.json", result)
    write_json(destination / "evaluation_status.json", {"status": "evaluated", "file": "evaluation.json"})
    onsets, normals = ranking_examples(records, rows, methods)
    fields = list((onsets or normals)[0]) if onsets or normals else ["method", "response_id", "target"]
    write_csv(destination / "onsets.csv", onsets, fields)
    write_csv(destination / "high_risk_normals.csv", normals, fields)
    return result
