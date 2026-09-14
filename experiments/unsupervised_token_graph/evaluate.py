"""Join RAGTruth token labels only after prediction freeze."""

import json

import numpy as np


def _auc(y, score):
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order)) + 1
    positives = y == 1
    negatives = y == 0
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) /
                 (positives.sum() * negatives.sum()))


def _average_precision(y, score):
    order = np.argsort(-score, kind="stable")
    ranked = y[order]
    cumulative = np.cumsum(ranked)
    return float((cumulative[ranked == 1] / (np.flatnonzero(ranked == 1) + 1)).sum() / ranked.sum())


def evaluate_predictions(predictions, annotations):
    labels = {}
    with open(annotations, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            labels[str(row["id"])] = row
    y, score = [], []
    for prediction in predictions:
        row = labels[prediction["response_id"]]
        offsets = np.asarray(row["offsets"])
        target = np.zeros(len(offsets), dtype=int)
        for span in row["labels"]:
            target[(offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])] = 1
        values = np.asarray(prediction["score"])[-len(offsets):]
        y.extend(target.tolist())
        score.extend(values.tolist())
    result = {"tokens": len(y), "positives": int(sum(y))}
    if len(set(y)) == 2:
        labels, values = np.asarray(y), np.asarray(score)
        result.update(auroc=_auc(labels, values), ap=_average_precision(labels, values))
    return result
