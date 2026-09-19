"""Stratified diagnostics for completed grounding-dynamics token scores."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SCORES = (
    "raw_surprise",
    "head_contrast_surprise",
    "source_gain",
    "history_gain",
    "grounding_balance",
    "self_jump",
)


def scope_masks(table, high_jump):
    previous = table.previous_gold.to_numpy()
    sentence = table.sentence_start.to_numpy(bool)
    jump = table.self_jump.to_numpy()
    return {
        "all": np.ones(len(table), dtype=bool),
        "previous_gold_0": previous == 0,
        "previous_gold_0_sentence_start": (previous == 0) & sentence,
        "previous_gold_0_high_transition": (
            (previous == 0) & (jump >= high_jump)
        ),
        "previous_gold_1": previous == 1,
    }


def metric(labels, scores):
    finite = np.isfinite(scores)
    labels = np.asarray(labels)[finite]
    scores = np.asarray(scores)[finite]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return np.nan, np.nan, len(labels), int(labels.sum()) if len(labels) else 0
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
        len(labels),
        int(labels.sum()),
    )


def grouped_metrics(table, high_jump, group):
    rows = []
    for name, frame in table.groupby(group, sort=True):
        for scope, mask in scope_masks(frame, high_jump).items():
            for score in SCORES:
                auroc, ap, tokens, positives = metric(
                    frame.gold.to_numpy()[mask],
                    frame[score].to_numpy()[mask],
                )
                rows.append({
                    group: name,
                    "scope": scope,
                    "score": score,
                    "tokens": tokens,
                    "positives": positives,
                    "auroc": auroc,
                    "ap": ap,
                })
    return pd.DataFrame(rows)


def macro_within_answer(table, high_jump):
    rows = []
    for scope in scope_masks(table, high_jump):
        values = {score: [] for score in SCORES}
        aps = {score: [] for score in SCORES}
        for _, answer in table.groupby("id", sort=False):
            mask = scope_masks(answer, high_jump)[scope]
            labels = answer.gold.to_numpy()[mask]
            if len(labels) == 0 or len(np.unique(labels)) < 2:
                continue
            for score in SCORES:
                current = answer[score].to_numpy()[mask]
                finite = np.isfinite(current)
                if len(np.unique(labels[finite])) < 2:
                    continue
                values[score].append(
                    roc_auc_score(labels[finite], current[finite])
                )
                aps[score].append(
                    average_precision_score(labels[finite], current[finite])
                )
        for score in SCORES:
            rows.append(dict(
                scope=scope,
                score=score,
                mixed_answers=len(values[score]),
                macro_auroc=(
                    float(np.mean(values[score])) if values[score] else np.nan
                ),
                macro_ap=(
                    float(np.mean(aps[score])) if aps[score] else np.nan
                ),
            ))
    return pd.DataFrame(rows)


def write_stratified_reports(table, high_jump, output):
    grouped_metrics(table, high_jump, "task").to_csv(
        output / "metrics_by_task.csv", index=False
    )
    grouped_metrics(table, high_jump, "generator").to_csv(
        output / "metrics_by_generator.csv", index=False
    )
    macro_within_answer(table, high_jump).to_csv(
        output / "macro_within_answer.csv", index=False
    )
