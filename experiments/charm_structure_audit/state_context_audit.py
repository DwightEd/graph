"""Does supervised token detection read label continuity or context carried by x_t?"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import write_json
from .lda_data import read_audit_inputs
from .lda_math import moments, coefficients, fit_linear_prediction


VIEWS = (
    "current",
    "previous",
    "past_mean",
    "delta_previous",
    "current_residual_after_previous",
)


def temporal_views(values, table, window):
    """Build past-only views within each answer; no gold is used."""
    previous = np.full_like(values, np.nan, dtype=float)
    past_mean = np.full_like(values, np.nan, dtype=float)

    for _, group in table.groupby("id", sort=False):
        indices = group.index.to_numpy()
        block = values[indices]
        if len(block) < 2:
            continue
        previous[indices[1:]] = block[:-1]

        prefix = np.vstack((np.zeros((1, block.shape[1])), np.cumsum(block, axis=0)))
        for offset in range(1, len(block)):
            start = max(0, offset - window)
            past_mean[indices[offset]] = (prefix[offset] - prefix[start]) / (offset - start)

    return previous, past_mean


def fit_view(train, labels, ridge):
    stats = moments(train, labels, ridge)
    return coefficients(stats)


def score_view(values, model):
    return values @ model["weight"] + model["intercept"]


def safe_metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return dict(tokens=len(labels), positives=int(labels.sum()), auroc=np.nan, ap=np.nan)
    return dict(
        tokens=len(labels),
        positives=int(labels.sum()),
        auroc=float(roc_auc_score(labels, scores)),
        ap=float(average_precision_score(labels, scores)),
    )


def scopes(table):
    valid_previous = table.previous_gold.to_numpy() >= 0
    previous = table.previous_gold.to_numpy()
    return {
        "all_lag_valid": valid_previous,
        "previous_gold_0": valid_previous & (previous == 0),
        "previous_gold_1": valid_previous & (previous == 1),
    }


def evaluate_scores(table, score_columns):
    rows = []
    labels = table.gold.to_numpy()
    for scope, mask in scopes(table).items():
        for name, scores in score_columns.items():
            result = safe_metrics(labels[mask], np.asarray(scores)[mask])
            rows.append(dict(scope=scope, model=name, **result))
    return pd.DataFrame(rows)


def run_state_context(args):
    output = Path(args.output) if args.output else Path(args.root) / "audit_state_context"
    output.mkdir(parents=True, exist_ok=True)

    config = type("Args", (), dict(
        root=args.root,
        prepared=args.prepared,
        lda_prompt=False,
    ))()
    data, _ = read_audit_inputs(config)

    split_views = {}
    for split, (values, table, _, _) in data.items():
        previous, past_mean = temporal_views(values.astype(float), table, args.lda_window)
        split_views[split] = dict(
            current=values.astype(float),
            previous=previous,
            past_mean=past_mean,
            delta_previous=values.astype(float) - previous,
        )

    fit_table = data["fit"][1]
    fit_labels = fit_table.gold.to_numpy()
    valid_fit = np.isfinite(split_views["fit"]["previous"]).all(axis=1)

    prediction = fit_linear_prediction(
        split_views["fit"]["previous"][valid_fit],
        split_views["fit"]["current"][valid_fit],
        args.lda_ridge,
    )
    predict_rows = []
    for split in split_views:
        previous = split_views[split]["previous"]
        expected = previous @ prediction["weight"] + prediction["intercept"]
        residual = split_views[split]["current"] - expected
        split_views[split]["current_residual_after_previous"] = residual

        valid = np.isfinite(previous).all(axis=1)
        target = split_views[split]["current"][valid]
        estimate = expected[valid]
        denominator = np.square(target - target.mean(axis=0)).sum()
        r2 = np.nan if denominator == 0 else 1 - np.square(target - estimate).sum() / denominator
        predict_rows.append(dict(split=split, rows=int(valid.sum()), current_from_previous_r2=float(r2)))

    models = {}
    for name in VIEWS:
        values = split_views["fit"][name]
        valid = np.isfinite(values).all(axis=1)
        models[name] = fit_view(values[valid], fit_labels[valid], args.lda_ridge)

    test_table = data["test"][1].copy()
    score_columns = {}
    for name in VIEWS:
        values = split_views["test"][name]
        score = np.full(len(values), np.nan)
        valid = np.isfinite(values).all(axis=1)
        score[valid] = score_view(values[valid], models[name])
        score_columns[name] = score
        test_table[name] = score

    previous_gold = test_table.previous_gold.to_numpy(dtype=float)
    score_columns["previous_gold"] = np.where(previous_gold >= 0, previous_gold, np.nan)
    score_columns["past_run"] = test_table.past_run.to_numpy(dtype=float)

    metrics = evaluate_scores(test_table, score_columns)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(predict_rows).to_csv(output / "current_from_previous.csv", index=False)
    test_table.to_csv(output / "test_scores.csv.gz", index=False)

    write_json(output / "protocol.json", dict(
        question="Does supervised x_t performance reduce to temporal label persistence?",
        views=list(VIEWS),
        scopes=["all_lag_valid", "previous_gold_0", "previous_gold_1"],
        ridge=args.lda_ridge,
        window=args.lda_window,
        interpretation={
            "previous_gold_0": "onset-vs-normal under the same previous normal label",
            "previous_gold_1": "continuation-vs-recovery under the same previous error label",
            "current_residual_after_previous": "current x_t after subtracting FIT-only linear prediction from x_(t-1)",
        },
    ))

    pivot = metrics.pivot(index="model", columns="scope", values="auroc")
    lines = [
        "# Continuity vs context-carrying current state",
        "",
        "x_t is the original 1024-D self-attention-diagonal vector (layer×head).",
        "It is a single-token input to LDA, but it is not context-free: q_t/k_t and the attention denominator are prefix-conditioned.",
        "",
        "The decisive controls are fixed previous-label strata and the current-state residual after predicting x_t from x_(t-1).",
        "",
        pivot.to_string(),
        "",
        "Interpretation:",
        "- previous_gold baseline measures pure label persistence.",
        "- previous uses only x_(t-1), so any performance is past-state/continuity information.",
        "- current uses only x_t.",
        "- current_residual_after_previous removes the linearly predictable part of x_t from x_(t-1).",
        "- previous_gold_0 asks whether current state separates a new error from normal while the previous label is fixed normal.",
        "- previous_gold_1 asks whether it separates continued error from recovery while the previous label is fixed error.",
    ]
    (output / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)
    print("Results:", output, flush=True)
