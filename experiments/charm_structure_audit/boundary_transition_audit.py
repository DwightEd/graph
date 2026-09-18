"""Is hallucination onset special, or a generic sentence/state transition?"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import write_json
from .lda_data import read_audit_inputs
from .lda_math import moments, coefficients, fit_linear_prediction
from .state_context_audit import temporal_views


TRANSITIONS = {
    (0, 0): "normal",
    (0, 1): "onset",
    (1, 1): "continuation",
    (1, 0): "recovery",
}
SCORES = (
    "delta_norm",
    "innovation_norm",
    "layer_js",
    "head_turnover",
    "routing_entropy_change",
    "boundary_conditioned_outlier",
    "current_lda",
)


def sentence_start_flags(table):
    """Surface sentence starts only; this is a control, not a semantic parser."""
    flags = np.zeros(len(table), dtype=bool)
    terminal = (".", "?", "!", "。", "？", "！")
    for _, group in table.groupby("id", sort=False):
        indices = group.index.to_numpy()
        texts = group.text.astype(str).to_numpy()
        flags[indices[0]] = True
        for offset in range(1, len(indices)):
            previous = texts[offset - 1].rstrip()
            current = texts[offset]
            flags[indices[offset]] = (
                previous.endswith(terminal)
                or "\n" in previous
                or current.startswith("\n")
            )
    return flags


def standardize(values, center, scale):
    return (values - center) / scale


def js_by_layer(current, previous, heads):
    result = np.full(len(current), np.nan)
    valid = np.isfinite(previous).all(axis=1)
    if not valid.any():
        return result

    left = np.maximum(current[valid], 0).reshape(-1, current.shape[1] // heads, heads)
    right = np.maximum(previous[valid], 0).reshape(left.shape)
    eps = 1e-12
    left = (left + eps) / (left + eps).sum(axis=2, keepdims=True)
    right = (right + eps) / (right + eps).sum(axis=2, keepdims=True)
    middle = .5 * (left + right)
    js = .5 * (
        (left * np.log(left / middle)).sum(axis=2)
        + (right * np.log(right / middle)).sum(axis=2)
    )
    result[valid] = js.mean(axis=1)
    return result


def routing_entropy(values, heads):
    shaped = np.maximum(values, 0).reshape(-1, values.shape[1] // heads, heads)
    eps = 1e-12
    distribution = (shaped + eps) / (shaped + eps).sum(axis=2, keepdims=True)
    entropy = -(distribution * np.log(distribution)).sum(axis=2) / np.log(heads)
    return entropy.mean(axis=1)


def turnover_by_layer(current, previous, heads):
    result = np.full(len(current), np.nan)
    valid = np.isfinite(previous).all(axis=1)
    if not valid.any():
        return result
    left = current[valid].reshape(-1, current.shape[1] // heads, heads).argmax(axis=2)
    right = previous[valid].reshape(left.shape[0], left.shape[1], heads).argmax(axis=2)
    result[valid] = (left != right).mean(axis=1)
    return result


def robust_reference(features, boundary):
    references = {}
    for value in (False, True):
        rows = features[boundary == value]
        median = np.nanmedian(rows, axis=0)
        mad = np.nanmedian(np.abs(rows - median), axis=0)
        scale = 1.4826 * mad
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        references[value] = (median, scale)
    return references


def outlier_score(features, boundary, references):
    score = np.full(len(features), np.nan)
    for value in (False, True):
        selected = boundary == value
        center, scale = references[value]
        z = (features[selected] - center) / scale
        score[selected] = np.nanmean(z * z, axis=1)
    return score


def row_rms(values):
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values).all(axis=1)
    result[valid] = np.sqrt(np.mean(values[valid] * values[valid], axis=1))
    return result


def build_features(values, table, predictor, center, scale, residual_scale, heads):
    previous, _ = temporal_views(values, table, window=1)
    standardized = standardize(values, center, scale)
    previous_standardized = standardize(previous, center, scale)
    delta = standardized - previous_standardized

    expected = previous @ predictor["weight"] + predictor["intercept"]
    innovation = (values - expected) / residual_scale

    result = pd.DataFrame(index=table.index)
    result["sentence_start"] = sentence_start_flags(table)
    result["delta_norm"] = row_rms(delta)
    result["innovation_norm"] = row_rms(innovation)
    result["layer_js"] = js_by_layer(values, previous, heads)
    result["head_turnover"] = turnover_by_layer(values, previous, heads)
    current_entropy = routing_entropy(values, heads)
    previous_entropy = np.full(len(values), np.nan)
    valid = np.isfinite(previous).all(axis=1)
    previous_entropy[valid] = routing_entropy(previous[valid], heads)
    result["routing_entropy_change"] = current_entropy - previous_entropy
    return result, previous


def transition_labels(table):
    previous = table.previous_gold.to_numpy()
    current = table.gold.to_numpy()
    labels = np.full(len(table), "first", dtype=object)
    for pair, name in TRANSITIONS.items():
        labels[(previous == pair[0]) & (current == pair[1])] = name
    return labels


def safe_metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return dict(tokens=len(labels), positives=int(labels.sum()), auroc=np.nan, ap=np.nan)
    return dict(
        tokens=len(labels),
        positives=int(labels.sum()),
        auroc=float(roc_auc_score(labels, scores)),
        ap=float(average_precision_score(labels, scores)),
    )


def evaluate(table, high_threshold):
    previous = table.previous_gold.to_numpy()
    gold = table.gold.to_numpy()
    sentence_start = table.sentence_start.to_numpy(bool)
    high = table.innovation_norm.to_numpy() >= high_threshold

    scopes = {
        "previous_gold_0": previous == 0,
        "previous_gold_0_sentence_start": (previous == 0) & sentence_start,
        "previous_gold_0_not_sentence_start": (previous == 0) & ~sentence_start,
        "previous_gold_0_high_transition": (previous == 0) & high,
        "previous_gold_1": previous == 1,
    }

    rows = []
    for scope, mask in scopes.items():
        for score in SCORES:
            result = safe_metrics(gold[mask], table.loc[mask, score].to_numpy())
            rows.append(dict(scope=scope, score=score, **result))
    return pd.DataFrame(rows)


def transition_summary(table):
    columns = [
        "delta_norm", "innovation_norm", "layer_js",
        "head_turnover", "routing_entropy_change",
        "boundary_conditioned_outlier", "current_lda",
    ]
    rows = []
    for name, group in table.groupby("transition", sort=False):
        row = dict(
            transition=name,
            tokens=len(group),
            sources=group.source_id.nunique(),
            sentence_start_rate=float(group.sentence_start.mean()),
        )
        for column in columns:
            row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_boundary_transition(args):
    output = Path(args.output) if args.output else Path(args.root) / "audit_boundary_transition"
    output.mkdir(parents=True, exist_ok=True)

    config = type("Args", (), dict(
        root=args.root,
        prepared=args.prepared,
        lda_prompt=False,
    ))()
    data, _ = read_audit_inputs(config)
    heads = data["fit"][3][1]

    fit_values = data["fit"][0].astype(float)
    fit_table = data["fit"][1]
    fit_previous, _ = temporal_views(fit_values, fit_table, window=1)
    valid_fit = np.isfinite(fit_previous).all(axis=1)

    predictor = fit_linear_prediction(
        fit_previous[valid_fit], fit_values[valid_fit], args.lda_ridge
    )
    expected_fit = fit_previous[valid_fit] @ predictor["weight"] + predictor["intercept"]
    residual_fit = fit_values[valid_fit] - expected_fit
    residual_scale = residual_fit.std(axis=0)
    residual_scale[residual_scale < 1e-8] = 1.0

    center = fit_values.mean(axis=0)
    scale = fit_values.std(axis=0)
    scale[scale < 1e-8] = 1.0

    fit_features, _ = build_features(
        fit_values, fit_table, predictor, center, scale, residual_scale, heads
    )
    fit_boundary = fit_features.sentence_start.to_numpy(bool)
    raw_columns = [
        "delta_norm", "innovation_norm", "layer_js",
        "head_turnover", "routing_entropy_change"
    ]
    references = robust_reference(fit_features[raw_columns].to_numpy(), fit_boundary)

    high_threshold = float(
        np.nanquantile(fit_features.innovation_norm.to_numpy(), .90)
    )

    labels = fit_table.gold.to_numpy()
    lda = coefficients(moments(fit_values, labels, args.lda_ridge))

    test_values = data["test"][0].astype(float)
    test_table = data["test"][1].copy()
    features, _ = build_features(
        test_values, test_table, predictor, center, scale, residual_scale, heads
    )
    features["boundary_conditioned_outlier"] = outlier_score(
        features[raw_columns].to_numpy(),
        features.sentence_start.to_numpy(bool),
        references,
    )
    features["current_lda"] = test_values @ lda["weight"] + lda["intercept"]

    for column in features:
        test_table[column] = features[column].to_numpy()
    test_table["transition"] = transition_labels(test_table)

    metrics = evaluate(test_table, high_threshold)
    summary = transition_summary(test_table)
    metrics.to_csv(output / "metrics.csv", index=False)
    summary.to_csv(output / "transition_summary.csv", index=False)
    test_table.to_csv(output / "test_transitions.csv.gz", index=False)

    write_json(output / "protocol.json", dict(
        question="Is hallucination onset a special failure or a generic sentence/state transition?",
        sentence_start="surface punctuation/newline boundary only; not semantic parsing",
        unsupervised_scores=raw_columns + ["boundary_conditioned_outlier"],
        high_transition_threshold=dict(metric="innovation_norm", fit_quantile=.90, value=high_threshold),
        prediction="FIT-only unlabeled x_(t-1) -> x_t linear prediction",
        routing_entropy="normalized entropy across heads within each layer; NOT vocabulary/logit entropy",
        boundary_reference="FIT-only median/MAD, separately for sentence-start and non-start tokens; no hallucination labels",
        current_lda="supervised diagnostic only; never part of the unsupervised score",
    ))

    lines = [
        "# Hallucination onset vs generic state transition",
        "",
        "The audit treats sentence/state transition as a competing explanation, not as hallucination mechanism.",
        "All transition magnitudes and the boundary-conditioned outlier score are label-free.",
        "",
        "## Transition means",
        summary.to_string(index=False),
        "",
        "## Detection under stricter controls",
        metrics.to_string(index=False),
        "",
        "Read previous_gold_0_sentence_start first: if current_lda collapses there, much of onset separability was generic sentence boundary.",
        "Read previous_gold_0_high_transition next: if onset is still separable among generic high-innovation tokens, the signal is more specific than transition magnitude.",
        "boundary_conditioned_outlier is the direct unsupervised candidate: unusual transition relative to the same surface-boundary class.",
    ]
    (output / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")

    print(summary.to_string(index=False), flush=True)
    print(metrics.to_string(index=False), flush=True)
    print("Results:", output, flush=True)
