"""Label-only evaluation utilities with exact tie handling and coverage."""

import json

import numpy as np


class Ranking:
    """Pre-sort once; evaluate weighted AUROC/AP at distinct score thresholds."""

    def __init__(self, labels, scores):
        self.labels = np.asarray(labels, bool)
        scores = np.asarray(scores, float)
        if scores.shape != self.labels.shape or scores.ndim != 1 or np.isnan(scores).any():
            raise ValueError("Ranking requires aligned vectors without NaN scores")
        self.order = np.argsort(scores, kind="stable")
        ordered = scores[self.order]
        self.starts = np.r_[0, 1 + np.flatnonzero(ordered[1:] != ordered[:-1])] if len(scores) else np.array([], int)

    def measure(self, weights=None):
        weights = np.ones(len(self.labels)) if weights is None else np.asarray(weights, float)
        positive = float(weights @ self.labels)
        negative = float(weights.sum() - positive)
        result = dict(prevalence=positive / weights.sum() if weights.sum() else None, auroc=None, ap=None)
        if min(positive, negative) <= 0:
            return result
        p = np.add.reduceat((weights * self.labels)[self.order], self.starts)
        n = np.add.reduceat((weights * ~self.labels)[self.order], self.starts)
        result["auroc"] = float((p * (np.cumsum(n) - .5 * n)).sum() / (positive * negative))
        cp, cn = np.cumsum(p[::-1]), np.cumsum(n[::-1])
        precision = np.divide(cp, cp + cn, out=np.zeros_like(cp), where=(cp + cn) > 0)
        result["ap"] = float((p[::-1] * precision).sum() / positive)
        return result


def _auc(labels, scores):
    return Ranking(labels, scores).measure()["auroc"]


def _average_precision(labels, scores):
    return Ranking(labels, scores).measure()["ap"]


def label_views(offsets, spans):
    offsets = np.asarray(offsets)
    error, onset = np.zeros(len(offsets), bool), np.zeros(len(offsets), bool)
    for span in spans:
        hit = (offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"]) & (offsets[:, 1] > offsets[:, 0])
        if not hit.any():
            raise ValueError("a gold span has no token-offset coverage")
        error |= hit
        onset[np.flatnonzero(hit)[0]] = True
    first, after = np.zeros(len(error), bool), np.zeros(len(error), bool)
    until = np.ones(len(error), bool)
    if error.any():
        t = np.flatnonzero(error)[0]
        first[t], after[t + 1:], until[t + 1:] = True, True, False
    full = np.ones(len(error), bool)
    return dict(all_error=(error, full), span_onset_full_stream=(onset, full),
                span_onset_vs_normal=(onset, ~error | onset), first_error_full_stream=(first, full),
                first_error_until_first=(first, until), continuation_vs_normal=(error & ~onset, ~onset),
                strict_post_first=(error, after))


def scoped_metrics(labels, scores, sources, full_weights, bootstrap=0):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    sources, full_weights = np.asarray(sources), np.asarray(full_weights, float)
    valid = ~np.isnan(scores)
    rank = Ranking(labels[valid], scores[valid])
    result = dict(eligible_tokens=len(labels), evaluated_tokens=int(valid.sum()),
                  eligible_positives=int(labels.sum()), evaluated_positives=int(labels[valid].sum()),
                  coverage=float(valid.mean()) if len(valid) else None,
                  positive_coverage=float(valid[labels].mean()) if labels.any() else None,
                  pooled=rank.measure(), source_fixed_full_answer=rank.measure(full_weights[valid]))
    # The bootstrap unit is the source, including sources with no score coverage.
    _, inverse = np.unique(sources, return_inverse=True)
    count = int(inverse.max()) + 1 if len(inverse) else 0
    draws = []; rng = np.random.default_rng(20260915)
    for _ in range(bootstrap if count > 1 else 0):
        multiplicity = np.bincount(rng.integers(count, size=count), minlength=count)[inverse][valid]
        values = rank.measure(multiplicity)
        if values["auroc"] is not None:
            draws.append([values["auroc"], values["ap"]])
    if bootstrap:
        result["source_bootstrap"] = dict(valid=len(draws), order=["auroc", "ap"],
            ci95=np.quantile(draws, [.025, .975], axis=0).tolist() if draws else None)
    return result


def evaluate_predictions(predictions, annotations):
    """Legacy autoencoder entry: requires prepared annotations with offsets."""
    with open(annotations, encoding="utf-8") as stream:
        rows = {str(row["id"]): row for row in map(json.loads, stream)}
    labels, scores = [], []
    for prediction in predictions:
        row = rows[prediction["response_id"]]
        if "offsets" not in row:
            raise ValueError("legacy GAE evaluation needs prepared offsets; source-flow evaluation reads offsets from its saved artifacts")
        target = label_views(row["offsets"], row["labels"])["all_error"][0]
        values = np.asarray(prediction["score"])
        start = prediction.get("response_idx", len(values) - len(target))
        values = values[start:start + len(target)]
        if len(values) != len(target):
            raise ValueError("token scores and response offsets are not aligned")
        labels.extend(target); scores.extend(values)
    return dict(tokens=len(labels), positives=int(np.sum(labels)), **Ranking(labels, scores).measure())
