"""Joint score/label permutations diagnose smoothing, never generate detector scores."""

import numpy as np

from ..evaluate import Ranking
from .cross_terms import causal_mean


def observed_runs(values):
    positions = np.flatnonzero(np.isfinite(values))
    return np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)


def smooth_run(values, window):
    positions = np.arange(len(values))
    counts = np.minimum(positions + 1, window)
    return causal_mean(values, positions, counts, window)


def ordering_arrays(blocks, method, window, random=None):
    labels, current, smoothed = [], [], []
    adjacent_positive = previous_positive = 0
    for block in blocks:
        scores = block["scores"][method]
        target = block["views"]["all_error"][0]
        for run in observed_runs(scores):
            if not len(run):
                continue
            # Move each (label, score) pair together: pointwise ranking is unchanged.
            order = run if random is None else random.permutation(run)
            values, error = scores[order], target[order]
            labels.extend(error)
            current.extend(values)
            smoothed.extend(smooth_run(values, window))
            adjacent_positive += int((error[:-1] & error[1:]).sum())
            previous_positive += int(error[:-1].sum())
    return (np.asarray(labels, bool), np.asarray(current), np.asarray(smoothed),
            adjacent_positive / previous_positive if previous_positive else None)


def ordering_measure(blocks, method, window, random=None):
    labels, current, smoothed, persistence = ordering_arrays(blocks, method, window, random)
    first = Ranking(labels, current).measure()
    second = Ranking(labels, smoothed).measure()
    result = {"tokens": len(labels), "positives": int(labels.sum()), "raw": first, "smooth": second,
                  "positive_after_positive": persistence}
    for metric in ("auroc", "ap"):
        result[metric + "_gain"] = (
            second[metric] - first[metric] if first[metric] is not None else None
        )
    return result


def order_audit(blocks, method, window, permutations=20, seed=17):
    """Within-answer finite-run null; destroys order, including precise position effects."""
    observed = ordering_measure(blocks, method, window)
    random = np.random.default_rng(seed)
    draws = [dict(draw=index, **ordering_measure(blocks, method, window, random))
             for index in range(permutations)]
    summary = {}
    for field in ("auroc_gain", "ap_gain", "positive_after_positive"):
        values = [row[field] for row in draws if row[field] is not None]
        summary[field] = {
            "mean": float(np.mean(values)) if values else None,
            "null_range95": np.quantile(values, [.025, .975]).tolist() if values else None,
        }
    return {
        "method": method, "window": window, "permutations": permutations, "seed": seed,
        "purpose": "posthoc_order_null_not_detector_evaluation",
        "preserves": "answer identity, missing gaps, pointwise score-label pairs",
        "destroys": "within-run order, label continuity and precise position effects",
        "score_space": "saved calibrated scalar; smoothing recomputed only for this diagnostic",
        "coverage": "selected-method common finite coverage; recomputed windows reset at common gaps",
        "threshold": "no new alarms or thresholds; not the second-moment detector",
        "observed": observed, "null": summary, "draws": draws,
    }
