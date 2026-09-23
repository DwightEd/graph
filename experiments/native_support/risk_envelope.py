"""Frozen directional percentiles: any view may raise risk, none cancels another."""

from collections import Counter

import numpy as np

VIEWS = ("route_state", "attention", "entropy")
REFERENCE_QUANTILE = 0.95


def source_weights(source_ids):
    """Choose a source uniformly, then a token uniformly within that source."""
    counts = Counter(source_ids)
    return np.asarray([1 / (len(counts) * counts[source]) for source in source_ids])


def fit_distribution(values, weights):
    order = np.argsort(values, kind="stable")
    cumulative = np.r_[0., np.cumsum(weights[order])]
    cumulative /= cumulative[-1]
    return {"values": np.asarray(values)[order], "cumulative": cumulative}


def percentile(values, distribution):
    """Weighted midrank; equal observations receive equal scores."""
    support = distribution["values"]
    cumulative = distribution["cumulative"]
    left = np.searchsorted(support, values, side="left")
    right = np.searchsorted(support, values, side="right")
    return (cumulative[left] + cumulative[right]) / 2


def reference_threshold(values, weights):
    distribution = fit_distribution(values, weights)
    index = np.searchsorted(distribution["cumulative"][1:], REFERENCE_QUANTILE)
    threshold = float(distribution["values"][index])
    # Strict > preserves ties and does not arbitrarily select tokens at the cutoff.
    return {"threshold": threshold, "comparison": ">",
            "reference_weighted_alarm_rate": float(weights[values > threshold].sum())}


def fit_envelope(reference, source_ids, baseline, attention):
    weights = source_weights(source_ids)
    names = {"route_state": "route_state", "attention": attention, "entropy": "entropy", "route": baseline}
    distributions = {view: fit_distribution(reference[name], weights) for view, name in names.items()}
    scores = apply_envelope(reference, distributions, baseline, attention)
    alarms = (baseline, "route_state", attention, "entropy", "risk_envelope", "instant_envelope")
    thresholds = {name: reference_threshold(scores[name], weights) for name in alarms}
    return distributions, thresholds, scores


def apply_envelope(scores, distributions, baseline, attention):
    names = {"route_state": "route_state", "attention": attention, "entropy": "entropy", "route": baseline}
    ranks = {view: percentile(scores[name], distributions[view]) for view, name in names.items()}
    state_views = np.column_stack([ranks[view] for view in VIEWS])
    instant_views = np.column_stack([ranks[view] for view in ("route", "attention", "entropy")])
    result = {**scores, **{f"percentile_{view}": value for view, value in ranks.items()},
              "risk_envelope": state_views.max(1), "instant_envelope": instant_views.max(1)}
    for view in VIEWS:
        result[f"dominant_{view}"] = ranks[view] == result["risk_envelope"]
    return result
