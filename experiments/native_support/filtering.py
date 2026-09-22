"""One causal estimator of routing risk, with mean and pooled-head controls."""

import numpy as np

REGIONS = ("source", "history", "self", "other", "inactive")
BANDWIDTH_FLOOR = 1e-8


def head_profile(magnitude, groups, prompt, evidence):
    """Joint head/region probabilities use the same per-layer budget as R_t."""
    keys = magnitude.shape[-1]
    source = groups == 0 if evidence is None else np.r_[evidence, np.zeros(keys - prompt, bool)]
    history = groups == 1 if evidence is None else np.arange(keys) >= prompt
    if evidence is not None:
        source[-1] = False
    own = np.zeros(keys, dtype=bool)
    own[-1] = history[-1]
    history = history & ~own
    other = ~(source | history | own)
    mass = np.stack([magnitude[..., mask].sum(-1) for mask in (source, history, own, other)], axis=-1)
    total = mass.sum((-1, -2), keepdims=True)
    probability = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    inactive = np.broadcast_to((total == 0) / mass.shape[1], (*mass.shape[:-1], 1))
    return np.concatenate((probability, inactive.astype(probability.dtype)), axis=-1)


def squared_distance(root_profiles, current):
    # Hellinger distance over joint head/region budgets, averaged across layers.
    return np.square(root_profiles - current).sum(axis=(-1, -2)).mean(-1) / 2


def causal_filter(scores, profiles, window=16):
    """No fitting, future rows, labels, score propagation or boundary resets."""
    root = np.sqrt(profiles.astype(np.float64))
    adjacent = squared_distance(root[1:], root[:-1])
    count = len(scores)
    weights = np.zeros((count, window))  # column 0=current, column 1=previous
    filtered, mean = np.empty(count), np.empty(count)
    bandwidth = np.full(count, BANDWIDTH_FLOOR)
    for target in range(count):
        start = max(0, target - window + 1)
        indices = np.arange(target, start - 1, -1)
        distances = squared_distance(root[indices], root[target])
        changes = adjacent[max(0, target - window):target]
        if len(changes):
            bandwidth[target] = max(float(np.median(changes)), BANDWIDTH_FLOOR)
        kernel = np.exp(-distances / bandwidth[target])
        kernel /= kernel.sum()
        weights[target, :len(indices)] = kernel
        filtered[target] = kernel @ scores[indices]
        mean[target] = scores[indices].mean()
    return {
        "route_state_filter": filtered, "route_mean": mean, "filter_weights": weights,
        "filter_bandwidth": bandwidth, "filter_current_weight": weights[:, 0],
        "filter_effective_tokens": 1 / np.square(weights).sum(-1),
    }


def filter_scores(features, baseline, window):
    values = features[baseline]
    result = causal_filter(values, features["head_profile"], window)
    pooled = features["head_profile"].sum(axis=2, keepdims=True)
    control = causal_filter(values, pooled, window)
    result["route_pooled_filter"] = control["route_state_filter"]
    return result
