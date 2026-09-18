"""Closed-form LDA and exact score decompositions; no neural network or graph propagation."""

import numpy as np
from scipy.linalg import solve


def moments(values, labels, ridge=1e-3, weights=None):
    """FIT-only weighted means and pooled within-class covariance, in raw units."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.ones(len(values)) if weights is None else np.asarray(weights)
    center = np.average(values, axis=0, weights=weights)
    variance = np.average((values - center) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1
    counts = np.array([weights[labels == label].sum() for label in (0, 1)])
    means = np.stack([np.average(values[labels == label], axis=0,
                                weights=weights[labels == label]) for label in (0, 1)])
    residual = (values - means[labels.astype(int)]) * np.sqrt(weights[:, None])
    covariance = residual.T @ residual / weights.sum()
    covariance.flat[::values.shape[1] + 1] += ridge * scale ** 2
    return dict(means=means, covariance=covariance, scale=scale,
                center=center, priors=counts / counts.sum())


def coefficients(stats, structure='full', heads=32):
    """Change only the covariance entries used to solve Sigma w = mean_gap."""
    covariance = stats['covariance']
    difference = stats['means'][1] - stats['means'][0]
    if structure == 'raw_mean':
        weight = difference.copy()
    elif structure == 'diagonal':
        weight = difference / np.diag(covariance)
    elif structure == 'layer_block':
        weight = np.empty_like(difference)
        for start in range(0, len(weight), heads):
            block = slice(start, start + heads)
            weight[block] = solve(covariance[block, block], difference[block], assume_a='pos')
    else:
        weight = solve(covariance, difference, assume_a='pos')
    intercept = np.log(stats['priors'][1] / stats['priors'][0])
    intercept -= .5 * stats['means'].sum(axis=0) @ weight
    return dict(weight=weight, intercept=intercept)


def covariance_terms(stats, weight, heads):
    """Exact identity: w = direct + same_layer_compensation + cross_layer_compensation."""
    covariance = stats['covariance']
    diagonal = np.diag(covariance)
    direct = (stats['means'][1] - stats['means'][0]) / diagonal
    same_layer = np.zeros_like(weight)
    for start in range(0, len(weight), heads):
        block = slice(start, start + heads)
        same_layer[block] = -(covariance[block, block] @ weight[block]
                              - diagonal[block] * weight[block]) / diagonal[block]
    cross_layer = -(covariance @ weight - diagonal * weight) / diagonal - same_layer
    return dict(direct=direct, same_layer=same_layer, cross_layer=cross_layer)


def common_and_contrast(weight, heads):
    """In RAW attention units: full weight = layer-common + within-layer contrast."""
    by_head = weight.reshape(-1, heads)
    common = np.repeat(by_head.mean(axis=1), heads)
    return common, weight - common


def head_features(values, heads, mode):
    by_head = values.reshape(len(values), -1, heads)
    average = by_head.mean(axis=2)
    if mode == 'layer_mean':
        return average
    return (by_head - average[:, :, None]).reshape(values.shape)


def past_mean(scores, ids, window=10):
    """Past-only arithmetic mean; no gold boundaries. First token has no observation."""
    result = np.full(len(scores), np.nan)
    for identity in np.unique(ids):
        indices = np.flatnonzero(ids == identity)
        values = np.asarray(scores)[indices]
        prefix = np.r_[0., np.cumsum(values)]
        stop = np.arange(1, len(values))
        start = np.maximum(0, stop - window)
        result[indices[1:]] = (prefix[stop] - prefix[start]) / (stop - start)
    return result


def balance_context(table, minimum=3):
    """Oracle TRAIN diagnostic: balance labels within past-run/position/surface strata."""
    weights = np.zeros(len(table))
    rows = []
    keys = ['position_bin', 'surface', 'past_run_bin']
    for key, indices in table.groupby(keys, sort=True).indices.items():
        labels = table.gold.to_numpy()[indices]
        counts = np.bincount(labels, minlength=2)
        eligible = bool(counts.min() >= minimum)
        if eligible:
            weights[indices] = counts.min() / counts[labels]
        rows.append(dict(zip(keys, key), normal=int(counts[0]), error=int(counts[1]), eligible=eligible))
    return weights, rows


def fit_linear_prediction(features, target, ridge=1e-3):
    """FIT-only regression onto prompt reading; no test/gold access here."""
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    inputs = (features - center) / scale
    covariance = inputs.T @ inputs / len(inputs)
    covariance.flat[::len(scale) + 1] += ridge
    target_center = target.mean(axis=0)
    weight = solve(covariance, inputs.T @ (target - target_center) / len(inputs), assume_a='pos')
    return dict(weight=weight / scale[:, None],
                intercept=target_center - (center / scale) @ weight)
