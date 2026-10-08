"""Fit-only likelihood-conditioned source-effect ranks, not factual probabilities.

An absent-minus-present actual-token logp effect is bounded above by the
present-context surprisal. Conditioning its reference on present logp tests
this heterogeneity while retaining the old risk direction and source weights.
"""
import numpy as np

from .unlabeled import equal_source_weights, fit_weighted_cdf, reference_rank


LIKELIHOOD_BINS = 16


def fit_conditional_cdf(effect, likelihood, source_index):
    """Use global equal-source weights, then normalize their mass within each bin.

    Tied likelihoods stay together; repeated cuts merge. A constant condition
    therefore recovers the unconditional reference rather than empty bins.
    """
    effect = np.asarray(effect, dtype=np.float64)
    likelihood = np.asarray(likelihood, dtype=np.float64)
    source_index = np.asarray(source_index)
    if effect.shape != likelihood.shape or effect.shape != source_index.shape or not len(effect):
        raise ValueError('Fit effect, likelihood and source must share a nonempty axis')
    if not np.isfinite(effect).all() or not np.isfinite(likelihood).all():
        raise ValueError('Conditional reference inputs must be finite')
    weights = equal_source_weights(source_index)
    distinct, cumulative = fit_weighted_cdf(likelihood, weights)
    quantiles = np.arange(1, LIKELIHOOD_BINS) / LIKELIHOOD_BINS
    cuts = np.searchsorted(cumulative[1:], quantiles - 8 * np.finfo(float).eps)
    cuts = cuts[cuts + 1 < len(distinct)]
    edges = np.unique(distinct[cuts + 1])
    bins = np.searchsorted(edges, likelihood, side='right')
    result = dict(likelihood_edges=edges)
    for index in range(len(edges) + 1):
        selected = bins == index
        if not selected.any():
            raise ValueError('Conditional reference contains an empty bin')
        values, mass = fit_weighted_cdf(effect[selected], weights[selected])
        result[f'bin_{index}_values'] = values
        result[f'bin_{index}_cumulative'] = mass
    return result


def conditional_cdf(reference, effect, likelihood):
    """Mid-CDF in the frozen likelihood bin; extrapolate using its outer tails."""
    effect = np.asarray(effect, dtype=np.float64)
    likelihood = np.asarray(likelihood, dtype=np.float64)
    if effect.shape != likelihood.shape or not np.isfinite(effect).all() or not np.isfinite(likelihood).all():
        raise ValueError('Effects and corresponding native logp must be finite and aligned')
    bins = np.searchsorted(reference['likelihood_edges'], likelihood, side='right')
    ranks = np.empty_like(effect)
    for index in np.unique(bins):
        selected = bins == index
        ranks[selected] = reference_rank(reference, f'bin_{index}', effect[selected])
    return ranks


def shuffle_condition(likelihood, source_index):
    """Break fit effect/condition pairing within each source, preserving its multiset."""
    result = np.asarray(likelihood).copy()
    random = np.random.default_rng(42)
    for source in np.unique(source_index):
        selected = np.flatnonzero(source_index == source)
        result[selected] = result[random.permutation(selected)]
    return result
