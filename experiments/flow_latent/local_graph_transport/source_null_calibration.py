"""Anchor source-effect ranks at zero output effect, without factual labels.

The midpoint is a statistical coordinate. Equal actual-token log probabilities
do not establish absent source use, and these ranks are not truth probabilities.
"""
import numpy as np

from .likelihood_calibration import conditional_cdf


def bounded_center(ranks, null_ranks):
    """Preserve ordering with slope <=1, and set the null coordinate to .5."""
    ranks = np.asarray(ranks, dtype=np.float64)
    null_ranks = np.asarray(null_ranks, dtype=np.float64)
    denominator = 2 * np.maximum(null_ranks, 1 - null_ranks)
    return .5 + (ranks - null_ranks) / denominator


def balanced_center(ranks, null_ranks):
    """Fixed control: scale each observable tail separately, retaining gaps.

    Strict inequalities make empty-tail divisions impossible. A rare tail can
    amplify rank changes; it is deliberately not the primary transformation.
    """
    ranks, null_ranks = np.broadcast_arrays(
        np.asarray(ranks, dtype=np.float64), np.asarray(null_ranks, dtype=np.float64))
    centered = np.full(ranks.shape, .5)
    below = ranks < null_ranks
    above = ranks > null_ranks
    centered[below] = ranks[below] / (2 * null_ranks[below])
    centered[above] = .5 + (ranks[above] - null_ranks[above]) / (2 * (1 - null_ranks[above]))
    return centered


def conditional_coordinates(reference, effect, likelihood):
    """Compare an observed effect and exact zero in the same frozen bin."""
    ranks = conditional_cdf(reference, effect, likelihood)
    null_ranks = conditional_cdf(reference, np.zeros_like(effect), likelihood)
    return ranks, null_ranks
