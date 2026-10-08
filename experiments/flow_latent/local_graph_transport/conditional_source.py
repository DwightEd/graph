"""Source-balanced conditional ranks of message routing given attention.

The reference population is the unlabeled mixture of fit answers. The rank is
a measurement of unusual routing at similar attention, not a truth probability.
"""
import numpy as np

from .unlabeled import equal_source_weights, fit_weighted_cdf, reference_rank


ATTENTION_BINS = 16


def fit_conditional_reference(route, attention, source_index):
    """Fit fixed equal-mass attention bins and a weighted route CDF in each."""
    weights = equal_source_weights(source_index)
    values, cumulative = fit_weighted_cdf(attention, weights)
    quantiles = np.arange(1, ATTENTION_BINS) / ATTENTION_BINS
    # Put an entire tied attention value in the lower bin. The next distinct
    # value starts the upper bin; tolerance handles cumulative summation error.
    cuts = np.searchsorted(cumulative[1:], quantiles - 8 * np.finfo(float).eps)
    cuts = cuts[cuts + 1 < len(values)]
    edges = np.unique(values[cuts + 1])
    bins = np.searchsorted(edges, attention, side='right')
    reference = dict(attention_edges=edges)
    for index in range(len(edges) + 1):
        selected = bins == index
        if not selected.any():
            raise ValueError('Empty conditional reference bin')
        distinct, mass = fit_weighted_cdf(route[selected], weights[selected])
        reference[f'bin_{index}_values'] = distinct
        reference[f'bin_{index}_cumulative'] = mass
    return reference


def conditional_rank(reference, route, attention):
    """Evaluate P(R<r|attention bin)+.5 P(R=r|attention bin)."""
    bins = np.searchsorted(reference['attention_edges'], attention, side='right')
    result = np.empty(len(route), dtype=np.float64)
    for index in np.unique(bins):
        selected = bins == index
        result[selected] = reference_rank(reference, f'bin_{index}', route[selected])
    return result
