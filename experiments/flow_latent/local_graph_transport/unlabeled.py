"""Fixed source-directed token scores and local-graph continuity controls.

Reference ranks use fit-source observations without natural hallucination labels.
They describe relative source sensitivity, not calibrated factual probabilities.
All graph smoothing here uses the complete answer and is explicitly offline.
"""
import numpy as np
import torch

from .smooth import smooth_field


CHANNELS = ('source_local', 'source_full', 'raw_route')
PENALTY = .5
HUBER_DELTA = 1.


def equal_source_weights(source_index):
    """Give each fit source equal mass, distributed over all its answer tokens."""
    _, inverse, counts = np.unique(source_index, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    return weights / weights.sum()


def fit_weighted_cdf(values, weights):
    """Archive distinct observed values and their cumulative reference mass."""
    order = np.argsort(values, kind='stable')
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    distinct, starts = np.unique(ordered_values, return_index=True)
    masses = np.add.reduceat(weights[order], starts)
    cumulative = np.r_[0., np.cumsum(masses) / masses.sum()]
    return distinct, cumulative


def fit_reference(source_local, source_full, raw_route, source_index):
    """Fit three weighted empirical CDFs; inputs contain no labels or dev/test rows.

    Higher absent-minus-present original-token logp and higher raw_route retain
    the existing risk direction. The returned flat arrays can be saved as NPZ.
    """
    weights = equal_source_weights(source_index)
    reference = {}
    for name, values in zip(CHANNELS, (source_local, source_full, raw_route)):
        distinct, cumulative = fit_weighted_cdf(values, weights)
        reference[f'{name}_values'] = distinct
        reference[f'{name}_cumulative'] = cumulative
    return reference


def reference_rank(reference, name, values):
    """Weighted mid-CDF: P(X<x) + .5 P(X=x), preserving ties and direction."""
    distinct = reference[f'{name}_values']
    cumulative = reference[f'{name}_cumulative']
    left = np.searchsorted(distinct, values, side='left')
    right = np.searchsorted(distinct, values, side='right')
    return .5 * (cumulative[left] + cumulative[right])


def local_weights(local_attention):
    """Mean physical-head attention [token, head, lag] to strict answer history.

    Caller supplies post-token rows only: collector local_attention[0, 1:].
    Lag slot zero denotes t-1; this mean defines a scalar continuity prior,
    without claiming native signed value transport or head-level detection.
    """
    weights = np.asarray(local_attention, dtype=np.float64).mean(axis=1)
    targets = np.arange(len(weights))[:, None]
    lags = np.arange(1, weights.shape[1] + 1)[None]
    return np.where(targets >= lags, weights, 0.)


def local_graph_edges(weights):
    """Flatten complete strict-past lag slots; zero-weight edges stay explicit."""
    targets, slots = np.indices(weights.shape)
    sources = targets - slots - 1
    valid = sources >= 0
    return (torch.from_numpy(sources[valid]).long(),
            torch.from_numpy(targets[valid]).long(),
            torch.from_numpy(weights[valid]).double())


def chain_weights(weights):
    """Collapse each receiver's local attention mass onto its immediate predecessor."""
    chain = np.zeros_like(weights)
    chain[1:, 0] = weights[1:].sum(axis=1)
    return chain


def rewire_weights(weights, seed=42):
    """Permute endpoint weights within lag {3,4} and {5,...,8}; lag1/2 fixed.

    Only eligible strict-past endpoints participate. A nonzero circular shift
    preserves receiver mass and each group's weight multiset. No token labels
    or punctuation define groups; equal weights can make a shift ineffective.
    """
    rewired = weights.copy()
    random = np.random.default_rng(seed)
    for target in range(len(weights)):
        for first, stop in ((2, 4), (4, 8)):
            slots = np.arange(first, min(stop, weights.shape[1], target))
            if len(slots) > 1:
                shift = random.integers(1, len(slots))
                rewired[target, slots] = np.roll(weights[target, slots], shift)
    return rewired


def graph_fields(unary, graphs):
    """Keep each original token unary and optimize all fixed graph controls."""
    field = torch.from_numpy(np.asarray(unary, dtype=np.float64))
    scores = dict(unary=field.numpy().copy())
    diagnostics = {}
    for name, weights in graphs.items():
        sources, targets, raw = local_graph_edges(weights)
        result = smooth_field(field, sources, targets, raw, PENALTY, HUBER_DELTA)
        scores[f'{name}_huber'] = result.pop('solution').numpy()
        diagnostics[name] = result
    return scores, diagnostics


def score_answer(reference, source_local, source_full, raw_route,
                 local_attention, seed=42):
    """Return eight fixed scores plus ranks and numeric smoothing diagnostics.

    Primary is source=.5*(rank_local+rank_full). The additional source_route
    family is fixed .75*source+.25*rank_route; labels never choose between them.
    Each token keeps its own unary, independent of textual or gold spans.
    """
    ranks = {name: reference_rank(reference, name, values)
             for name, values in zip(CHANNELS, (source_local, source_full, raw_route))}
    source = .5 * (ranks['source_local'] + ranks['source_full'])
    fields = dict(source=source, source_route=.75 * source + .25 * ranks['raw_route'])
    native = local_weights(local_attention)
    graphs = dict(native=native, chain=chain_weights(native),
                  rewired=rewire_weights(native, seed))
    scores = {}
    diagnostics = dict(offline=True, penalty=PENALTY, huber_delta=HUBER_DELTA,
        rewire_seed=seed, primary='source_native_huber', labels_used=False,
        rewired_edge_fraction=float(np.count_nonzero(native != graphs['rewired']) /
            max(1, np.count_nonzero(np.tri(*native.shape, k=-1)))),
        rewired_weight_fraction=float(np.abs(native - graphs['rewired']).sum() /
            max(np.finfo(float).tiny, 2 * native.sum())))
    for name, unary in fields.items():
        family_scores, family_diagnostics = graph_fields(unary, graphs)
        scores.update({f'{name}_{method}': values for method, values in family_scores.items()})
        diagnostics[name] = family_diagnostics
    return dict(scores=scores, ranks=ranks, diagnostics=diagnostics)
