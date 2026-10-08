"""Two fixed refinements of source/route unaries and offline graph continuity.

Max preserves a strong deletion-sensitivity view; it does not certify evidence.
Reference calibration and the Huber scale use mixed, unlabeled fit sources.
The primary and controls are specified before natural-label evaluation.
"""
import numpy as np
import torch

from .smooth import normalize_edge_weights, smooth_field
from .unlabeled import (CHANNELS, PENALTY, chain_weights, equal_source_weights,
                        fit_weighted_cdf, local_graph_edges, local_weights,
                        reference_rank, rewire_weights)


PRIMARY = 'union_robust'
ROUTE_WEIGHT = .25
METHODS = ('old_unary', 'old_native', 'old_robust', 'union_unary',
           'union_native', 'union_robust', 'mean_cdf_unary',
           'mean_cdf_native', 'mean_cdf_robust', 'union_chain', 'union_rewired')


def fit_fusion_reference(local_rank, full_rank, source_index):
    """Calibrate max and mean using the same equal-source valid-token population."""
    weights = equal_source_weights(source_index)
    reference = {}
    for name, values in (('union', np.maximum(local_rank, full_rank)),
                         ('mean', .5 * (local_rank + full_rank))):
        distinct, cumulative = fit_weighted_cdf(values, weights)
        reference[f'{name}_values'] = distinct
        reference[f'{name}_cumulative'] = cumulative
    return reference


def weighted_median(values, weights):
    """One predeclared quantile, with a failure rather than a fitted fallback."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('A finite positive reference mass is required')
    distinct, cumulative = fit_weighted_cdf(values, weights)
    median = float(distinct[np.searchsorted(cumulative[1:], .5, side='left')])
    if median <= 0:
        raise ValueError('Fit edge-difference median is zero; no fallback is authorized')
    return median


def unaries(scalar_reference, fusion_reference, values):
    ranks = {name: reference_rank(scalar_reference, name, values[name])
             for name in CHANNELS}
    mean = .5 * (ranks['source_local'] + ranks['source_full'])
    union = np.maximum(ranks['source_local'], ranks['source_full'])
    source = dict(old=mean, union=reference_rank(fusion_reference, 'union', union),
                  mean_cdf=reference_rank(fusion_reference, 'mean', mean))
    fields = {name: (1 - ROUTE_WEIGHT) * field + ROUTE_WEIGHT * ranks['raw_route']
              for name, field in source.items()}
    return fields, dict(ranks, source_mean=mean, source_max=union,
                       source_union_cdf=source['union'], source_mean_cdf=source['mean_cdf'])


def normalized_edges(attention):
    native = local_weights(attention)
    senders, receivers, raw = local_graph_edges(native)
    weights = normalize_edge_weights(senders, receivers, raw, len(native))
    return senders, receivers, weights


def edge_reference(unary, attention):
    """Full-answer old-unary differences and the actual normalized objective mass."""
    senders, receivers, weights = normalized_edges(attention)
    differences = np.abs(unary[receivers.numpy()] - unary[senders.numpy()])
    positive = weights.numpy() > 0
    return differences[positive], weights.numpy()[positive]


def score_answer(scalar_reference, fusion_reference, delta, values, attention):
    """Solve the eleven fixed full-answer fields, without reading token labels."""
    fields, ranks = unaries(scalar_reference, fusion_reference, values)
    native = local_weights(attention)
    graphs = dict(native=native, chain=chain_weights(native),
                  rewired=rewire_weights(native, seed=42))
    scores = {f'{name}_unary': field.copy() for name, field in fields.items()}
    diagnostics = {}
    for name, unary in fields.items():
        jobs = [('native', 'native', 1.), ('robust', 'native', delta)]
        if name == 'union':
            jobs += [('chain', 'chain', delta), ('rewired', 'rewired', delta)]
        for suffix, graph, scale in jobs:
            senders, receivers, raw = local_graph_edges(graphs[graph])
            field = torch.from_numpy(unary).double()
            result = smooth_field(field, senders, receivers, raw, PENALTY, scale)
            solution = result.pop('solution')
            score_name = f'{name}_{suffix}'
            scores[score_name] = solution.numpy()
            normalized = normalize_edge_weights(senders, receivers, raw, len(field))
            difference = (solution[receivers] - solution[senders]).abs()
            active = difference > scale
            positive = normalized > 0
            result.update(delta=float(scale),
                positive_edges=int(positive.sum()),
                linear_edges=int((active & positive).sum()),
                edge_mass=float(normalized.sum()),
                linear_mass=float(normalized[active].sum()),
                linear_edge_fraction=float((active & positive).sum() / positive.sum())
                    if positive.any() else 0.,
                linear_mass_fraction=float(normalized[active].sum() / normalized.sum())
                    if normalized.sum() > 0 else 0.)
            if result['max_actual_change'] > PENALTY * scale + 2e-8:
                raise ValueError(f'{score_name}: incident-budget Huber bound failed')
            diagnostics[score_name] = result
    if set(scores) != set(METHODS):
        raise ValueError('The frozen method set changed')
    return dict(scores=scores, ranks=ranks, diagnostics=diagnostics)
