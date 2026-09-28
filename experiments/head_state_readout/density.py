"""Source-balanced conditional neighbour distance ratios; no truth labels."""

import numpy as np

CONTEXT_NEIGHBORS = 16
STATE_NEIGHBORS = 8
REFERENCE_PER_SOURCE = 64


def fit_scale(reference):
    center = np.nanmedian(reference, axis=0)
    quartiles = np.nanquantile(reference, [.25, .75], axis=0)
    scale = quartiles[1] - quartiles[0]
    # Constant coordinates cannot define a near-zero unit for new observations.
    scale = np.where(scale > 1e-5, scale, 1.)
    return center, scale


def standardize(values, fitted):
    center, scale = fitted
    missing = ~np.isfinite(values)
    normalized = (np.where(missing, center, values) - center) / scale
    # Missing observations remain distinguishable from an observed median.
    return np.concatenate((normalized, missing.astype(np.float32)), axis=1).astype(np.float32)


def distances(left, right):
    """Squared full-coordinate distance; no feature projection or head averaging."""
    left_norm = np.sum(left.astype(np.float64) ** 2, axis=1)
    right_norm = np.sum(right.astype(np.float64) ** 2, axis=1)
    product = left @ right.T
    squared = left_norm[:, None] + right_norm[None] - 2 * product
    return np.maximum(squared / left.shape[1], 0)


def select_neighbors(state_distance, context_distance, query_sources, reference_sources, conditional):
    selected = []
    source_ids = np.unique(reference_sources)
    for index, source in enumerate(query_sources):
        indices = []
        for reference_source in source_ids:
            if reference_source == source:
                continue
            candidates = np.flatnonzero(reference_sources == reference_source)
            if conditional:
                ranking = np.argsort(context_distance[index, candidates], kind='stable')
                candidates = candidates[ranking[:CONTEXT_NEIGHBORS]]
            ranking = np.argsort(state_distance[index, candidates], kind='stable')
            indices.extend(candidates[ranking[:STATE_NEIGHBORS]])
        selected.append(np.asarray(indices))
    return selected


def local_ratios(query_distance, reference_distance, query_context, reference_context,
                 query_sources, reference_sources, conditional=True):
    reference_neighbors = select_neighbors(reference_distance, reference_context,
        reference_sources, reference_sources, conditional)
    radius = np.array([np.sqrt(reference_distance[index, ids]).mean()
                      for index, ids in enumerate(reference_neighbors)])
    neighbors = select_neighbors(query_distance, query_context, query_sources, reference_sources, conditional)
    query_radius = np.array([np.sqrt(query_distance[index, ids]).mean()
                            for index, ids in enumerate(neighbors)])
    neighbor_radius = np.array([radius[ids].mean() for ids in neighbors])
    ratio = query_radius / np.maximum(neighbor_radius, 1e-8)
    return ratio, neighbors, query_radius, neighbor_radius
