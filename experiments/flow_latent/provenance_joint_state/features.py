"""Fit-only multi-head observations and matched message-edge controls."""
from dataclasses import dataclass

import numpy as np
from sklearn.utils.extmath import randomized_svd


def observations(record, view='full'):
    if view == 'self':
        return np.column_stack((record['source_gap'],
            np.log1p(1000 * record['self_diagonal']).reshape(len(record['rows']), -1))).astype(float)
    return np.column_stack((record['source_gap'], record['signed'].reshape(len(record['rows']), -1),
        np.log1p(1000 * record['self_diagonal']).reshape(len(record['rows']), -1))).astype(float)


@dataclass
class FeatureMap:
    center: np.ndarray
    scale: np.ndarray
    signed_basis: np.ndarray
    mass_basis: np.ndarray
    edge_scale: np.ndarray
    local_center: np.ndarray = None
    local_scale: np.ndarray = None
    view: str = 'full'


def source_anchored_basis(records, signed, center, scale):
    """Predict observed donor/receiver source gaps; no hallucination labels."""
    targets = []
    for record in records:
        gap = (record['source_gap'] - center) / scale
        donor = np.arange(len(gap))[:, None] - np.arange(1, 9)
        valid = donor >= 0
        pair = np.stack((gap[donor.clip(min=0)] * valid,
                         np.broadcast_to(gap[:, None], donor.shape) * valid), axis=2)
        targets.append(pair.reshape(-1, 2))
    targets = np.concatenate(targets)
    head_scale = np.maximum(signed.std(0), 1e-4)
    whitened = signed / head_scale
    penalty = .01 * len(signed)
    coefficient = np.linalg.solve(whitened.T @ whitened + penalty * np.eye(1024),
                                  whitened.T @ targets)
    return coefficient / head_scale[:, None]


def fit_features(records, encoder='svd', view='full'):
    matrix = np.concatenate([observations(record, view) for record in records])
    center = matrix.mean(0)
    scale = np.maximum(matrix.std(0), 1e-4)
    signed = np.concatenate([record['edge_signed'].reshape(-1, 1024) for record in records])
    mass = np.concatenate([record['edge_mass'].reshape(-1, 1024) for record in records])
    # No across-head averaging: independent signed/mass bases retain head coordinates.
    if encoder == 'anchored':
        signed_basis = source_anchored_basis(records, signed, center[0], scale[0])
    else:
        signed_basis = randomized_svd(signed, 2, random_state=42)[2].T
    mass_basis = randomized_svd(mass, 2, random_state=42)[2].T
    edge = np.column_stack((signed @ signed_basis, mass @ mass_basis))
    edge_scale = np.maximum(np.sqrt(np.square(edge).mean(0)), 1e-5) * np.sqrt(8)
    local = (edge / edge_scale).reshape(len(matrix), -1)
    return FeatureMap(center, scale, signed_basis, mass_basis, edge_scale,
                      local.mean(0), np.maximum(local.std(0), 1e-4), view)


def matched_rewire(record, seed):
    """Per-query/head swaps within lag {3,4}/{5..8}; lag1/2 and sums preserved."""
    signed = record['edge_signed'].copy()
    mass = record['edge_mass'].copy()
    generator = np.random.default_rng(seed)
    moved = 0.
    for target in range(len(signed)):
        for lower, upper in ((2, 4), (4, 8)):
            columns = np.arange(lower, min(upper, target))
            if len(columns) < 2:
                continue
            # Each native layer/head keeps its own message/attention multiset.
            random = generator.random((len(columns), 32, 32))
            permutation = random.argsort(axis=0)
            selected = columns[permutation]
            changed = selected != columns[:, None, None]
            moved += float((np.abs(record['edge_signed'][target, columns]) * changed).sum())
            for original, destination in ((record['edge_signed'], signed), (record['edge_mass'], mass)):
                destination[target, columns] = np.take_along_axis(original[target, columns], permutation, axis=0)
    total = float(np.abs(record['edge_signed']).sum())
    distance = float(np.abs(signed - record['edge_signed']).sum() / max(2 * total, 1e-30))
    return signed, mass, dict(adopted_mass_reassigned=moved / max(total, 1e-30),
        adopted_relative_l1=distance, valid_null=moved / max(total, 1e-30) >= .25,
        exact_bpe_matched=False, lag_groups=[[1], [2], [3, 4], [5, 6, 7, 8]])


def transform(record, feature_map, kind, seed=42):
    matrix = (observations(record, feature_map.view) - feature_map.center) / feature_map.scale
    count = len(matrix)
    if kind == 'node':
        return matrix, np.zeros((count, 8, 0)), {}
    if kind == 'chain':
        edges = np.zeros((count, 8, 4))
        for lag in range(1, 9):
            group = 0 if lag == 1 else 1 if lag == 2 else 2 if lag <= 4 else 3
            edges[lag:, lag - 1, group] = 1 / 8
        return matrix, edges, {}
    signed, mass = record['edge_signed'], record['edge_mass']
    diagnostic = {}
    if kind == 'rewired':
        signed, mass, diagnostic = matched_rewire(record, seed)
    adopted = signed.reshape(count, 8, -1) @ feature_map.signed_basis
    routed = mass.reshape(count, 8, -1) @ feature_map.mass_basis
    edges = np.concatenate((adopted, routed), axis=2) / feature_map.edge_scale
    if kind == 'edge_node':
        local = (edges.reshape(count, -1) - feature_map.local_center) / feature_map.local_scale
        return np.column_stack((matrix, local)), np.zeros((count, 8, 0)), {}
    return matrix, edges, diagnostic


def initial_parameters(matrix, rank, basis, seed, penalty, head_weight=1.):
    from .gaussian import Parameters
    source = matrix[:, 0]
    loading = np.zeros((matrix.shape[1], rank))
    loading[:, 0] = matrix.T @ source / len(matrix)
    loading[0] = np.eye(rank)[0]
    remainder = matrix - source[:, None] * loading[:, 0]
    _, singular, vectors = randomized_svd(remainder, rank - 1, random_state=seed)
    loading[:, 1:] = vectors.T * (singular / np.sqrt(len(matrix)))[None]
    loading[0] = np.eye(rank)[0]
    transition = np.zeros((basis, rank, rank))
    weight = np.full(matrix.shape[1], head_weight)
    weight[0] = 1
    return Parameters(loading, transition, np.ones(rank), np.ones(matrix.shape[1]), penalty,
                      observation_weight=weight)


def local_mean(values, radius=4):
    return np.array([values[max(0, t - radius):min(len(values), t + radius + 1)].mean()
                     for t in range(len(values))])


def text_unit_mean(values, token_text):
    result = np.empty_like(values)
    begin = 0
    for index, token in enumerate(token_text):
        if any(mark in token for mark in ('.', '?', '!', '\n', '。', '？', '！')) or index == len(values) - 1:
            result[begin:index + 1] = values[begin:index + 1].mean()
            begin = index + 1
    return result
