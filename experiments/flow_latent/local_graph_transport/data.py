"""Train-only transforms and exact prechoice/post-token row alignment."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.probabilistic_detection.data import load_pack


PACKS = Path('outputs/probabilistic_detection_20260928_full/packs')


def scalar_features(pack, timing='posttoken'):
    context, observed = pack['context'], pack['observations']
    local = context[:, 0] + observed[:, 0]
    full = context[:, 1] + observed[:, 1]
    result = np.column_stack([local, full, observed[:, 2], observed[:, 4],
        observed[:, 5], observed[:, 6], context[:, 2], context[:, 5]]).astype(np.float32)
    if timing == 'prechoice':
        # Actual-token likelihoods and total answer length are unavailable here.
        result[:, :3] = 0
        result[:, 6] = context[:, 3]
    return result


def load_partition(split, timing='posttoken'):
    pack, metadata = load_pack(PACKS, 'QA', split)
    pack['scalars'] = scalar_features(pack, timing)
    return pack, metadata


def read_fields(directory):
    """Keep every archived coordinate; the supervised reader learns projections."""
    with np.load(directory / 'arrays.npz') as arrays:
        nodes = arrays['nodes'].astype(np.float32)
        groups = arrays['group_heads'].astype(np.float32).reshape(2, nodes.shape[1], 5, -1)
        values = arrays['value'][:, arrays['query_positions']].astype(np.float32)
        values = np.repeat(values, 4, axis=2)
        fields = dict(node_fields=np.concatenate([nodes[0], nodes[0] - nodes[1]], axis=1),
            boundary_fields=np.concatenate([groups[0], groups[0] - groups[1]], axis=1),
            value_fields=np.concatenate([values[0], values[0] - values[1]], axis=-1),
            local_attention=arrays['local_attention'].astype(np.float32),
            indices=np.maximum(0, arrays['local_positions'] - arrays['query_positions'][0]),
            valid=arrays['local_valid'])
    return fields


def fit_normalization(capture, records, pack):
    """Fit complete-coordinate means/scales on fit sources only, never dev/test."""
    sums, squares = {}, {}
    count = 0
    for index, record in enumerate(records):
        if record['partition'] != 'fit':
            continue
        fields = read_fields(capture / record['id'])
        for name in ('node_fields', 'boundary_fields'):
            value = fields[name].astype(np.float64)
            sums[name] = sums.get(name, 0) + value.sum(axis=0)
            squares[name] = squares.get(name, 0) + np.square(value).sum(axis=0)
        count += len(fields['node_fields'])
        if index % 300 == 0:
            print(f'NORMALIZE answers={index + 1} rows={count}', flush=True)
    result = {}
    for name in sums:
        mean = sums[name] / count
        variance = np.maximum(0, squares[name] / count - mean ** 2)
        result[name + '_mean'] = mean.astype(np.float32)
        result[name + '_scale'] = np.maximum(np.sqrt(variance), .01).astype(np.float32)
    fit = ~pack['development']
    result['scalar_mean'] = pack['scalars'][fit].mean(0)
    result['scalar_scale'] = np.maximum(pack['scalars'][fit].std(0), .01)
    return result


def answer_batch(capture, record, pack, normalization, timing, device='cuda'):
    fields = read_fields(capture / record['id'])
    for name in ('node_fields', 'boundary_fields'):
        fields[name] = (fields[name] - normalization[name + '_mean']) / normalization[name + '_scale']
    region = slice(record['packed_start'], record['packed_stop'])
    shift = 1 if timing == 'posttoken' else 0
    targets = pack['target'][region] + shift
    scalars = np.zeros((len(fields['node_fields']), 8), dtype=np.float32)
    scalars[targets] = (pack['scalars'][region] - normalization['scalar_mean']) / normalization['scalar_scale']
    fields['scalars'] = scalars
    tensors = {name: torch.from_numpy(value).to(device) for name, value in fields.items()}
    return tensors, torch.from_numpy(targets).to(device), region


def source_loss_weights(pack):
    fit = ~pack['development']
    counts = np.bincount(pack['source_index'][fit])
    weights = np.zeros(len(pack['source_index']), dtype=np.float32)
    weights[fit] = len(weights[fit]) / (np.count_nonzero(counts) * counts[pack['source_index'][fit]])
    return weights
