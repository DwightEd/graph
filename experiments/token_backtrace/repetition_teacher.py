"""Frozen supervised readouts for mechanism diagnosis, never unsupervised scores."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


BASE = Path('outputs/cached_head_transport_20261007_v1')
TEACHERS = Path('outputs/cached_head_transport_20261007_v3')


class SelfReadout:
    """Exact self-only network with its constant normalized memory branch baked in."""

    def __init__(self, seed):
        saved = torch.load(TEACHERS / f'self_only_seed{seed}/model.pt', weights_only=False)
        state = saved['state']
        mean, scale = saved['node_normalization']
        assert np.all(mean.reshape(1024, 4)[:, 1:] == 0)
        self.mean = mean.reshape(1024, 4)[:, 0]
        self.scale = scale.reshape(1024, 4)[:, 0]
        self.weight = state['node.0.weight'][:, torch.arange(1024) * 4]
        self.bias = state['node.0.bias']
        memory_mean, memory_scale = saved['memory_normalization']
        memory = torch.tensor(-memory_mean / memory_scale)
        self.carried = F.gelu(F.linear(memory, state['message.0.weight'], state['message.0.bias']))
        self.gate_weight = state['gate.weight'][:, :128]
        self.gate_bias = state['gate.bias'] + F.linear(self.carried, state['gate.weight'][:, 128:])
        self.readout_weight = state['readout.weight']
        self.readout_bias = state['readout.bias']

    def standardized(self, row):
        values = row['node'].reshape(-1, 1024, 4)[..., 0]
        return torch.tensor((values - self.mean) / self.scale)

    def logits(self, values):
        receiver = F.gelu(F.linear(values, self.weight, self.bias))
        gate = torch.sigmoid(F.linear(receiver, self.gate_weight, self.gate_bias))
        return F.linear(receiver + gate * self.carried, self.readout_weight, self.readout_bias)[:, 0]

    def tangent(self):
        reference = torch.zeros(1, 1024, requires_grad=True)
        return torch.autograd.grad(self.logits(reference).sum(), reference)[0][0].detach().numpy()


def literal_repetition(token_ids):
    counts = np.zeros((len(token_ids), 3), dtype=np.int32)
    lag = np.zeros(len(token_ids), dtype=np.int32)
    tables = {size: {} for size in (1, 2, 4)}
    previous = {}
    for target, identity in enumerate(token_ids):
        if int(identity) in previous:
            lag[target] = target - previous[int(identity)]
        previous[int(identity)] = target
        for column, size in enumerate((1, 2, 4)):
            if target + 1 < size:
                continue
            phrase = tuple(token_ids[target + 1 - size:target + 1])
            counts[target, column] = tables[size].get(phrase, 0)
            tables[size][phrase] = counts[target, column] + 1
    return counts, lag


def different_word_persistence(values, token_ids):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    unit = values / np.maximum(norms, 1e-12)
    similarity = unit @ unit.T
    target = np.arange(len(values))[:, None]
    key = np.arange(len(values))[None]
    distinct = token_ids[:, None] != token_ids[None]
    result = {}
    for name, lower, upper in (('recent_pattern', 1, 8), ('remote_pattern', 9, 32)):
        eligible = distinct & (target - key >= lower) & (target - key <= upper)
        maximum = np.where(eligible, similarity, -np.inf).max(axis=1)
        maximum[~eligible.any(axis=1)] = np.nan
        result[name] = maximum
    return result


def prototype_banks(rows, teacher, limit=16, seed=42):
    generator = np.random.default_rng(seed)
    values, identities = {0: [], 1: []}, {0: [], 1: []}
    for row in rows:
        standardized = teacher.standardized(row).numpy()
        for label in (0, 1):
            indices = np.where(row['labels'] == label)[0]
            indices = generator.permutation(indices)[:limit]
            values[label].extend(standardized[indices])
            identities[label].extend(row['token_ids'][indices])
    count = min(len(v) for v in values.values())
    banks = {}
    for label in (0, 1):
        selected = generator.permutation(len(values[label]))[:count]
        features = np.asarray(values[label])[selected]
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        banks[label] = dict(values=features, token_ids=np.asarray(identities[label])[selected])
    return banks


def different_word_prototype(values, token_ids, banks):
    unit = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    similarities = {}
    for label, bank in banks.items():
        similarity = unit @ bank['values'].T
        similarity[token_ids[:, None] == bank['token_ids'][None]] = -np.inf
        best = np.partition(similarity, -3, axis=1)[:, -3:]
        similarities[label] = best.mean(axis=1)
    return similarities[1] - similarities[0]
