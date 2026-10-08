"""Full-coordinate candidate readout with explicit adjacent-state interactions."""
import math

import numpy as np
import torch
from torch import nn


class OrderedCompatibility(nn.Module):
    def __init__(self, window=8, layers=32, sites=4, width=4096):
        super().__init__()
        self.coordinate = nn.Parameter(torch.ones(width))
        self.level = nn.Parameter(torch.full((window, layers, sites), 1 / (window * layers * sites)))
        self.transition = nn.Parameter(torch.zeros(window - 1, layers, sites))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, states, candidates):
        # [batch,time,layer,site,coordinate]; no temporal/head average input.
        level = torch.einsum('btlsd,tls->bd', states, self.level)
        neighbors = states[:, :-1] * states[:, 1:]
        transition = torch.einsum('btlsd,tls->bd', neighbors, self.transition)
        vector = (level + transition) * self.coordinate
        return torch.einsum('bd,bcd->bc', vector, candidates) / math.sqrt(states.shape[-1]) + self.bias


class WideCurrentCompatibility(nn.Module):
    """8449 parameters; two independent coordinate maps at current row."""
    def __init__(self, layers=32, sites=4, width=4096):
        super().__init__()
        self.coordinate_linear = nn.Parameter(torch.ones(width))
        self.coordinate_square = nn.Parameter(torch.ones(width))
        self.level = nn.Parameter(torch.full((layers, sites), 1 / (layers * sites)))
        self.square = nn.Parameter(torch.zeros(layers, sites))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, states, candidates):
        current = states[:, -1]
        level = torch.einsum('blsd,ls->bd', current, self.level) * self.coordinate_linear
        square = torch.einsum('blsd,ls->bd', current.square(), self.square) * self.coordinate_square
        return torch.einsum('bd,bcd->bc', level + square, candidates) / math.sqrt(current.shape[-1]) + self.bias


def fit_statistics(states, fit_indices):
    """Fit-only per-site/coordinate statistics, shared over the lag axis."""
    shape = states.shape[2:]
    total = np.zeros(shape, dtype=np.float64)
    squared = np.zeros(shape, dtype=np.float64)
    count = 0
    for index in fit_indices:
        values = np.asarray(states[index], dtype=np.float64)
        total += values.sum(axis=0)
        squared += (values * values).sum(axis=0)
        count += len(values)
    mean = total / count
    variance = np.maximum(squared / count - mean * mean, 0)
    scale = np.maximum(np.sqrt(variance), 1e-3)
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(scale.astype(np.float32))


def transform_view(states, variant, permutations=None):
    if variant == 'single':
        return states[:, -1:].expand_as(states)
    if variant == 'mean':
        return states.mean(dim=1, keepdim=True).expand_as(states)
    if variant == 'shuffled':
        batch = torch.arange(len(states), device=states.device)[:, None]
        return states[batch, permutations]
    if variant == 'drop_source':
        values = states.clone()
        values[:, :, :, 3] = 0
        return values
    return states


def source_loss(scores, truth, source_ids, model, ridge=1e-4):
    labels = torch.zeros_like(scores).scatter_(1, truth[:, None], 1.)
    binary = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction='none').mean(1)
    right = scores.gather(1, truth[:, None]).squeeze(1)
    wrong = scores.gather(1, (1 - truth)[:, None]).squeeze(1)
    loss = binary + torch.nn.functional.softplus(wrong - right)
    balanced = torch.stack([loss[source_ids == identity].mean() for identity in source_ids.unique()]).mean()
    penalty = sum(weight.square().mean() for name, weight in model.named_parameters() if name != 'bias')
    return balanced + ridge * penalty


def row_permutations(count, window, seed=73):
    generator = np.random.default_rng(seed)
    values = np.tile(np.arange(window), (count, 1))
    for row in values:
        row[:-1] = generator.permutation(window - 1)
    return torch.tensor(values)
