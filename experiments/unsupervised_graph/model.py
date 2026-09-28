"""CoLA-inspired node/context contrast, not a faithful CoLA reproduction."""

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import IsolationForest

from experiments.probabilistic_detection.data import source_weights
from .data import NodeFeatures, matched_donors, neighbor_mean
from .scalar import fit_ranks, rank_scores, scalar_scores


class PairNetwork(nn.Module):
    def __init__(self, width=123):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 32))
        self.context = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 32))
        self.readout = nn.Linear(96, 1)

    def forward(self, nodes, neighbors):
        left, right = self.node(nodes), self.context(neighbors)
        pair = torch.cat((left * right, (left - right).abs(), left + right), dim=1)
        return self.readout(pair).squeeze(1)


def fit_contrast(matrix, neighbors, pack, epochs=12):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    network = PairNetwork(matrix.shape[1]).cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    nodes = torch.as_tensor(matrix, device="cuda")
    contexts = torch.as_tensor(neighbors, device="cuda")
    weights = source_weights(pack["source_index"])
    probability = weights / weights.sum()
    losses = []
    for epoch in range(epochs):
        donors = matched_donors(pack, rng)
        batches = rng.choice(len(matrix), size=(32, 4096), p=probability)
        total = 0.
        for indices in batches:
            selected = torch.as_tensor(indices, device="cuda")
            negative = torch.as_tensor(donors[indices], device="cuda")
            positive_logits = network(nodes[selected], contexts[selected])
            negative_logits = network(nodes[selected], contexts[negative])
            loss = .5 * (nn.functional.softplus(-positive_logits).mean()
                         + nn.functional.softplus(negative_logits).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        losses.append(total / len(batches))
        print(f"contrast epoch={epoch + 1} loss={losses[-1]:.5f}", flush=True)
    return {name: value.cpu() for name, value in network.state_dict().items()}, losses


def contrast_scores(state, matrix, neighbors):
    network = PairNetwork(matrix.shape[1]).cuda().eval()
    network.load_state_dict(state)
    result = []
    with torch.no_grad():
        for start in range(0, len(matrix), 8192):
            nodes = torch.as_tensor(matrix[start:start + 8192], device="cuda")
            context = torch.as_tensor(neighbors[start:start + 8192], device="cuda")
            result.append((-network(nodes, context)).cpu().numpy())
    return np.concatenate(result)


def fit_models(pack):
    rng = np.random.default_rng(42)
    weights = source_weights(pack["source_index"])
    reference = rng.choice(len(weights), min(200000, len(weights)), replace=True,
                           p=weights / weights.sum())
    transform = NodeFeatures().fit(pack["context"][reference], pack["observations"][reference])
    matrix = transform.transform(pack["context"], pack["observations"])
    forest = IsolationForest(n_estimators=150, max_samples=512, random_state=42,
                             n_jobs=4).fit(matrix[reference])
    models = dict(transform=transform, forest=forest)
    for name, shuffled in (("graph", False), ("shuffled", True)):
        neighbors = neighbor_mean(matrix, pack["answer_index"], shuffled=shuffled)
        models[name], models[f"{name}_loss"] = fit_contrast(matrix, neighbors, pack)
    return models


def raw_scores(models, pack):
    matrix = models["transform"].transform(pack["context"], pack["observations"])
    scores = scalar_scores(pack)
    scores["isolation"] = -models["forest"].score_samples(matrix)
    for name, shuffled in (("graph", False), ("shuffled", True)):
        neighbors = neighbor_mean(matrix, pack["answer_index"], shuffled=shuffled)
        scores[name] = contrast_scores(models[name], matrix, neighbors)
    return scores


def fusion_scores(scores):
    result = dict(scores)
    for anchor in ("local", "full", "pair"):
        for detail in ("route", "token_source", "isolation", "graph", "shuffled"):
            for weight in (.1, .25, .5):
                name = f"{anchor}+{weight:g}{detail}"
                result[name] = (1 - weight) * scores[anchor] + weight * scores[detail]
    return result
