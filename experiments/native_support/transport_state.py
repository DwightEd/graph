"""Offline Gaussian field on source/head budgets and observed reuse edges.

The field estimates a representation, not a truth state. Its mean minimizes
one half of the squared observation error plus strength / 2 times the weighted
sum of squared differences across edges. All tensor coordinates share the same
token graph; source, layer and head identities are never mixed.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .routes import EPS


def infer_budget(budget, edge_weight, strength=1.):
    """Smooth [token,layer,head,group] through strictly lower reuse edges.

    Inputs belong to one answer; weights and strength are nonnegative. Each
    coordinate has unit observation precision. Variance describes this working
    model, not correctness. Retention is the total neighbor coefficient in the
    conditional mean; it is not the full posterior's inherited-mass fraction.
    """
    values = np.asarray(budget, dtype=np.float64)
    edges = np.asarray(edge_weight, dtype=np.float64)
    weights = strength * (edges + edges.T)
    degree = weights.sum(axis=1)
    precision = np.diag(1. + degree) - weights
    factor = cho_factor(precision, lower=True)

    inferred = cho_solve(factor, values.reshape(len(values), -1))
    covariance = cho_solve(factor, np.eye(len(values)))
    return {
        "inferred_budget": inferred.reshape(values.shape),
        "retention_weight": degree / (1. + degree),
        "posterior_variance": np.diag(covariance).copy(),
    }


def budget_risk(budget, source_count):
    """Historical routing readout on pre-grouped per-edge message-norm budgets.

    Groups are source groups, route history, then remaining mass. For the
    evidence baseline, history includes answer self and answer special tokens;
    the first query's prompt self belongs to remaining mass. The caller must
    construct these roles from key positions, not reuse capture's B+4 roles.
    """
    source = budget[..., :source_count].sum(axis=-1)
    history = budget[..., source_count]
    difference = (history - source).sum(axis=-1)
    total = budget.sum(axis=(-1, -2))
    return (difference / np.maximum(total, EPS)).mean(axis=-1)
