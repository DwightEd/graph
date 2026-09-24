"""Parallel physical-head effects become dependence penalties, never truth labels."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def unit_operator(units, count):
    rows = np.concatenate([np.full(unit["stop"] - unit["start"], index)
                           for index, unit in enumerate(units)])
    weights = np.concatenate([np.full(unit["stop"] - unit["start"], 1 / (unit["stop"] - unit["start"]))
                              for unit in units])
    return sparse.csr_matrix((weights, (rows, np.arange(count))), shape=(len(units), count))


def dependence_laplacian(edges, count, random_endpoints=False):
    """Keep signed per-head effects on disk; couple their absolute source interactions.

    One nat is a fixed regularization scale. Weak effects are not normalized to one.
    The sham control keeps each weight, changing only its cached matched endpoint.
    """
    target = edges["target"]
    key = edges["sham_key"] if random_endpoints else edges["key"]
    magnitude = abs(edges["effect_with_source"] - edges["effect_without_source"])
    total = np.bincount(target, weights=magnitude, minlength=count)
    weight = magnitude / (1 + total[target])
    directed = sparse.csr_matrix((weight, (target, key)), shape=(count, count))
    symmetric = .5 * (directed + directed.T)
    degree = np.asarray(symmetric.sum(axis=1)).ravel()
    scaling = sparse.diags(1 / np.sqrt(np.maximum(degree, 1)))
    adjacency = (scaling @ symmetric @ scaling).tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    return laplacian.tocsr(), adjacency


def solve_residual(route, units, laplacian, strength):
    """Minimize ||delta-centered_route||² + strength*delta.T L delta, B delta=0.

    Unit evidence anchors remain unchanged. Smoothness applies to routing deviations,
    not a claim that causally connected words have identical factual correctness.
    """
    count = len(route)
    operator = unit_operator(units, count)
    unit_ids = np.concatenate([np.full(unit["stop"] - unit["start"], index)
                               for index, unit in enumerate(units)])
    centered = route - np.asarray(operator @ route)[unit_ids]
    precision = sparse.eye(count, format="csr") + strength * laplacian
    system = sparse.bmat([[precision, operator.T], [operator, None]], format="csc")
    right = np.r_[centered, np.zeros(len(units))]
    solution = spsolve(system, right)
    residual = solution[:count]
    diagnostics = dict(constraint_max_error=float(abs(operator @ residual).max()),
        stationarity_max_error=float(abs(system @ solution - right).max()),
        route_residual_norm=float(np.linalg.norm(centered)),
        smoothed_residual_norm=float(np.linalg.norm(residual)))
    return residual, centered, diagnostics
