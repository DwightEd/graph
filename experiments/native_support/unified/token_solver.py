"""Soft unit means and first-difference TV, solved jointly for a whole answer."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, spsolve

from .graph import unit_operator

SOLVER_TOLERANCE = 1e-9
MAX_ITERATIONS = 10000


def difference_operator(count):
    rows = np.repeat(np.arange(count - 1), 2)
    columns = np.column_stack((np.arange(count - 1), np.arange(1, count))).ravel()
    return sparse.csr_matrix((np.tile([-1., 1.], count - 1), (rows, columns)),
                             shape=(count - 1, count))


def quadratic_terms(observed, source, anchor, units, laplacian, anchor_strength, graph_strength):
    """Graph smoothness applies to z-source, preserving explicit source differences."""
    count = len(observed)
    means = unit_operator(units, count)
    lengths = np.array([unit["stop"] - unit["start"] for unit in units], dtype=float)
    prior = anchor_strength * means.T @ sparse.diags(lengths)
    precision = sparse.eye(count, format="csr") + prior @ means + graph_strength * laplacian
    right = observed + prior @ (means @ anchor) + graph_strength * (laplacian @ source)
    return precision.tocsc(), np.asarray(right)


def polish_plateaus(precision, right, difference, split, strength):
    """Solve the identified TV face exactly; numerical jitter must not rank a plateau."""
    groups = np.r_[0, np.cumsum(split != 0)]
    membership = sparse.csr_matrix((np.ones(len(groups)), (np.arange(len(groups)), groups)))
    subgradient = strength * np.sign(split)
    reduced = membership.T @ precision @ membership
    values = spsolve(reduced.tocsc(), membership.T @ (right - difference.T @ subgradient))
    score = np.asarray(membership @ np.atleast_1d(values))
    gradient = precision @ score - right
    dual = np.cumsum(gradient)[:-1]
    jumps = np.asarray(difference @ score)
    active = split != 0
    violation = max(float(abs(gradient.sum())),
        float(np.maximum(abs(dual) - strength, 0).max(initial=0)),
        float(abs(dual[active] - strength * np.sign(jumps[active])).max(initial=0)))
    if violation > 1e-7:
        raise RuntimeError(f"TV plateau solution failed optimality check: {violation}")
    return score, violation


def solve_tv(precision, right, strength):
    """ADMM with fixed rho=1; factorize once and update all token coordinates together."""
    difference = difference_operator(len(right))
    if strength == 0 or len(right) == 1:
        score = np.atleast_1d(spsolve(precision, right))
        return score, dict(iterations=0, optimality_max_error=float(abs(precision @ score - right).max()))
    solve = factorized((precision + difference.T @ difference).tocsc())
    score = np.asarray(right).copy()
    split = difference @ score
    dual = np.zeros_like(split)
    for iteration in range(1, MAX_ITERATIONS + 1):
        score = solve(right + difference.T @ (split - dual))
        jumps = difference @ score
        previous = split
        proposal = jumps + dual
        split = np.sign(proposal) * np.maximum(abs(proposal) - strength, 0)
        dual += jumps - split
        primal_error = float(abs(jumps - split).max())
        dual_error = float(abs(difference.T @ (split - previous)).max())
        if max(primal_error, dual_error) <= SOLVER_TOLERANCE:
            score, optimality = polish_plateaus(precision, right, difference, split, strength)
            return score, dict(iterations=iteration, optimality_max_error=optimality,
                               primal_max_error=primal_error, dual_max_error=dual_error)
    raise RuntimeError(f"Token TV solver did not converge in {MAX_ITERATIONS} iterations")


def solve_tokens(observed, source, anchor, units, laplacian, anchor_strength,
                 graph_strength, continuity_strength):
    precision, right = quadratic_terms(observed, source, anchor, units, laplacian,
                                       anchor_strength, graph_strength)
    score, diagnostic = solve_tv(precision, right, continuity_strength)
    means = unit_operator(units, len(score))
    diagnostic.update(unit_anchor_max_shift=float(abs(means @ (score - anchor)).max()),
                      token_observation_rmse=float(np.sqrt(np.mean((score - observed) ** 2))))
    return score, diagnostic
