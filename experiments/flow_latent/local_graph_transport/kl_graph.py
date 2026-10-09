"""Graph propagation with Bernoulli KL fidelity to each token's own score.

Scores are statistical risk coordinates, not calibrated truth probabilities.
Strict-past attention edges become a symmetric offline regularizer; this is
neither a directed causal propagation model nor a semantic boundary detector.
"""
import numpy as np
from scipy.linalg import solveh_banded
from scipy.special import expit, logit, rel_entr
import torch

from .smooth import normalize_edge_weights


def prepare_graph(edge_sources, edge_targets, raw_weights, node_count):
    """Normalize the original lag<=8 edges once and retain every token node."""
    sources = np.asarray(edge_sources, dtype=np.int64)
    targets = np.asarray(edge_targets, dtype=np.int64)
    raw_weights = np.asarray(raw_weights, dtype=np.float64)
    lags = targets - sources
    if np.any(lags <= 0) or np.any(lags > 8):
        raise ValueError('Graph edges require strict-past lags between 1 and 8')
    weights = normalize_edge_weights(torch.from_numpy(sources), torch.from_numpy(targets),
                                     torch.from_numpy(raw_weights), node_count).numpy()
    degree = np.zeros(node_count, dtype=np.float64)
    np.add.at(degree, sources, weights)
    np.add.at(degree, targets, weights)

    bandwidth = int(lags.max()) if len(lags) else 0
    laplacian = np.zeros((bandwidth + 1, node_count), dtype=np.float64)
    laplacian[0] = degree
    np.add.at(laplacian, (lags, sources), -weights)
    return dict(sources=sources, targets=targets, weights=weights,
                degree=degree, laplacian=laplacian)


def apply_laplacian(field, graph):
    """Accumulate each normalized undirected edge's signed endpoint effect."""
    difference = field[graph['targets']] - field[graph['sources']]
    weighted_difference = graph['weights'] * difference
    result = np.zeros_like(field)
    np.add.at(result, graph['targets'], weighted_difference)
    np.add.at(result, graph['sources'], -weighted_difference)
    return result


def kl_objective(field, unary, graph, gamma=2.):
    """Sum KL(Bern(field)||Bern(unary)) plus gamma/2 edge squared gaps."""
    fidelity = rel_entr(field, unary) + rel_entr(1 - field, 1 - unary)
    difference = field[graph['targets']] - field[graph['sources']]
    continuity = .5 * gamma * np.dot(graph['weights'], np.square(difference))
    return float(fidelity.sum() + continuity)


def kl_gradient(field, unary_logits, graph, gamma=2.):
    """Exact stationary equation: logit(field)-logit(unary)+gamma*L*field."""
    return logit(field) - unary_logits + gamma * apply_laplacian(field, graph)


def _newton_direction(field, gradient, graph, gamma):
    """Solve the SPD banded Hessian system without a dense token matrix."""
    hessian = gamma * graph['laplacian'].copy()
    hessian[0] += 1 / (field * (1 - field))
    return solveh_banded(hessian, -gradient, lower=True, check_finite=False)


def _interior_step(field, direction):
    """Find a strictly feasible Newton step; no score clipping is applied."""
    increasing = direction > 0
    decreasing = direction < 0
    limits = np.concatenate(((1 - field[increasing]) / direction[increasing],
                             -field[decreasing] / direction[decreasing]))
    boundary_step = float(limits.min()) if len(limits) else np.inf
    return min(1., .99 * boundary_step)


def kl_objective_change(field, candidate, gradient, graph, gamma=2.):
    """Compute exact delta-J without subtracting two nearly equal objectives.

    The KL Bregman identity gives delta-J=KL(candidate||field)+gradient*delta
    + gamma/2*delta*L*delta. log1p retains precision for tiny Newton updates.
    """
    change = candidate - field
    divergence = candidate * np.log1p(change / field)
    divergence += (1 - candidate) * np.log1p(-change / (1 - field))
    edge_change = change[graph['targets']] - change[graph['sources']]
    continuity = .5 * gamma * np.dot(graph['weights'], np.square(edge_change))
    return float(divergence.sum() + np.dot(gradient, change) + continuity)


def _damped_step(field, direction, gradient, graph, gamma):
    """Backtrack the convex objective while keeping each coordinate interior."""
    step = _interior_step(field, direction)
    for _ in range(60):
        candidate = field + step * direction
        objective_change = kl_objective_change(field, candidate, gradient, graph, gamma)
        derivative = float(np.dot(gradient, candidate - field))
        if objective_change <= 1e-4 * derivative:
            return candidate
        step *= .5
    raise RuntimeError('KL graph Newton line search did not decrease the objective')


def _solution_diagnostics(field, unary, graph, gradient, iterations, gamma):
    """Report stationarity, endpoints, and the per-node log-odds change bound."""
    logit_shift = logit(field) - logit(unary)
    return dict(solution=field, objective=kl_objective(field, unary, graph, gamma),
                max_gradient=float(np.max(np.abs(gradient))), converged=True,
                iterations=iterations, max_actual_change=float(np.max(np.abs(field - unary))),
                max_logit_shift=float(np.max(np.abs(logit_shift))),
                logit_shift=logit_shift, logit_shift_bound=gamma * graph['degree'],
                endpoint_bounds=np.array([unary.min(), unary.max(), field.min(), field.max()]),
                min_endpoint_distance=float(np.minimum(field, 1 - field).min()),
                degree=graph['degree'])


def kl_smooth_normalized(unary, graph, gamma=2., tolerance=1e-9, max_iterations=100):
    """Minimize strictly convex KL fidelity plus the original graph penalty.

    Strict interior unaries are required. Unsupported endpoint scores fail
    explicitly instead of silently introducing epsilon smoothing. At the
    midpoint, gamma=2 matches the old quadratic graph's lambda=.5 response.
    """
    unary = np.asarray(unary, dtype=np.float64)
    if np.any(unary <= 0) or np.any(unary >= 1):
        raise ValueError('KL graph fidelity requires strict interior unaries')
    if gamma < 0:
        raise ValueError('KL graph gamma must be nonnegative')
    unary_logits = logit(unary)
    field = unary.copy()
    for iteration in range(max_iterations + 1):
        gradient = kl_gradient(field, unary_logits, graph, gamma)
        if np.max(np.abs(gradient)) <= tolerance:
            return _solution_diagnostics(field, unary, graph, gradient, iteration, gamma)
        if iteration < max_iterations:
            direction = _newton_direction(field, gradient, graph, gamma)
            field = _damped_step(field, direction, gradient, graph, gamma)
    raise RuntimeError(f'KL graph did not converge after {max_iterations} iterations: '
                       f'max_gradient={np.max(np.abs(gradient)):.6g}')


def kl_smooth_field(unary, edge_sources, edge_targets, raw_weights, gamma=2.,
                    tolerance=1e-9, max_iterations=100):
    """Prepare the original raw graph once, then solve the full-node KL field."""
    graph = prepare_graph(edge_sources, edge_targets, raw_weights, len(unary))
    return kl_smooth_normalized(unary, graph, gamma, tolerance, max_iterations)


def quadratic_smooth_normalized(unary, graph, penalty=.5, coordinate='score'):
    """Exact score diffusion or the fixed logit-coordinate diffusion control."""
    unary = np.asarray(unary, dtype=np.float64)
    if coordinate not in ('score', 'logit'):
        raise ValueError('Quadratic coordinate must be score or logit')
    if coordinate == 'logit' and (np.any(unary <= 0) or np.any(unary >= 1)):
        raise ValueError('Logit graph control requires strict interior unaries')
    target = unary if coordinate == 'score' else logit(unary)
    system = penalty * graph['laplacian'].copy()
    system[0] += 1
    coordinate_solution = solveh_banded(system, target, lower=True, check_finite=False)
    gradient = coordinate_solution - target + penalty * apply_laplacian(coordinate_solution, graph)
    solution = coordinate_solution if coordinate == 'score' else expit(coordinate_solution)
    difference = coordinate_solution[graph['targets']] - coordinate_solution[graph['sources']]
    objective = .5 * np.square(coordinate_solution - target).sum()
    objective += .5 * penalty * np.dot(graph['weights'], np.square(difference))
    return dict(solution=solution, coordinate_solution=coordinate_solution,
                objective=float(objective), max_gradient=float(np.max(np.abs(gradient))),
                converged=True, iterations=1,
                max_actual_change=float(np.max(np.abs(solution - unary))), degree=graph['degree'])


def quadratic_smooth_field(unary, edge_sources, edge_targets, raw_weights,
                           penalty=.5, coordinate='score'):
    """Normalize raw edges once; retain all nodes before any validity filter."""
    graph = prepare_graph(edge_sources, edge_targets, raw_weights, len(unary))
    return quadratic_smooth_normalized(unary, graph, penalty, coordinate)
