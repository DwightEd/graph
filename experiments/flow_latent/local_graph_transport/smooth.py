"""Huber continuity prior for an externally supplied semantic scalar field.

This optimizer cannot create a truth direction. Directed token edges enter a
symmetric penalty, not causal error propagation or an exact CORTEX HMM.
"""
import torch


def normalize_edge_weights(edge_sources, edge_targets, raw_weights, node_count):
    """Budget each node's incident weight by at most one.

    Edges are [edge] integer sender j and receiver t, with j<t. Raw weights
    are nonnegative [edge], already made dimensionless using an independent
    reference scale. Divide by max(1, raw_degree_j, raw_degree_t); weak edges
    are not amplified. CPU scientific checks use float64 weights.
    """
    if (raw_weights < 0).any():
        raise ValueError('Huber continuity requires nonnegative edge weights')
    degree = torch.zeros(node_count, dtype=raw_weights.dtype, device=raw_weights.device)
    degree.index_add_(0, edge_sources, raw_weights)
    degree.index_add_(0, edge_targets, raw_weights)
    denominator = torch.maximum(degree[edge_sources], degree[edge_targets]).clamp_min(1.)
    return raw_weights / denominator


def huber_graph_objective(field, unary, edge_sources, edge_targets, weights,
                          penalty=.5, huber_delta=1.):
    """Return the scalar objective; weights are already normalized.

    J=0.5*||field-unary||^2 + penalty*sum(w*Huber_delta(field_t-field_j)).
    No grouping of token values into punctuation spans occurs.
    """
    difference = field[edge_targets] - field[edge_sources]
    magnitude = difference.abs()
    huber = torch.where(magnitude <= huber_delta, .5 * difference.square(),
                        huber_delta * (magnitude - .5 * huber_delta))
    return .5 * (field - unary).square().sum() + penalty * (weights * huber).sum()


def _objective_gradient(field, unary, edge_sources, edge_targets, weights, penalty, huber_delta):
    """Accumulate signed endpoint gradients without a dense incidence matrix."""
    difference = field[edge_targets] - field[edge_sources]
    edge_gradient = penalty * weights * difference.clamp(-huber_delta, huber_delta)
    gradient = field - unary
    gradient.index_add_(0, edge_targets, edge_gradient)
    gradient.index_add_(0, edge_sources, -edge_gradient)
    return gradient


def smooth_field(unary, edge_sources, edge_targets, raw_weights, penalty=.5,
                 huber_delta=1., tolerance=1e-8, max_iterations=1000):
    """Optimize a CPU float64 scalar field; preserve every token's own unary.

    Normalized incident degrees<=1 imply gradient Lipschitz bound 1+2*penalty.
    Fixed step=1/(1+2*penalty); strong convexity comes from the unary term.
    Returns convergence diagnostics, including max change and penalty*delta
    bound. Failed convergence raises an explicit RuntimeError.
    """
    if penalty < 0 or huber_delta <= 0:
        raise ValueError('penalty must be nonnegative and huber_delta positive')
    if (edge_sources >= edge_targets).any():
        raise ValueError('Token graph edges require strict-past sender j < receiver t')
    weights = normalize_edge_weights(edge_sources, edge_targets, raw_weights, len(unary))
    step = 1 / (1 + 2 * penalty)
    field = unary.clone()
    for iteration in range(max_iterations + 1):
        gradient = _objective_gradient(field, unary, edge_sources, edge_targets,
                                       weights, penalty, huber_delta)
        max_gradient = float(gradient.abs().max())
        if max_gradient <= tolerance:
            objective = huber_graph_objective(field, unary, edge_sources, edge_targets,
                                              weights, penalty, huber_delta)
            return dict(solution=field, objective=float(objective), max_gradient=max_gradient,
                        converged=True, iterations=iteration,
                        max_actual_change=float((field - unary).abs().max()),
                        lambda_delta_bound=penalty * huber_delta)
        if iteration < max_iterations:
            field = field - step * gradient
    raise RuntimeError(f'Huber graph smoothing did not converge after {max_iterations} iterations: '
                       f'max_gradient={max_gradient:.6g}, tolerance={tolerance:.6g}')


def filter_prefix_field(unary, edge_sources, edge_targets, raw_weights, penalty=.5,
                        huber_delta=1., tolerance=1e-8, max_iterations=1000):
    """Emit r_t from prefix<=t only; never retrospectively revise prior outputs.

    Recompute degrees from eligible raw edges at every prefix. Unknown future
    edge weights cannot rescale the past. Future unaries are never optimized
    for an earlier emitted value. This is prefix filtering, not HMM inference.
    """
    if (edge_sources >= edge_targets).any():
        raise ValueError('Token graph edges require strict-past sender j < receiver t')
    filtered = torch.empty_like(unary)
    for target in range(len(unary)):
        eligible = edge_targets <= target
        result = smooth_field(unary[:target + 1], edge_sources[eligible], edge_targets[eligible],
                              raw_weights[eligible], penalty, huber_delta, tolerance, max_iterations)
        filtered[target] = result['solution'][-1]
    return filtered
