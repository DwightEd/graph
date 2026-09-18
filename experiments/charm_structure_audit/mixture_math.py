"""Two tied Gaussians in all input dimensions; no labels or CHARM parameters.

A fixed invertible whitening makes the tied covariance a rank-one update.
This is the ordinary two-component covariance update, not a rank-one input
projection. All original channels are retained. See MIXTURE.md for derivation.
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit, logsumexp
from tqdm import tqdm


RESPONSIBILITY_FLOOR = 1e-8


def prepare_coordinates(values, ridge=1e-3):
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (values - center) / scale
    covariance = standardized.T @ standardized / len(values)
    covariance.flat[::len(scale) + 1] += ridge
    factor = np.linalg.cholesky(covariance)
    coordinates = solve_triangular(factor, standardized.T, lower=True).T
    transform = dict(center=center, scale=scale, factor=factor)
    return coordinates, transform


def transform_values(values, transform):
    standardized = (values - transform['center']) / transform['scale']
    return solve_triangular(transform['factor'], standardized.T, lower=True).T


def estimate_components(coordinates, responsibility):
    """The mixture mean equals the FIT sample mean (zero in these coordinates)."""
    weight = float(responsibility.mean())
    between_weight = weight * (1 - weight)
    difference = coordinates.T @ (responsibility - weight) / (len(coordinates) * between_weight)
    squared_length = float(difference @ difference)
    denominator = 1 - between_weight * squared_length
    coefficient = difference / denominator
    intercept = np.log(weight / (1 - weight)) - .5 * (1 - 2 * weight) * squared_length / denominator
    return dict(weight=weight, difference=difference, denominator=denominator,
                coefficient=coefficient, intercept=intercept)


def log_densities(coordinates, state, transform):
    """Return densities in original x units, including the fixed Jacobian."""
    log_determinant = 2 * np.log(np.diag(transform['factor'])).sum()
    log_determinant += 2 * np.log(transform['scale']).sum()
    base = -.5 * (coordinates.shape[1] * np.log(2 * np.pi) + log_determinant)
    base -= .5 * np.square(coordinates).sum(axis=1)
    weight = state['weight']
    difference = state['difference']
    denominator = state['denominator']
    projection = coordinates @ difference
    squared_length = difference @ difference
    common = base - .5 * np.log(denominator)
    common -= .5 * weight * (1 - weight) * projection ** 2 / denominator
    component_terms = []
    for prior, offset in ((1 - weight, -weight), (weight, 1 - weight)):
        term = np.log(prior) + offset * projection / denominator
        term -= .5 * offset ** 2 * squared_length / denominator
        component_terms.append(term)
    return base, common + logsumexp(np.stack(component_terms), axis=0)


def fit_two(coordinates, transform, initial, iterations=200, tolerance=1e-5, progress=False):
    """EM responsibilities, with a fixed covariance ridge; disclose convergence."""
    responsibility = .01 + .98 * np.asarray(initial, float)
    history = []
    for iteration in tqdm(range(iterations), desc="mixture EM", disable=not progress, leave=False):
        state = estimate_components(coordinates, responsibility)
        updated = expit(coordinates @ state['coefficient'] + state['intercept'])
        updated = np.clip(updated, RESPONSIBILITY_FLOOR, 1 - RESPONSIBILITY_FLOOR)
        change = float(np.max(abs(updated - responsibility)))
        _, density = log_densities(coordinates, state, transform)
        history.append(dict(iteration=iteration + 1, log_density=float(density.mean()),
                            responsibility_change=change, weight=state['weight']))
        responsibility = updated
        if change < tolerance:
            break
    state = estimate_components(coordinates, responsibility)
    state = canonical_order(state, transform)
    state['converged'] = change < tolerance
    state['iterations'] = len(history)
    return state, history


def raw_parameters(state, transform):
    """Map coefficient, means and covariance back to ORIGINAL layer/head units."""
    scale, factor = transform['scale'], transform['factor']
    difference = factor @ state['difference']
    weight = state['weight']
    coefficient = solve_triangular(factor.T, state['coefficient'], lower=False) / scale
    intercept = state['intercept'] - transform['center'] @ coefficient
    means = transform['center'] + np.outer([-weight, 1 - weight], scale * difference)
    covariance = factor @ factor.T - weight * (1 - weight) * np.outer(difference, difference)
    covariance = scale[:, None] * covariance * scale[None, :]
    return dict(coefficient=coefficient, intercept=intercept, means=means,
                covariance=covariance, priors=np.array([1 - weight, weight]))


def canonical_order(state, transform):
    """Largest absolute raw coefficient is positive. Never orient by gold/P."""
    raw = raw_parameters(state, transform)['coefficient']
    if raw[np.argmax(abs(raw))] < 0:
        state = dict(state, weight=1 - state['weight'], difference=-state['difference'],
                     coefficient=-state['coefficient'], intercept=-state['intercept'])
    return state
