"""Exact banded Gaussian posterior and MAP EM with a fixed source observation."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import cholesky_banded, cho_solve_banded


def selected_inverse(cholesky):
    """Takahashi recursion: covariance entries within the Cholesky bandwidth."""
    bandwidth, count = cholesky.shape[0] - 1, cholesky.shape[1]
    covariance = np.zeros_like(cholesky)
    for index in range(count - 1, -1, -1):
        neighbors = np.arange(index + 1, min(count, index + bandwidth + 1))
        distance = np.abs(neighbors[:, None] - neighbors[None, :])
        column = np.minimum(neighbors[:, None], neighbors[None, :])
        block = covariance[distance, column]
        coefficients = cholesky[1:len(neighbors) + 1, index] / cholesky[0, index]
        off_diagonal = -coefficients @ block
        covariance[1:len(neighbors) + 1, index] = off_diagonal
        covariance[0, index] = 1 / cholesky[0, index] ** 2 - coefficients @ off_diagonal
    return covariance


def covariance_block(covariance, first, second, rank):
    rows = first * rank + np.arange(rank)
    columns = second * rank + np.arange(rank)
    return covariance[np.abs(rows[:, None] - columns[None, :]),
                      np.minimum(rows[:, None], columns[None, :])]


def residual_operator(edges, transition):
    """I-F, where each edge conditions a shared matrix acting on a latent parent."""
    count, neighbors, _ = edges.shape
    rank = transition.shape[1]
    row_parts, column_parts, values = [], [], []
    for lag in range(1, min(neighbors, count - 1) + 1):
        target = np.arange(lag, count)
        matrices = -np.einsum('tb,brs->trs', edges[target, lag - 1], transition)
        row = target[:, None, None] * rank + np.arange(rank)[None, :, None]
        column = (target - lag)[:, None, None] * rank + np.arange(rank)[None, None, :]
        row_parts.append(np.broadcast_to(row, matrices.shape).ravel())
        column_parts.append(np.broadcast_to(column, matrices.shape).ravel())
        values.append(matrices.ravel())
    dimension = count * rank
    if not values:
        return sparse.eye(dimension, format='csr')
    dependence = sparse.coo_matrix((np.concatenate(values),
        (np.concatenate(row_parts), np.concatenate(column_parts))), shape=(dimension, dimension))
    return sparse.eye(dimension, format='csr') + dependence.tocsr()


@dataclass
class Parameters:
    loading: np.ndarray
    transition: np.ndarray
    innovation: np.ndarray
    noise: np.ndarray
    regularization: float
    noise_floor: float = .02
    observation_weight: np.ndarray = None


def posterior(observations, edges, parameters, hide_anchor=False):
    count, width = observations.shape
    rank = parameters.loading.shape[1]
    weight = parameters.observation_weight
    if weight is None:
        weight = np.ones(width)
    inverse_noise = weight / parameters.noise
    if isinstance(hide_anchor, np.ndarray):
        hidden = hide_anchor
    else:
        hidden = np.full(count, hide_anchor, dtype=bool)
    emission_precision = parameters.loading.T @ (inverse_noise[:, None] * parameters.loading)
    operator = residual_operator(edges, parameters.transition)
    precision = operator.T @ sparse.diags(np.tile(1 / parameters.innovation, count)) @ operator
    precision += sparse.kron(sparse.eye(count), sparse.csr_matrix(emission_precision))
    if hidden.any():
        removed = np.zeros(count * rank)
        removed[np.flatnonzero(hidden) * rank] = inverse_noise[0]
        precision -= sparse.diags(removed)
    upper = sparse.tril(precision).tocoo()
    bandwidth = min(count * rank - 1, edges.shape[1] * rank + rank - 1)
    banded = np.zeros((bandwidth + 1, count * rank))
    banded[upper.row - upper.col, upper.col] = upper.data
    cholesky = cholesky_banded(banded, lower=True, check_finite=False)
    information = (observations * inverse_noise) @ parameters.loading
    information[hidden, 0] -= observations[hidden, 0] * inverse_noise[0]
    mean = cho_solve_banded((cholesky, True), information.ravel(), check_finite=False)
    covariance = selected_inverse(cholesky)
    logdet_precision = 2 * np.log(cholesky[0]).sum()
    quadratic = np.square(observations).sum(0) @ inverse_noise - information.ravel() @ mean
    quadratic -= np.square(observations[hidden, 0]).sum() * inverse_noise[0]
    normalization = count * (weight @ (np.log(2 * np.pi) + np.log(parameters.noise))
        + np.log(parameters.innovation).sum())
    normalization -= hidden.sum() * weight[0] * np.log(2 * np.pi * parameters.noise[0])
    loglikelihood = -.5 * (normalization
        + logdet_precision + quadratic)
    return mean.reshape(count, rank), covariance, float(loglikelihood)


def posterior_moments(observations, edges, parameters):
    mean, covariance, loglikelihood = posterior(observations, edges, parameters)
    rank = mean.shape[1]
    diagonal = np.stack([covariance_block(covariance, t, t, rank) for t in range(len(mean))])
    return mean, covariance, diagonal, loglikelihood


def transition_statistics(mean, covariance, edges):
    rank, basis = mean.shape[1], edges.shape[2]
    dimension = basis * rank
    second = np.zeros((dimension, dimension))
    cross = np.zeros((rank, dimension))
    for target in range(len(mean)):
        parents = target - np.arange(1, min(target, edges.shape[1]) + 1)
        if not len(parents):
            continue
        indices = (parents[:, None] * rank + np.arange(rank)).ravel()
        block = covariance[np.abs(indices[:, None] - indices[None, :]),
                           np.minimum(indices[:, None], indices[None, :])]
        mapping = np.kron(edges[target, :len(parents)].T, np.eye(rank))
        predictors = mapping @ mean[parents].ravel()
        second += mapping @ block @ mapping.T + np.outer(predictors, predictors)
        cross_cov = np.concatenate([
            covariance_block(covariance, target, parent, rank) for parent in parents], axis=1)
        cross += cross_cov @ mapping.T + np.outer(mean[target], predictors)
    return second, cross


def maximize(sequences, parameters, moments):
    rank = parameters.loading.shape[1]
    state_second = np.zeros((rank, rank))
    observation_cross = np.zeros_like(parameters.loading)
    observation_square = np.zeros(len(parameters.noise))
    feature_width = parameters.transition.shape[0] * rank
    predictor_second = np.zeros((feature_width, feature_width))
    state_predictor = np.zeros((rank, feature_width))
    count = 0
    for (observations, edges), (mean, covariance, diagonal, _) in zip(sequences, moments):
        state_second += mean.T @ mean + diagonal.sum(0)
        observation_cross += observations.T @ mean
        observation_square += np.square(observations).sum(0)
        second, cross = transition_statistics(mean, covariance, edges)
        predictor_second += second
        state_predictor += cross
        count += len(mean)
    penalty = parameters.regularization
    loading = np.linalg.solve(state_second + penalty * np.eye(rank), observation_cross.T).T
    loading[0] = np.eye(rank)[0]
    residual = observation_square - 2 * (loading * observation_cross).sum(1)
    residual += np.einsum('pr,rs,ps->p', loading, state_second, loading)
    residual[1:] += penalty * np.square(loading[1:]).sum(1)
    denominator = np.full(len(residual), count + rank, dtype=float)
    denominator[0] = count
    noise = np.maximum(residual / denominator, parameters.noise_floor)

    coefficient = np.linalg.solve(predictor_second + penalty * np.eye(feature_width), state_predictor.T).T
    transition_residual = np.diag(state_second) - 2 * (coefficient * state_predictor).sum(1)
    transition_residual += np.einsum('ri,ij,rj->r', coefficient, predictor_second, coefficient)
    transition_residual += penalty * np.square(coefficient).sum(1)
    innovation = np.maximum(transition_residual / (count + feature_width), parameters.noise_floor)
    transition = coefficient.reshape(rank, -1, rank).transpose(1, 0, 2)
    return Parameters(loading, transition, innovation, noise, penalty, parameters.noise_floor,
                      parameters.observation_weight)


def log_prior(parameters):
    loading = parameters.loading[1:]
    weight = parameters.observation_weight
    if weight is None:
        weight = np.ones(len(parameters.noise))
    prior = -.5 * (loading.shape[1] * (weight[1:] * np.log(parameters.noise[1:])).sum()
        + parameters.regularization * (weight[1:] * np.square(loading).sum(1) / parameters.noise[1:]).sum())
    coefficients = parameters.transition.transpose(1, 0, 2).reshape(len(parameters.innovation), -1)
    prior -= .5 * (coefficients.shape[1] * np.log(parameters.innovation).sum()
        + parameters.regularization * (np.square(coefficients).sum(1) / parameters.innovation).sum())
    return float(prior)


def fit(sequences, initial, iterations=40, tolerance=1e-5):
    parameters, history = initial, []
    for iteration in range(iterations + 1):
        moments = [posterior_moments(observations, edges, parameters) for observations, edges in sequences]
        objective = sum(moment[3] for moment in moments) + log_prior(parameters)
        history.append(objective)
        if iteration:
            difference = history[-1] - history[-2]
            assert difference >= -1e-7 * max(1, abs(history[-2])), 'MAP objective decreased'
            if difference <= tolerance * max(1, abs(history[-2])):
                break
        if iteration < iterations:
            parameters = maximize(sequences, parameters, moments)
    return parameters, dict(objective=history, iterations=len(history) - 1,
        converged=len(history) - 1 < iterations, exact_joint_posterior=True)
