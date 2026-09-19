"""Two-state shared-covariance Gaussian HMM fitted only to natural TRAIN sequences."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans


@dataclass
class RegimeModel:
    means: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    logdet: np.ndarray
    initial: np.ndarray
    transition: np.ndarray
    occupancy: np.ndarray
    log_likelihood: float


def contiguous_segments(values, coverage, minimum=2):
    starts = np.flatnonzero(coverage & ~np.r_[False, coverage[:-1]])
    ends = np.flatnonzero(coverage & ~np.r_[coverage[1:], False]) + 1
    return [values[start:end] for start, end in zip(starts, ends)
            if end - start >= minimum]


def standardize(sequences, weights):
    count = sum(
        len(values) * weight
        for values, weight in zip(sequences, weights)
    )
    total = sum(
        (values.sum(axis=0) * weight
         for values, weight in zip(sequences, weights)),
        start=np.zeros_like(sequences[0][0], dtype=np.float64),
    )
    square = sum(
        (((values.astype(np.float64) ** 2).sum(axis=0)) * weight
         for values, weight in zip(sequences, weights)),
        start=np.zeros_like(sequences[0][0], dtype=np.float64),
    )
    center = total / count
    variance = np.maximum(square / count - center * center, 0.)
    scale = np.sqrt(variance)
    scale[scale < 1e-4] = 1.
    return center.astype(np.float32), scale.astype(np.float32)


def apply_standardization(values, center, scale):
    return (values - center) / scale


def block_logpdf(values, means, precision, logdet):
    states = means.shape[0]
    result = np.zeros((len(values), states), dtype=np.float64)
    constant = values.shape[2] * np.log(2 * np.pi)

    for state in range(states):
        residual = values - means[state]
        quadratic = np.einsum(
            "tld,ldk,tlk->tl",
            residual,
            precision,
            residual,
        )
        result[:, state] = -.5 * (
            quadratic + logdet[None, :] + constant
        ).sum(axis=1)
    return result


def forward_backward(log_emission, initial, transition):
    log_initial = np.log(initial)
    log_transition = np.log(transition)
    length = len(log_emission)

    forward = np.empty_like(log_emission)
    forward[0] = log_initial + log_emission[0]
    for token in range(1, length):
        forward[token] = (
            log_emission[token]
            + logsumexp(
                forward[token - 1][:, None] + log_transition,
                axis=0,
            )
        )

    backward = np.zeros_like(log_emission)
    for token in range(length - 2, -1, -1):
        backward[token] = logsumexp(
            log_transition
            + log_emission[token + 1][None, :]
            + backward[token + 1][None, :],
            axis=1,
        )

    log_likelihood = float(logsumexp(forward[-1]))
    gamma = np.exp(forward + backward - log_likelihood)

    xi_sum = np.zeros((2, 2), dtype=np.float64)
    for token in range(length - 1):
        log_xi = (
            forward[token][:, None]
            + log_transition
            + log_emission[token + 1][None, :]
            + backward[token + 1][None, :]
            - log_likelihood
        )
        xi_sum += np.exp(log_xi)

    return gamma, xi_sum, log_likelihood


def initial_means(sequences, seed):
    random = np.random.default_rng(seed)
    budget = 10000
    per_sequence = max(1, budget // len(sequences))
    rows = []
    for values in sequences:
        count = min(len(values), per_sequence)
        selected = random.choice(len(values), count, replace=False)
        rows.append(values[selected])
    sampled = np.concatenate(rows, axis=0)
    if len(sampled) > budget:
        sampled = sampled[random.choice(len(sampled), budget, replace=False)]
    flat = sampled.reshape(len(sampled), -1)

    labels = MiniBatchKMeans(
        n_clusters=2,
        random_state=seed,
        batch_size=512,
        n_init=5,
    ).fit(flat)

    centers = labels.cluster_centers_.reshape(
        2,
        sequences[0].shape[1],
        sequences[0].shape[2],
    )
    return centers.astype(np.float64)


def shared_covariance(sequences, gammas, weights, means, ridge):
    layers = means.shape[1]
    width = means.shape[2]
    covariance = np.zeros((layers, width, width), dtype=np.float64)
    total = 0.

    for values, gamma, sequence_weight in zip(sequences, gammas, weights):
        for state in range(2):
            residual = values - means[state]
            state_weight = gamma[:, state] * sequence_weight
            covariance += np.einsum(
                "t,tld,tle->lde",
                state_weight,
                residual,
                residual,
            )
            total += state_weight.sum()

    covariance /= max(total, 1.)
    for layer in range(layers):
        scale = np.trace(covariance[layer]) / width
        covariance[layer].flat[::width + 1] += ridge * max(scale, 1e-6)

    precision = np.linalg.inv(covariance)
    _, logdet = np.linalg.slogdet(covariance)
    return covariance, precision, logdet


def em_fit(sequences, weights, args, seed):
    means = initial_means(sequences, seed)
    layers = means.shape[1]
    width = means.shape[2]
    covariance = np.repeat(
        np.eye(width, dtype=np.float64)[None, :, :],
        layers,
        axis=0,
    )
    precision = covariance.copy()
    logdet = np.zeros(layers)
    initial = np.array([.5, .5], dtype=np.float64)
    transition = np.array([[.95, .05], [.05, .95]], dtype=np.float64)

    last_likelihood = -np.inf
    occupancy = np.ones(2)
    for iteration in range(args.iterations):
        gammas = []
        xi_sum = np.zeros((2, 2), dtype=np.float64)
        initial_sum = np.zeros(2, dtype=np.float64)
        occupancy = np.zeros(2, dtype=np.float64)
        weighted_sum = np.zeros_like(means)
        likelihood = 0.

        for values, sequence_weight in zip(sequences, weights):
            emission = block_logpdf(values, means, precision, logdet)
            gamma, xi, current = forward_backward(
                emission,
                initial,
                transition,
            )
            gammas.append(gamma)
            xi_sum += xi * sequence_weight
            initial_sum += gamma[0] * sequence_weight
            occupancy += gamma.sum(axis=0) * sequence_weight
            weighted_sum += np.einsum(
                "ts,tld->sld",
                gamma * sequence_weight,
                values,
            )
            likelihood += current * sequence_weight

        means = weighted_sum / occupancy[:, None, None]
        covariance, precision, logdet = shared_covariance(
            sequences,
            gammas,
            weights,
            means,
            args.ridge,
        )

        initial = (initial_sum + 1.) / (initial_sum.sum() + 2.)
        counts = xi_sum + 1.
        counts.flat[::3] += args.sticky_prior
        transition = counts / counts.sum(axis=1, keepdims=True)

        print(
            f"latent-regime seed={seed} iter={iteration + 1}/{args.iterations} "
            f"loglik={likelihood:.3f} occupancy={occupancy.tolist()}",
            flush=True,
        )
        if abs(likelihood - last_likelihood) < args.tolerance:
            break
        last_likelihood = likelihood

    occupancy /= occupancy.sum()
    return RegimeModel(
        means,
        covariance,
        precision,
        logdet,
        initial,
        transition,
        occupancy,
        likelihood,
    )


def orient_rare_state(model):
    if model.occupancy[1] <= model.occupancy[0]:
        return model

    order = np.array([1, 0])
    return RegimeModel(
        means=model.means[order],
        covariance=model.covariance,
        precision=model.precision,
        logdet=model.logdet,
        initial=model.initial[order],
        transition=model.transition[np.ix_(order, order)],
        occupancy=model.occupancy[order],
        log_likelihood=model.log_likelihood,
    )


def fit_best(sequences, weights, args):
    models = [
        orient_rare_state(
            em_fit(sequences, weights, args, args.seed + start)
        )
        for start in range(args.starts)
    ]
    return max(models, key=lambda model: model.log_likelihood)


def emission_score(values, model):
    emission = block_logpdf(
        values,
        model.means,
        model.precision,
        model.logdet,
    )
    return emission[:, 1] - emission[:, 0]


def iid_score(values, model):
    score = emission_score(values, model)
    prior = np.log(model.occupancy[1]) - np.log(model.occupancy[0])
    return score + prior


def filtered_score(values, model):
    emission = block_logpdf(
        values,
        model.means,
        model.precision,
        model.logdet,
    )
    alpha = np.empty_like(emission)
    alpha[0] = np.log(model.initial) + emission[0]
    alpha[0] -= logsumexp(alpha[0])

    log_transition = np.log(model.transition)
    for token in range(1, len(values)):
        alpha[token] = (
            emission[token]
            + logsumexp(alpha[token - 1][:, None] + log_transition, axis=0)
        )
        alpha[token] -= logsumexp(alpha[token])

    return alpha[:, 1] - alpha[:, 0]
