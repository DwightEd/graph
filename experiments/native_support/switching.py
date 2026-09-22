"""Exact causal inference over segment lengths with correlated observations."""

import numpy as np
from scipy.special import gammaln, logsumexp

PRIOR_COUNT = 1.0


def prior_state(mean, covariance):
    dimension = len(mean)
    return {"count": np.array([PRIOR_COUNT]), "degrees": np.array([dimension + 2.]),
            "mean": np.asarray(mean)[None], "scale": np.asarray(covariance)[None]}


def predictive_logpdf(value, state):
    """NIW parameter integration gives a multivariate Student-t density."""
    dimension = len(value)
    count, degrees = state["count"], state["degrees"] - dimension + 1
    scale = state["scale"] * ((count + 1) / (count * degrees))[:, None, None]
    cholesky = np.linalg.cholesky(scale)
    whitened = np.linalg.solve(cholesky, (value - state["mean"])[..., None])[..., 0]
    squared = np.square(whitened).sum(-1)
    logdet = 2 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(-1)
    return (gammaln((degrees + dimension) / 2) - gammaln(degrees / 2)
            - (dimension * np.log(degrees * np.pi) + logdet) / 2
            - (degrees + dimension) * np.log1p(squared / degrees) / 2)


def update_state(value, state):
    count = state["count"]
    difference = value - state["mean"]
    outer = difference[..., :, None] * difference[..., None, :]
    return {"count": count + 1, "degrees": state["degrees"] + 1,
            "mean": state["mean"] + difference / (count + 1)[:, None],
            "scale": state["scale"] + outer * (count / (count + 1))[:, None, None]}


def posterior_moments(probability, state):
    means = state["mean"]
    dimension = means.shape[-1]
    route_variance = state["scale"][:, 0, 0] / (state["count"] * (state["degrees"] - dimension - 1))
    mean = probability @ means
    variance = probability @ (route_variance + means[:, 0] ** 2) - mean[0] ** 2
    return {"state_mean": mean, "state_route_sd": np.sqrt(max(float(variance), 0)),
            "reset_contribution": probability[0] * means[0, 0],
            "continuation_contribution": probability[1:] @ means[1:, 0]}


def switching_filter(observations, mean, covariance, expected_run=16):
    """A switch occurs BEFORE x_t, so posterior reset probability is data-dependent."""
    prior = prior_state(mean, covariance)
    state, log_probability = prior, np.zeros(1)
    posterior = np.zeros((len(observations), len(observations)))  # column n-1 = length n
    rows = []
    hazard = 1 / expected_run
    stay = np.log1p(-hazard) if hazard < 1 else -np.inf
    for target, value in enumerate(observations):
        if target:
            state = {name: np.concatenate((prior[name], state[name])) for name in prior}
            log_probability = np.r_[np.log(hazard), stay + log_probability]
        joint = log_probability + predictive_logpdf(value, state)
        log_normalizer = logsumexp(joint)
        log_probability = joint - log_normalizer
        probability = np.exp(log_probability)
        state = update_state(value, state)
        posterior[target, :len(probability)] = probability
        rows.append({**posterior_moments(probability, state),
                     "reset_probability": probability[0],
                     "expected_run_length": probability @ np.arange(1, len(probability) + 1),
                     "observation_nll": -log_normalizer})
    return {**{name: np.stack([row[name] for row in rows]) for name in rows[0]},
            "run_posterior": posterior}
