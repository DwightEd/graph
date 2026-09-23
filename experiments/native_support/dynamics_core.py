"""Numerical primitives used by offline native-response dynamics."""

import numpy as np
from scipy.special import logsumexp, xlogy


def evidence_reads(attention, source_masks):
    """[layer,head,query,key] attention -> [layer,head,query,source] readings.

    Masks [source,key] are fixed from the prompt, not error annotations. Neither
    a distance increase nor a local-to-remote switch is required to retain a read.
    """
    profile = np.einsum("lhqj,bj->lhqb", attention, source_masks)
    change = np.full_like(profile, np.nan, dtype=float)
    change[..., 1:, :] = np.diff(profile, axis=-2)
    return {"read_mass": profile, "read_change": change}


def future_attention(attention):
    """Small-reference [layer,head,N,N] FAI; last position is right-censored.

    This intentionally uses later QUERY positions, not later prediction targets.
    Full square attention is only for the prototype, not a scalable collector.
    """
    count = attention.shape[-1]
    future = np.arange(count)[:, None] > np.arange(count)[None, :]
    mass = (attention * future).sum(-2)
    observations = future.sum(0)
    average = np.divide(mass, observations, out=np.full_like(mass, np.nan, dtype=float),
                        where=observations > 0)
    return {"future_mean_attention": average, "future_query_count": observations}


def source_moments(mass, effects):
    """[head,source] mass and [head,source,D] effects in one common coordinate frame.

    A source is an alternative in this working mixture, not a truth label. Heads
    remain separate; pooled entropy is only a diagnostic. Zero-mass heads are
    missing observations, never confident or uniform observations.
    """
    total = mass.sum(-1, keepdims=True)
    valid = total[:, 0] > 0
    probability = np.divide(mass, total, out=np.zeros_like(mass, dtype=float), where=total > 0)
    mean = np.einsum("hs,hsd->hd", probability, effects)
    centered = effects - mean[:, None, :]
    covariance = np.einsum("hs,hsd,hse->hde", probability, centered, centered)
    entropy = -xlogy(probability, probability).sum(-1)
    weights = valid / max(int(valid.sum()), 1)
    pooled = weights @ probability
    within = float(weights @ entropy)
    between = float(-xlogy(pooled, pooled).sum() - within)
    mean[~valid] = np.nan
    covariance[~valid] = np.nan
    entropy[~valid] = np.nan
    if not valid.any():
        within, between = np.nan, np.nan
    return {"probability": probability, "valid": valid, "mean": mean,
            "covariance": covariance, "entropy": entropy,
            "within_entropy": within, "between_head_js": between}


def effect_source_moments(effects):
    """Normalize responses ONCE; scale * mean recovers the sum for valid heads.

    Covariance describes alternative source DIRECTIONS under saliency weights,
    not stochastic sampling error of the known deterministic summed response.
    """
    mass = np.linalg.norm(effects, axis=-1)
    direction = np.divide(effects, mass[..., None], out=np.zeros_like(effects, dtype=float),
                          where=mass[..., None] > 0)
    result = source_moments(mass, direction)
    result["effect_scale"] = mass.sum(-1)
    result["net_effect"] = effects.sum(-2)
    return result


def predict_state(mean, covariance, transition, injection, control, control_covariance,
                  process_covariance, bias):
    """Independent state/control uncertainties; cross-covariances require extra terms."""
    predicted = transition @ mean + injection @ control + bias
    uncertainty = transition @ covariance @ transition.T
    uncertainty += injection @ control_covariance @ injection.T + process_covariance
    return predicted, uncertainty


def observe_state(mean, covariance, value, readout, noise):
    """Gaussian update and predictive log density; do not interpret NLL as truth."""
    innovation = value - readout @ mean
    innovation_covariance = readout @ covariance @ readout.T + noise
    factor = np.linalg.cholesky(innovation_covariance)
    whitened = np.linalg.solve(factor, innovation)
    gain = np.linalg.solve(innovation_covariance, readout @ covariance).T
    updated = mean + gain @ innovation
    remaining = np.eye(len(mean)) - gain @ readout
    updated_covariance = remaining @ covariance @ remaining.T + gain @ noise @ gain.T
    log_density = -.5 * (
        len(value) * np.log(2 * np.pi)
        + 2 * np.log(np.diag(factor)).sum() + whitened @ whitened
    )
    return updated, updated_covariance, float(log_density)


def transition_emissions(observed, history, controls, control_covariance,
                         transitions, injections, biases, noise):
    """Batched switching autoregression; states are OBSERVED, modes are latent.

    observed/history [T,D], controls [T,U], control_covariance [T,U,U];
    transitions [K,D,D], injections [K,D,U], biases [K,D], noise [K,D,D].
    Source/control covariance is a supplied modeling assumption, not gold labels.
    """
    mean = np.einsum("kij,tj->tki", transitions, history)
    mean += np.einsum("kiu,tu->tki", injections, controls) + biases
    covariance = noise[None] + np.einsum(
        "kiu,tuv,kjv->tkij", injections, control_covariance, injections
    )
    factor = np.linalg.cholesky(covariance)
    innovation = observed[:, None, :] - mean
    whitened = np.linalg.solve(factor, innovation[..., None])[..., 0]
    logdet = 2 * np.log(np.diagonal(factor, axis1=-2, axis2=-1)).sum(-1)
    return -.5 * (observed.shape[1] * np.log(2 * np.pi) + logdet + (whitened**2).sum(-1))


def mode_posteriors(log_emission, transition, initial):
    """Exact finite-state forward/backward recursion with supplied emissions.

    Filter uses prefix observations; smoother uses the whole response. This is
    NOT a complete switching Kalman filter: constructing mode emissions and
    fitting a continuous/discrete joint model remain separate work.
    """
    log_transition = np.full_like(transition, -np.inf, dtype=float)
    np.log(transition, out=log_transition, where=transition > 0)
    log_initial = np.full_like(initial, -np.inf, dtype=float)
    np.log(initial, out=log_initial, where=initial > 0)
    count = len(log_emission)
    forward = np.empty_like(log_emission, dtype=float)
    normalizers = np.empty(count)
    for position in range(count):
        prior = log_initial if position == 0 else logsumexp(
            forward[position - 1, :, None] + log_transition, axis=0
        )
        joint = prior + log_emission[position]
        normalizers[position] = logsumexp(joint)
        forward[position] = joint - normalizers[position]

    backward = np.zeros_like(forward)
    for position in range(count - 2, -1, -1):
        backward[position] = logsumexp(
            log_transition + log_emission[position + 1] + backward[position + 1], axis=1
        ) - normalizers[position + 1]
    joint = forward + backward
    log_smoothed = joint - logsumexp(joint, axis=1, keepdims=True)
    smoothed = np.exp(log_smoothed)
    return {"filtered": np.exp(forward), "smoothed": smoothed,
            "log_filtered": forward, "log_smoothed": log_smoothed,
            "log_likelihood": float(normalizers.sum())}
