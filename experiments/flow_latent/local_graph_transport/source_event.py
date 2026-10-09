"""Finite signed source contrast for the observed token versus all other tokens.

Uniform local/full probability mixing defines an artificial observer view
distribution. Its Bernoulli Fisher coordinate is neither a factual probability
nor an attribution of internal causal information flow. No reference is fitted
here, and rounded cached logp=0 does not recover the true complement mass.
"""
import numpy as np


def uniform_view_logmix(local_logp, full_logp):
    """Stable logaddexp(local,full)-log(2), including exact zero endpoints.

    For nonpositive inputs both terms below are nonpositive. log1p/expm1
    preserve tiny complement mass when one view is zero and the other is near
    zero, without clipping an invalid positive log probability.
    """
    largest = np.maximum(local_logp, full_logp)
    gap = np.abs(local_logp - full_logp)
    return largest + np.log1p(.5 * np.expm1(-gap))


def bernoulli_fisher_angle(logp):
    """Return theta=2*atan2(sqrt(p),sqrt(1-p)) in [0,pi], without epsilon."""
    event_root = np.exp(.5 * logp)
    complement_root = np.sqrt(-np.expm1(logp))
    return 2 * np.arctan2(event_root, complement_root)


def probability_difference(present_logp, absent_logp):
    """Compute p_absent-p_present without subtraction near probability one.

    Values below float64 probability resolution can still underflow to zero;
    no artificial nonzero sign or complement probability is introduced.
    """
    logp_difference = absent_logp - present_logp
    largest = np.maximum(present_logp, absent_logp)
    magnitude = np.exp(largest) * -np.expm1(-np.abs(logp_difference))
    return np.sign(logp_difference) * magnitude


def source_event_effects(present_local, present_full, absent_local, absent_full):
    """Read paired native logp arrays; positive effect means source suppression.

    Each argument has the same token shape. The primary effect is the signed
    Fisher coordinate difference after uniform view mixing. Logp/probability
    differences are fixed controls with the same mathematical direction.
    """
    logp = np.stack([present_local, present_full, absent_local, absent_full]).astype(np.float64)
    if not np.isfinite(logp).all() or np.any(logp > 0):
        raise ValueError('Paired native log probabilities must be finite and nonpositive')
    present_logmix = uniform_view_logmix(logp[0], logp[1])
    absent_logmix = uniform_view_logmix(logp[2], logp[3])
    present_theta = bernoulli_fisher_angle(present_logmix)
    absent_theta = bernoulli_fisher_angle(absent_logmix)
    return dict(fisher_effect=absent_theta - present_theta,
                logp_effect=absent_logmix - present_logmix,
                probability_effect=probability_difference(present_logmix, absent_logmix),
                present_logmix=present_logmix, absent_logmix=absent_logmix,
                present_probability=np.exp(present_logmix),
                absent_probability=np.exp(absent_logmix),
                present_theta=present_theta, absent_theta=absent_theta)
