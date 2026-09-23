"""Finite-bank meaning/form readouts; no truth labels or routing mixtures."""

import numpy as np
from scipy.special import logsumexp, softmax, xlogy


def entropy(probability):
    return -xlogy(probability, probability).sum(axis=-1)


def group_distribution(log_scores, groups):
    """Normalize a declared sequence bank, not all possible LM meanings."""
    groups = np.asarray(groups)
    log_scores = np.asarray(log_scores, dtype=float)
    probability = softmax(log_scores)
    group_ids = np.unique(groups)
    mass = np.array([probability[groups == group].sum() for group in group_ids])
    within = np.array([entropy(softmax(log_scores[groups == group])) for group in group_ids])
    return probability, mass, within


def entropy_partition(log_scores, groups):
    probability, mass, within = group_distribution(log_scores, groups)
    meaning = entropy(mass)
    expression = mass @ within
    total = entropy(probability)
    return dict(candidate_entropy=float(total), meaning_entropy=float(meaning),
                expression_entropy=float(expression),
                entropy_chain_error=float(total - meaning - expression))


def distribution_change(before, after, groups):
    """Exact KL chain rule on the same finite bank and fixed meaning partition."""
    groups = np.asarray(groups)
    log_before = before - logsumexp(before)
    log_after = after - logsumexp(after)
    probability = np.exp(log_after)
    total = probability @ (log_after - log_before)
    meaning, expression = 0., 0.
    for group in np.unique(groups):
        selected = groups == group
        previous = logsumexp(log_before[selected])
        current = logsumexp(log_after[selected])
        mass = np.exp(current)
        meaning += mass * (current - previous)
        expression += probability[selected] @ (
            log_after[selected] - current - log_before[selected] + previous)
    return dict(candidate_kl=float(total), meaning_kl=float(meaning),
                expression_kl=float(expression), kl_chain_error=float(total - meaning - expression))


def contrast_weights(log_scores, groups, positive, negative):
    """Derivative of a group log-odds with respect to candidate log scores."""
    weights = np.zeros(len(log_scores))
    for group, sign in ((positive, 1.), (negative, -1.)):
        selected = np.asarray(groups) == group
        weights[selected] = sign * softmax(np.asarray(log_scores)[selected])
    return weights


def contrast_effects(weights, candidates):
    """Candidate arrays must refer to the SAME observed prefix coordinates."""
    return np.tensordot(weights, np.stack(candidates), axes=(0, 0))
