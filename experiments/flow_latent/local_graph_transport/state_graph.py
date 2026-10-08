"""Exact finite-bandwidth conditional binary states, not calibrated truth.

Historical directed attention edges supply symmetric Potts compatibility. The
last eight bits are a DP frontier; ordinary two-state HMM inference would lose
nonadjacent factors. All returned marginals use the complete answer offline.
"""
import numpy as np
from scipy.special import logsumexp
import torch

from .smooth import normalize_edge_weights


EPSILON = 1e-12


def normalized_lag_weights(weights, normalize=True):
    """Validate lag edges; by default reuse the scalar graph's incident budget."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] > 8:
        raise ValueError('Weights must have shape [token, lag<=8]')
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('State compatibility weights must be finite and nonnegative')
    targets, slots = np.indices(weights.shape)
    senders = targets - slots - 1
    valid = senders >= 0
    if np.any(weights[~valid] != 0):
        raise ValueError('An edge cannot address a token before the answer starts')
    if not normalize:
        return weights.copy()
    normalized = normalize_edge_weights(torch.from_numpy(senders[valid]),
        torch.from_numpy(targets[valid]), torch.from_numpy(weights[valid]), len(weights))
    result = np.zeros_like(weights)
    result[valid] = normalized.numpy()
    return result


def frontier_step(index, width, log_odds, edge_weights, coupling):
    """Bit zero is the newest past state; add exactly the factors ending here."""
    states = np.arange(1 << min(index, width), dtype=np.int64)
    current = np.array([0, 1], dtype=np.int64)
    next_mask = (1 << min(index + 1, width)) - 1
    next_states = ((states[:, None] << 1) | current) & next_mask
    potential = np.broadcast_to(current * log_odds, next_states.shape).copy()
    for slot in range(min(index, len(edge_weights))):
        sender_state = (states >> slot) & 1
        different = sender_state[:, None] != current
        potential -= coupling * edge_weights[slot] * different
    return states, next_states, potential


def posterior_fields(state1, adjacent, log_partition):
    return dict(state1=state1, adjacent_joint=adjacent,
                enter=adjacent[:, 0, 1], continue_state=adjacent[:, 1, 1],
                exit=adjacent[:, 1, 0], log_partition=float(log_partition),
                exact=True, offline=True, calibrated_truth_probability=False)


def infer_states(unary, weights, coupling, normalize=True):
    """Sum exp(sum logit(U)z - coupling*sum w*[z_j!=z_t]) exactly.

    The fixed 1e-12 clipping makes boundary log odds finite. With zero edges
    or coupling, node marginals equal this clipped unary exactly.
    normalize=False retains supplied weights, solely for an unbudgeted prior
    control; main graph configurations use the default incident budget.
    """
    unary = np.asarray(unary, dtype=np.float64)
    if unary.ndim != 1 or not np.isfinite(unary).all() or ((unary < 0) | (unary > 1)).any():
        raise ValueError('Unary must be a finite [token] compatibility in [0,1]')
    if not np.isfinite(coupling) or coupling < 0:
        raise ValueError('Potts coupling must be finite and nonnegative')
    edges = normalized_lag_weights(weights, normalize=normalize)
    if len(edges) != len(unary):
        raise ValueError('Unary and graph token axes differ')
    probability = np.clip(unary, EPSILON, 1 - EPSILON)
    log_odds = np.log(probability) - np.log1p(-probability)
    if coupling == 0 or not edges.any():
        marginals = np.column_stack((1 - probability, probability))
        adjacent = marginals[:-1, :, None] * marginals[1:, None, :]
        return posterior_fields(probability.copy(), adjacent, np.logaddexp(0, log_odds).sum())
    width = max(1, edges.shape[1])
    forward, steps = [np.zeros(1)], []
    for index, emission in enumerate(log_odds):
        states, next_states, potential = frontier_step(index, width, emission, edges[index], coupling)
        next_forward = np.full(1 << min(index + 1, width), -np.inf)
        np.logaddexp.at(next_forward, next_states.ravel(), (forward[-1][:, None] + potential).ravel())
        forward.append(next_forward)
        steps.append((states, next_states, potential))
    state1 = np.empty(len(unary))
    adjacent = np.empty((max(0, len(unary) - 1), 2, 2))
    backward = np.zeros_like(forward[-1])
    for index in range(len(unary) - 1, -1, -1):
        states, next_states, potential = steps[index]
        suffix = potential + backward[next_states]
        joint_log = forward[index][:, None] + suffix
        joint = np.exp(joint_log - logsumexp(joint_log))
        state1[index] = joint[:, 1].sum()
        if index:
            for previous in (0, 1):
                adjacent[index - 1, previous] = joint[(states & 1) == previous].sum(axis=0)
        backward = logsumexp(suffix, axis=1)
    return posterior_fields(state1, adjacent, logsumexp(forward[-1]))


def gated_weights(attention, group_heads):
    """Per-head full-coordinate delta-message cosine gates continuity only.

    Input heads are [condition=2,T+1,group=5,head=32,width=128]. Post-token
    rows [1:] retain all five channels; no risk score or token label is used.
    """
    attention = np.asarray(attention, dtype=np.float64)
    heads = np.asarray(group_heads)
    count = len(attention)
    if attention.shape != (count, 32, 8) or heads.shape != (2, count + 1, 5, 32, 128):
        raise ValueError('Require attention [T,32,8] and group heads [2,T+1,5,32,128]')
    if not np.isfinite(attention).all() or (attention < 0).any() or not np.isfinite(heads).all():
        raise ValueError('Attention and delta-message coordinates must be finite')
    delta = heads[0, 1:].astype(np.float64) - heads[1, 1:].astype(np.float64)
    vectors = delta.transpose(0, 2, 1, 3).reshape(count, 32, 5 * 128)
    norms = np.linalg.norm(vectors, axis=-1)
    gates = np.zeros((count, 8, 32), dtype=np.float64)
    for lag in range(1, min(8, count - 1) + 1):
        dot = np.einsum('thd,thd->th', vectors[lag:], vectors[:-lag])
        denominator = norms[lag:] * norms[:-lag]
        valid = (norms[lag:] > EPSILON) & (norms[:-lag] > EPSILON)
        cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=valid)
        gates[lag:, lag - 1] = np.clip(cosine, 0, 1)
    raw_weights = (attention.transpose(0, 2, 1) * gates).mean(axis=-1)
    return raw_weights, gates
