"""Keep physical head and root identities; summarize only at final scoring."""

import numpy as np
from functools import lru_cache
from scipy.linalg import solve_triangular
from scipy.special import rel_entr

FIELDS = ('attention_js', 'influence_js', 'read_use_js', 'prompt_deficit',
          'prompt_positive', 'prompt_negative', 'history_positive', 'history_negative')


def normalize(value):
    total = value.sum(-1, keepdims=True)
    return np.divide(value, total, out=np.zeros_like(value), where=total > 0)


def js_rows(left, right, alpha):
    left_mass, right_mass = left.sum(-1), right.sum(-1)
    left, right = normalize(left), normalize(right)
    mixture = alpha[:, None] * left + (1 - alpha[:, None]) * right
    result = (alpha * rel_entr(left, mixture).sum(-1)
              + (1 - alpha) * rel_entr(right, mixture).sum(-1)) / np.log(2)
    return np.where((left_mass > 0) & (right_mass > 0), result, np.nan)


def direct_relay(weights, prompt):
    """Same-head token DAG surrogate, not cross-layer semantic transport.

    Row t is query prompt-1+t. A history key j uses query row j-prompt+1.
    Self is an absorbing terminal, never recursively reinjected.
    """
    count = len(weights)
    direct = weights[:, :prompt].copy()
    # At t=0 the prompt query self belongs to the direct prompt branch.
    history = np.zeros((count, count), dtype=weights.dtype)
    history[:, 1:] = np.tril(weights[:, prompt:prompt + count - 1], -1)
    # Row t, history column c corresponds to parent row c+1; exclude self.
    history = np.tril(history, -1)
    roots = solve_triangular(np.eye(count, dtype=weights.dtype) - history,
                             direct, lower=True, unit_diagonal=True)
    relay = history @ roots
    mass = direct.sum(-1) + relay.sum(-1)
    alpha = np.divide(direct.sum(-1), mass, out=np.zeros_like(mass), where=mass > 0)
    return js_rows(direct, relay, alpha)


def measure_head(attention, derivative, prompt):
    magnitude = np.abs(derivative)
    influence = normalize(magnitude)
    prompt_values = derivative[:, :prompt]
    history_values = derivative[:, prompt:]
    positive = np.maximum(prompt_values, 0).sum(-1)
    negative = np.maximum(-prompt_values, 0).sum(-1)
    total = magnitude.sum(-1)
    deficit = 1 - np.divide(positive, total, out=np.zeros_like(total), where=total > 0)
    fields = (direct_relay(attention, prompt), direct_relay(influence, prompt),
              js_rows(attention, influence, np.full(len(attention), .5)), deficit,
              positive, negative, np.maximum(history_values, 0).sum(-1),
              np.maximum(-history_values, 0).sum(-1))
    return np.stack(fields, -1)


def branch_alpha(weights, prompt):
    """Exact root total by linearity; no prompt-root binning is needed."""
    count = len(weights)
    history = np.zeros((count, count), dtype=weights.dtype)
    history[:, 1:] = weights[:, prompt:prompt + count - 1]
    history = np.tril(history, -1)
    direct = weights[:, :prompt].sum(-1)
    reach = solve_triangular(np.eye(count) - history, direct, lower=True, unit_diagonal=True)
    return np.divide(direct, reach, out=np.full_like(reach, np.nan), where=reach > 0)


def branch_relative_js(divergence, alpha):
    """Fraction of binary branch entropy explained by prompt-root identity."""
    from scipy.special import entr
    bound = (entr(alpha) + entr(1 - alpha)) / np.log(2)
    return np.divide(divergence, bound, out=np.full_like(divergence, np.nan), where=bound > 1e-12)


@lru_cache(maxsize=128)
def history_permutation(count):
    columns = np.broadcast_to(np.arange(count), (count, count)).copy()
    generator = np.random.default_rng(42)
    for token in range(count):
        lag = token - np.arange(token)
        for low, high in ((1, 4), (4, 16), (16, 64), (64, count + 1)):
            indices = np.flatnonzero((lag >= low) & (lag < high))
            columns[token, indices] = generator.permutation(indices)
    return columns


def propagate(score, influence, prompt, permuted=False):
    """Risk inheritance hypothesis on token keys, separate from root ancestry.

    Risk at key j was predicted at query j-1: parent score index is j-prompt.
    Prompt influence acts as an innovation weight. Self response keys can carry
    the previous generated token's risk. No truth interpretation of signs.
    """
    count = len(score)
    history = np.zeros((count, count), dtype=influence.dtype)
    history[:, :count - 1] = influence[:, prompt:prompt + count - 1]
    history = np.tril(history, -1)
    if permuted:
        history = np.take_along_axis(history, history_permutation(count), axis=1)
    innovation = (1 - history.sum(-1)) * score
    return solve_triangular(np.eye(count) - history, innovation, lower=True, unit_diagonal=True)
