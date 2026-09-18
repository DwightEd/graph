"""Label-free score controls. No span boundary, current label or future input."""

import zlib

import numpy as np


SCORE_NAMES = (
    'current', 'previous', 'past_mean', 'past_ewma',
    'causal_ewma', 'current_increment', 'prefix_mean', 'sampled_past',
)


def score_controls(probabilities, window=10, beta=.5, seed=17):
    """All arrays use original probability units; increment is a signed difference."""
    values = np.asarray(probabilities, dtype=float)
    count = len(values)
    if not count:
        return {name: np.empty(0) for name in SCORE_NAMES}

    position = np.arange(count)
    left = np.maximum(0, position - window)
    cumulative = np.r_[0., np.cumsum(values)]
    width = position - left
    past = np.divide(cumulative[position] - cumulative[left], width,
                     out=np.full(count, np.nan), where=width > 0)
    prefix = np.divide(cumulative[position], position,
                       out=np.full(count, np.nan), where=position > 0)

    smoothed = values.copy()
    for token in range(1, count):
        smoothed[token] = (1 - beta) * values[token] + beta * smoothed[token - 1]
    sampled = sample_prefix(values, window, seed)
    return dict(current=values.copy(), previous=np.r_[np.nan, values[:-1]],
                past_mean=past, past_ewma=np.r_[np.nan, smoothed[:-1]],
                causal_ewma=smoothed, current_increment=values - past,
                prefix_mean=prefix, sampled_past=sampled)


def sample_prefix(values, window, seed):
    """Same number of observations as past_mean, but drawn from the whole past."""
    random = np.random.default_rng(seed)
    result = np.full(len(values), np.nan)
    for token in range(1, len(values)):
        size = min(window, token)
        positions = random.choice(token, size, replace=False)
        result[token] = values[positions].mean()
    return result


def score_table(table, window=10, beta=.5, seed=17):
    """Only id/token/score are used. Other columns are copied, never consulted."""
    result = table.sort_values(['id', 'token']).reset_index(drop=True).copy()
    for name in SCORE_NAMES:
        result[name] = np.nan
    for identity, indices in result.groupby('id', sort=False).indices.items():
        values = result.loc[indices, 'score'].to_numpy()
        local_seed = seed + zlib.crc32(str(identity).encode())
        controls = score_controls(values, window, beta, local_seed)
        for name, scores in controls.items():
            result.loc[indices, name] = scores
    return result
