"""Per-head changes to the same old endpoints; no labels enter the calculation."""

import numpy as np


def content_row(channel, row, special):
    keys, weights = channel.row(row)
    keep = ~special[keys]
    return keys[keep], weights[keep]


def old_endpoint_change(previous, current, prompt_length, query, local_window):
    """Use one endpoint set for both rows so keys ageing out cannot create a jump.

    The score is sum_j max(A[q,j] - A[q-1,j], 0), for ordinary prompt
    tokens or response tokens at least local_window positions behind q.
    We retain original attention units: sparse discarded mass is never restored.
    """
    old_keys, old_weights = previous
    keys, weights = current
    union = np.union1d(old_keys, keys)
    change = np.zeros(len(union))
    change[np.searchsorted(union, keys)] += weights
    change[np.searchsorted(union, old_keys)] -= old_weights
    eligible = (union < prompt_length) | (union <= query - local_window)
    positive = np.where(eligible, np.maximum(change, 0), 0)
    gain = float(positive.sum())
    winner = int(union[np.argmax(positive)]) if gain > 0 else -1
    return gain, winner, float(positive.max(initial=0))


def head_changes(channel, answer, special, local_window):
    """Index t denotes prediction of answer token t, from query P+t-1."""
    length = len(answer.response_ids)
    gain = np.full(length, np.nan)
    endpoint = np.full(length, -1, dtype=int)
    endpoint_gain = np.full(length, np.nan)
    retained = np.full(length, np.nan)
    rows = {int(query): row for row, query in enumerate(channel.queries)}
    for query, row in rows.items():
        target = query + 1 - answer.prompt_length
        if not 0 <= target < length or query - 1 not in rows:
            continue
        if special[query - 1:query + 2].any():
            continue
        previous = content_row(channel, rows[query - 1], special)
        current = content_row(channel, row, special)
        retained[target] = min(float(previous[1].sum()), float(current[1].sum()))
        if retained[target] <= 0:
            continue
        gain[target], endpoint[target], endpoint_gain[target] = old_endpoint_change(
            previous, current, answer.prompt_length, query, local_window)
    return gain, endpoint, endpoint_gain, retained


def before_peaks(values, window):
    """Strictly t-window,...,t-1. A missing row makes that window unavailable."""
    result = np.full(len(values), np.nan)
    if len(values) > window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        result[window:] = windows[:-1].max(axis=1)
    return result
