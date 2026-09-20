"""Label-free local-to-old switches with bounds for omitted attention."""

import numpy as np


MEASURES = ('old_before', 'local_before', 'old_now', 'local_now',
            'missing_before', 'missing_now', 'shift_low', 'shift_high',
            'before_margin_low', 'before_margin_high', 'after_margin_low', 'after_margin_high')


def content_rows(channel, special):
    """Remove specials; correct only roundoff above one, never renormalize top-k loss."""
    rows = {}
    for index, query in enumerate(channel.queries):
        keys, weights = channel.row(index)
        if weights.sum() > 1.005:
            raise ValueError('Attention row mass exceeds the allowed rounding tolerance')
        weights = weights / max(1., float(weights.sum()))
        missing = max(0., 1. - float(weights.sum()))
        keep = ~special[keys]
        keys, weights = keys[keep], weights[keep]
        rows[int(query)] = (keys, np.r_[0., np.cumsum(weights)], missing)
    return rows


def partition(row, cutoff):
    keys, cumulative, missing = row
    old = float(cumulative[np.searchsorted(keys, cutoff)])
    return old, float(cumulative[-1] - old), missing


def switch_bounds(previous, current):
    """Both directions and both dominance inequalities must survive missing mass."""
    old_before, local_before, missing_before = previous
    old_now, local_now, missing_now = current
    old_gain = old_now - old_before
    local_drop = local_before - local_now
    before_margin = local_before - old_before
    after_margin = old_now - local_now
    return np.array([old_before, local_before, old_now, local_now,
        missing_before, missing_now,
        min(old_gain-missing_before, local_drop-missing_now),
        min(old_gain+missing_now, local_drop+missing_before),
        before_margin-missing_before, before_margin+missing_before,
        after_margin-missing_now, after_margin+missing_now])


def classify(measures, threshold):
    """1=certified switch; 0=ruled out; -1=unknown, never an imputed negative."""
    columns = dict(zip(MEASURES, measures.T))
    state = np.full(len(measures), -1, dtype=np.int8)
    finite = np.isfinite(measures).all(axis=1)
    positive = ((columns['shift_low'] >= threshold)
                & (columns['before_margin_low'] > 0) & (columns['after_margin_low'] > 0))
    negative = ((columns['shift_high'] < threshold)
                | (columns['before_margin_high'] <= 0) | (columns['after_margin_high'] <= 0))
    state[finite & negative] = 0
    state[finite & positive] = 1
    return state


def measure_head(channel, token_ids, prompt_length, special, baseline_steps, local_window):
    """Index t predicts answer token t: query q=P+t-1, node token index t-1.

    Every baseline row uses the CURRENT cutoff to prevent ageing artefacts.
    The final response node is retained even if its next token is unavailable.
    """
    rows = content_rows(channel, special)
    length = len(token_ids) - prompt_length + 1
    measures = np.full((length, len(MEASURES)), np.nan)
    for query in rows:
        target = query + 1 - prompt_length
        if not 0 <= target < length:
            continue
        baseline = range(query-baseline_steps, query)
        if any(position not in rows for position in baseline):
            continue
        if special[query-baseline_steps:query+1].any():
            continue
        cutoff = max(prompt_length, query-local_window+1)
        previous = np.mean([partition(rows[position], cutoff) for position in baseline], axis=0)
        measures[target] = switch_bounds(previous, partition(rows[query], cutoff))
    return measures, rows


def aggregate_heads(states):
    """One certified head proves existence; a negative needs every requested head."""
    result = np.full(states.shape[-1], -1, dtype=np.int8)
    result[np.all(states == 0, axis=0)] = 0
    result[np.any(states == 1, axis=0)] = 1
    return result


def event_starts(states):
    """Keep the first certified point of each contiguous per-head switching episode."""
    starts = states.copy()
    continuation = (states[1:] == 1) & (states[:-1] == 1)
    starts[1:][continuation] = 0
    return starts


def anchor_sources(rows, query, prompt_length, local_window, baseline_steps, coverage=.8):
    """Endpoints covering 80% of observed positive increments; bounds show uncertainty."""
    positions = list(range(query-baseline_steps, query+1))
    keys = np.unique(np.concatenate([rows[position][0] for position in positions]))
    weights = np.zeros((len(positions), len(keys)))
    for index, position in enumerate(positions):
        saved_keys, cumulative, _ = rows[position]
        weights[index, np.searchsorted(keys, saved_keys)] = np.diff(cumulative)
    previous = weights[:-1].mean(axis=0)
    gain = weights[-1] - previous
    old = keys < max(prompt_length, query-local_window+1)
    candidates = np.flatnonzero(old & (gain > 0))
    ordered = candidates[np.argsort(-gain[candidates], kind='stable')]
    count = np.searchsorted(np.cumsum(gain[ordered]), coverage*gain[ordered].sum()) + 1
    missing = np.mean([rows[position][2] for position in positions[:-1]])
    return [dict(source=int(keys[index]), attention_now=float(weights[-1, index]),
                 attention_before=float(previous[index]), source_gain=float(gain[index]),
                 source_gain_low=float(gain[index]-missing)) for index in ordered[:count]]
