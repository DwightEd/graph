"""Attach annotations AFTER detection: node→span and span→node, with censoring."""

import json

import numpy as np
import pandas as pd

from .matching import token_kind


def preceding_state(states, onset, horizon):
    window = states[max(0, onset-horizon):onset]
    if np.any(window == 1):
        return 1
    if len(window) < horizon or np.any(window == -1):
        return -1
    return 0


def normal_controls(answer, onset, horizon, position_gap=.25):
    candidates = []
    length = len(answer.response_ids)
    for target in range(horizon, length-horizon):
        if abs(target-onset)/length > position_gap:
            continue
        if answer.error_mask[target-horizon:target+horizon+1].any():
            continue
        if answer.error_mask[:target].any() != answer.error_mask[:onset].any():
            continue
        if token_kind(answer, target) == token_kind(answer, onset):
            candidates.append(target)
    return sorted(candidates, key=lambda target: (abs(target-onset), target))


def span_rows(answer, states, horizon):
    rows = []
    for number, span in enumerate(answer.spans):
        prior = np.flatnonzero(states[max(0, span.start-horizon):span.start] == 1)
        prior = prior + max(0, span.start-horizon)
        start, end = answer.offsets[span.start, 0], answer.offsets[span.end-1, 1]
        candidates = normal_controls(answer, span.start, horizon)
        control = candidates[0] if candidates else -1
        rows.append(dict(span_index=number, onset=span.start, end=span.end, text=answer.text[start:end],
            answer_first=number == 0, run_onset=span.start == 0 or not answer.error_mask[span.start-1],
            prior_state=preceding_state(states, span.start, horizon), decision_state=int(states[span.start]),
            prior_node_targets=json.dumps(prior.tolist()), prior_node_count=len(prior),
            normal_target=control, normal_prior_state=preceding_state(states, control, horizon) if control >= 0 else -1,
            prior_error=bool(answer.error_mask[max(0, span.start-horizon):span.start].any())))
    return pd.DataFrame(rows, columns=['span_index', 'onset', 'end', 'text', 'answer_first', 'run_onset',
        'prior_state', 'decision_state', 'prior_node_targets', 'prior_node_count', 'normal_target',
        'normal_prior_state', 'prior_error'])


def position_rows(answer, states, horizon):
    """Every position, including non-events, unknowns and right-censored endings."""
    onsets = np.array([span.start for span in answer.spans], dtype=int)
    length = len(answer.response_ids)
    rows = []
    for target, state in enumerate(states):
        future = onsets[onsets > target]
        following = future[future <= target+horizon]
        rows.append(dict(target=target, node_token=target-1, state=int(state),
            current_gold=int(answer.error_mask[target]) if target < length else -1,
            at_span_onset=bool(np.any(onsets == target)),
            next_onset=int(future[0]) if len(future) else -1,
            next_gap=int(future[0]-target) if len(future) else -1,
            future_span=bool(len(following)), full_followup=target+horizon < length,
            following_onsets=json.dumps(following.tolist())))
    return pd.DataFrame(rows)


def link_rows(answer, nodes, horizon):
    rows = []
    for node in nodes.itertuples():
        for number, span in enumerate(answer.spans):
            lag = span.start-node.target
            if 0 <= lag <= horizon:
                rows.append(dict(target=node.target, node_token=node.node_token, node_text=node.node_text,
                    span_index=number, onset=span.start, lag_from_decision=lag,
                    relation='at_decision' if lag == 0 else 'strictly_before'))
    return pd.DataFrame(rows, columns=['target', 'node_token', 'node_text', 'span_index',
                                       'onset', 'lag_from_decision', 'relation'])
