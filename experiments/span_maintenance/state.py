"""Maintain all physical heads; a reading regime is not a truth state."""
import numpy as np

FIELDS = (
    'history_mass', 'prompt_refresh', 'anchor_overlap', 'lag_overlap',
    'continuation', 'age', 'member_mass', 'distance_expected',
    'distance_excess', 'exchangeable_mass', 'effective_tokens',
)


def normalize(values):
    total = values.sum(-1, keepdims=True)
    return np.divide(values, total, out=np.zeros_like(values), where=total > 0)


def overlap(current, previous):
    return np.clip(np.sqrt(current * normalize(previous)).sum(-1), 0, 1)


def distance_expectation(attention, membership, answer_ids, target):
    """Exact endpoint-exchange expectation within distance and token-ID strata."""
    heads = attention.shape[0]
    expected = np.zeros(heads)
    exchangeable = np.zeros(heads)
    distance = target - np.arange(target)
    buckets = np.floor(np.log2(distance)).astype(int)
    matches = np.asarray(answer_ids[:target]) == answer_ids[target]
    groups = 2 * buckets + matches
    for group in np.unique(groups):
        selected = groups == group
        mass = attention[:, selected].sum(-1)
        expected += mass * membership[:, selected].mean(-1)
        if selected.sum() > 1:
            exchangeable += mass
    return expected, exchangeable


def reading_profiles(attention, prompt, target, count):
    reading = normalize(attention[:, target].astype(float))
    prompt_reading = reading[:, :prompt]
    history = reading[:, prompt:prompt + target]
    anchor = np.zeros((reading.shape[0], count))
    anchor[:, :target] = normalize(history)
    lag = np.zeros_like(anchor)
    lag[:, :target] = anchor[:, :target][:, ::-1]
    return prompt_reading, history, anchor, lag


def maintain_layer(attention, prompt, answer_ids, boundaries=None, boundary_mode='none'):
    """Input axes: [head, predicted-answer-token, absolute key]."""
    heads, count, _ = attention.shape
    prompt_memory = np.zeros((heads, prompt))
    anchor_memory = np.zeros((heads, count))
    lag_memory = np.zeros_like(anchor_memory)
    membership = np.zeros_like(anchor_memory)
    age = np.zeros(heads)
    measured = np.zeros((heads, count, len(FIELDS)))
    state_weights = np.zeros((count, count))
    dependency_weights = np.zeros_like(state_weights)
    for target in range(count):
        prompt_reading, history, anchor, lag = reading_profiles(attention, prompt, target, count)
        refresh = np.maximum(prompt_reading - prompt_memory, 0).sum(-1)
        anchor_overlap = overlap(anchor, anchor_memory)
        lag_overlap = overlap(lag, lag_memory)
        continuation = np.maximum(anchor_overlap, lag_overlap) * (1 - refresh)
        if boundary_mode != 'none' and boundaries[target]:
            # Soft confirmation preserves a perfectly stable regime across punctuation.
            # This is a fixed continuity heuristic, not a calibrated posterior.
            continuation *= 0 if boundary_mode == 'hard' else continuation
        retained = continuation * age
        age = 1 + retained
        membership *= continuation[:, None]
        member_mass = (history * membership[:, :target]).sum(-1)
        expected, exchangeable = distance_expectation(history, membership[:, :target], answer_ids, target)
        membership[:, target] = 1
        effective = age ** 2 / (membership ** 2).sum(-1)
        measured[:, target] = np.stack((history.sum(-1), refresh, anchor_overlap,
            lag_overlap, continuation, age, member_mass, expected, member_mass - expected,
            exchangeable, effective), axis=-1)
        state_weights[target] = (membership / age[:, None]).mean(0)
        dependency_weights[target, :target] = (history * membership[:, :target]).mean(0)
        dependency_weights[target, target] = 1
        dependency_weights[target] /= dependency_weights[target].sum()
        for memory, current in ((prompt_memory, prompt_reading), (anchor_memory, anchor), (lag_memory, lag)):
            memory *= retained[:, None]
            memory += current
            memory /= age[:, None]
    return measured, state_weights, dependency_weights


def signed_readout(effect, prompt):
    """Do not cancel signs; changing target margins prohibit semantic sign claims."""
    positive = np.maximum(effect, 0)
    negative = np.maximum(-effect, 0)
    result = np.stack((positive[..., :prompt].sum(-1), negative[..., :prompt].sum(-1),
        positive[..., prompt:].sum(-1), negative[..., prompt:].sum(-1)), axis=-1)
    signed = np.concatenate((positive, negative), axis=-1).astype(float)
    root = np.sqrt(normalize(signed))
    stability = np.zeros(effect.shape[:2])
    stability[:, 1:] = (root[:, 1:] * root[:, :-1]).sum(-1)
    return result, stability
