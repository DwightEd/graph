"""Read addresses and automatic competing-token trajectories, without semantic masks."""

import numpy as np


def normalize(values):
    return values / np.maximum(values.sum(-1, keepdims=True), 1e-30)


def js_divergence(left, right):
    """Jensen-Shannon divergence in nats; inputs are normalized distributions."""
    middle = (left + right) / 2
    first = left * np.log(np.maximum(left, 1e-30) / np.maximum(middle, 1e-30))
    second = right * np.log(np.maximum(right, 1e-30) / np.maximum(middle, 1e-30))
    return (first.sum(-1) + second.sum(-1)) / 2


def routing_layer(attention, prompt, special, history_window=16):
    """One layer [head, target, key]; compare identical prompt addresses over time."""
    heads, steps, _ = attention.shape
    prompt_keys = np.flatnonzero(~special[:prompt])
    weights = attention[:, :, prompt_keys].astype(np.float32)
    prompt_mass = weights.sum(-1)
    addresses = normalize(weights)
    address_change = np.zeros((heads, steps), dtype=np.float32)
    for position in range(1, steps):
        previous = normalize(weights[:, max(0, position - 3):position].mean(1))
        address_change[:, position] = js_divergence(addresses[:, position], previous)
    history_mass = np.zeros_like(prompt_mass)
    local_mass = np.zeros_like(prompt_mass)
    history_age = np.zeros_like(prompt_mass)
    for position in range(1, steps):
        keys = np.arange(prompt, prompt + position)
        valid = ~special[keys]
        past = attention[:, position, prompt:prompt + position].astype(np.float32)
        past *= valid
        history_mass[:, position] = past.sum(-1)
        local_mass[:, position] = past[:, -history_window:].sum(-1)
        ages = prompt + position - keys
        history_age[:, position] = (past * ages).sum(-1) / np.maximum(past.sum(-1), 1e-30)
    # Tiny prompt mass cannot create a large routing event just by normalization.
    weighted_change = prompt_mass * address_change
    return np.stack((prompt_mass, address_change, weighted_change,
                     history_mass, local_mass, history_age), -1)


def candidate_ids(top_ids, chosen_ids):
    """Five native top tokens plus the sampled token when it is outside that set."""
    ids = np.column_stack((top_ids, chosen_ids))
    valid = np.ones(ids.shape, dtype=bool)
    valid[:, -1] = ~np.any(top_ids == chosen_ids[:, None], axis=1)
    return ids, valid


def lens_readouts(logits, valid, ids, chosen_ids):
    """Restricted-candidate distributions are not full-vocabulary entropy or truth."""
    masked = np.where(valid[None], logits, -np.inf)
    probability = normalize(np.exp(masked - masked.max(-1, keepdims=True)))
    winner = masked.argmax(-1)
    agreement = winner == 0  # Column zero is the final native top-1, before sampling.
    settled = np.logical_and.accumulate(agreement[::-1], axis=0)[::-1]
    settle_layer = np.where(settled.any(0), settled.argmax(0) + 1, len(logits) + 1)
    winner_switches = (winner[1:] != winner[:-1]).sum(0)
    final = np.broadcast_to(probability[-1], probability.shape)
    divergence = js_divergence(probability, final)
    chosen_column = (ids == chosen_ids[:, None]).argmax(-1)
    target_logits = np.take_along_axis(logits, chosen_column[None, :, None], axis=2)[..., 0]
    alternative = masked.copy()
    for position, column in enumerate(chosen_column):
        alternative[:, position, column] = -np.inf
    margin = target_logits - alternative.max(-1)
    reversal = ((margin[1:] >= 0) != (margin[:-1] >= 0)).sum(0)
    return dict(candidate_probability=probability, chosen_margin=margin,
                final_top1_settle_layer=settle_layer, winner_switches=winner_switches,
                late_candidate_js=divergence[15:-1].mean(0),
                chosen_margin_reversals=reversal,
                chosen_margin_final=margin[-1],
                chosen_margin_late_min=margin[15:].min(0),
                chosen_margin_late_max=margin[15:].max(0))


def state_turns(hidden, prompt, steps):
    """Residual motion before final RMSNorm; never subtract normalized final hidden."""
    state = hidden[16:32, prompt - 1:prompt + steps - 1].astype(np.float32)
    direction = state / np.maximum(np.linalg.norm(state, axis=-1, keepdims=True), 1e-30)
    temporal = np.zeros((len(state), steps), dtype=np.float32)
    temporal[:, 1:] = 1 - (direction[:, 1:] * direction[:, :-1]).sum(-1)
    updates = state[1:] - state[:-1]
    update_norm = np.linalg.norm(updates, axis=-1)
    relative = update_norm / np.maximum(np.linalg.norm(state[:-1], axis=-1), 1e-30)
    unit = updates / np.maximum(update_norm[..., None], 1e-30)
    opposition = np.maximum(0, -(unit[1:] * unit[:-1]).sum(-1)).mean(0)
    return dict(state_temporal_turn=temporal.mean(0),
                late_update_relative_norm=relative.mean(0),
                adjacent_update_opposition=opposition)


def source_balanced_percentile(reference, sources, values):
    """Mixed, unlabeled reference ranks; neither null p-values nor error probabilities."""
    result = np.zeros(len(values), dtype=np.float64)
    unique_sources = np.unique(sources)
    for source in unique_sources:
        ordered = np.sort(reference[sources == source])
        result += np.searchsorted(ordered, values, side='left') / len(ordered)
    return result / len(unique_sources)
