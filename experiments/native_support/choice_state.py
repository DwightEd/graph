"""Conditional choice factors and an absorbing history-lineage model.

The native attribution stops at input embeddings. Crossing from a historical
embedding to its earlier decision state is an explicit detector hypothesis.
"""

import numpy as np
from scipy.special import logsumexp, softmax, xlogy

NORMALIZER_ROUNDOFF_TOLERANCE = 1e-5  # Log-probability units for the saved float32 normalizer.


def entropy(probability):
    return -xlogy(probability, probability).sum(-1)


def read_observations(group_attention, source_count):
    """Current key-address reading; distinct from full-prefix root support."""
    source = group_attention[..., :source_count].astype(float)
    total = source.sum(-1, keepdims=True)
    conditional = np.divide(source, total, out=np.zeros_like(source), where=total > 0)
    active = total[..., 0] > 0
    concentration = np.square(conditional).sum(-1)
    return {
        "source_read_mass": float(total.mean()),
        "source_read_concentration": float(concentration[active].mean()) if active.any() else 0.,
        "source_read_entropy": float(entropy(conditional)[active].mean()) if active.any() else 0.,
    }


def choice_factors(row):
    """Keep native probabilities separate from normalized attribution factors."""
    group_odds = row["root_positive"].astype(float) - row["root_negative"]
    logits = row["candidate_logits"].astype(float)
    conditional = softmax(logits)
    # Float32 (1 - sum(probability)) can be negative; conditioning amplifies it.
    # Surprisal provides the full-vocabulary normalizer without another forward.
    log_captured_mass = logsumexp(logits - logits[0]) - float(row["surprisal"])
    if log_captured_mass > NORMALIZER_ROUNDOFF_TOLERANCE:
        raise ValueError("saved candidate logits and surprisal have inconsistent normalizers")
    tail = float(-np.expm1(min(log_captured_mass, 0.)))
    native = conditional * (1 - tail)
    alternative_mass = native[1:].sum() + tail
    weights = native[1:] / alternative_mass if alternative_mass > 0 else np.zeros(len(logits) - 1)
    unknown = tail / alternative_mass if alternative_mass > 0 else 1.
    anchored = np.concatenate((np.zeros((len(group_odds), 1)), -group_odds), axis=1)
    reconstructed = softmax(anchored.sum(0))
    return {
        "group_log_odds": group_odds,
        "group_choice_factor": softmax(anchored, axis=-1),
        "native_candidate_probability": native,
        "conditional_candidate_probability": conditional,
        "reconstructed_candidate_probability": reconstructed,
        "alternative_weight": weights,
        "unobserved_alternative_mass": unknown,
        "reconstructed_tail_mass": tail,
        "tail_roundoff_correction": tail - float(row["candidate_tail_mass"]),
        "normalizer_roundoff": max(float(log_captured_mass), 0.),
        "choice_reconstruction_error": float(np.abs(reconstructed - conditional).max()),
    }


def transition(row, source_count, prompt):
    """Return per-candidate terminal mass and strictly earlier positive links.

    Terminal axes: group, sign (support/opposition), candidate. The extra last
    group is unresolved. Historical opposition terminates; its sign is not
    multiplied across unrelated token contrasts.
    """
    positive = row["root_positive"].astype(float)
    negative = row["root_negative"].astype(float)
    total = (positive + negative).sum(0)
    denominator = np.where(total > 0, total, 1)
    terminal = np.zeros((len(positive) + 1, 2, len(total)))
    terminal[:-1, 0] = positive / denominator
    terminal[:-1, 1] = negative / denominator
    terminal[source_count, 0] = 0
    terminal[-1, 0] = total == 0
    history = np.maximum(row["root_token_choice"][prompt:].astype(float), 0)
    history *= (row["group_ids"][prompt:] == source_count)[:, None]
    return terminal, history / denominator


def read_state(row, source_count, prompt, previous):
    """A source/sign distribution, not a scalar score used as hidden state."""
    factors = choice_factors(row)
    terminal, history = transition(row, source_count, prompt)
    candidate_state = terminal.copy()
    if len(previous):
        candidate_state += np.einsum("jc,jgs->gsc", history, previous)
    weights = factors["alternative_weight"]
    state = candidate_state @ weights
    state[-1, 0] += factors["unobserved_alternative_mass"]
    local = terminal @ weights
    local[-1, 0] += factors["unobserved_alternative_mass"]
    return factors, state, local, candidate_state, history, history @ weights


def descendant_opposition(history, local_opposition):
    """Offline path-origin average, with its exposure denominator saved.

    Each decision starts one path. The reverse recursion distributes origins
    onto positive historical contributors. This is not a hallucination posterior.
    """
    exposure = np.ones(len(history))
    opposition = local_opposition.copy()
    for target in range(len(history) - 1, -1, -1):
        exposure[:target] += history[target, :target] * exposure[target]
        opposition[:target] += history[target, :target] * opposition[target]
    return opposition / exposure, exposure


def source_deficit(state, source_count):
    """No positive source endpoint, conditional on a resolved path endpoint.

    All-unresolved states are unscored. Missing candidate information is not
    turned into opposition. Lower/upper bounds describe the omitted mass only.
    """
    support = state[:, :source_count, 0].sum(1)
    resolved = state[:, :-1].sum((1, 2))
    unsupported = resolved - support
    conditional = np.divide(unsupported, resolved, out=np.full(len(state), np.nan), where=resolved > 0)
    return conditional, unsupported, unsupported + state[:, -1, 0]


def summarize_states(states, local, history, source_count):
    support = states[:, :source_count, 0].sum(1)
    opposition = states[:, :source_count, 1].sum(1)
    local_opposition = local[:, :source_count, 1].sum(1)
    descendant, exposure = descendant_opposition(history, local_opposition)
    deficit, lower, upper = source_deficit(states, source_count)
    # The direct control stops at history roots instead of following their states.
    direct = local.copy()
    direct[:, source_count, 0] = history.sum(1)
    local_deficit, _, _ = source_deficit(direct, source_count)
    length = np.ones(len(history))
    for target in range(len(history)):
        length[target] += history[target, :target] @ length[:target]
    return {
        "lineage_source_deficit": deficit,
        "local_source_deficit": local_deficit,
        "source_deficit_lower": lower,
        "source_deficit_upper": upper,
        "lineage_opposition": opposition,
        "local_opposition": local_opposition,
        "descendant_opposition": descendant,
        "source_support_state": support,
        "source_support_terminal": local[:, :source_count, 0].sum(1),
        "inherited_opposition": opposition - local_opposition,
        "history_carry": history.sum(1),
        "history_opposition_terminal": local[:, source_count, 1],
        "unresolved_state": states[:, -1, 0],
        "state_entropy": entropy(states.reshape(len(states), -1)),
        "expected_path_nodes": length,
        "descendant_exposure": exposure,
        "state_mass_error": np.abs(states.sum((1, 2)) - 1),
    }
