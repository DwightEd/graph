"""Exact finite differences; interaction is not a factuality label."""

import numpy as np

from ..message_carriers.token_representation import aggregate_units

CACHED_TOKENS = ("gate_source_shapley", "gate_source_controlled", "gate_interaction_size",
                 "head_coalition_size", "head_cancellation", "gate_sham_adjusted")
NATIVE_TOKENS = ("mediated_direct", "mediated_history", "mediated_total",
                 "mediated_context_shapley", "mediated_history_shapley", "mediated_interaction_size")
BASE_TOKENS = ("source_local", "source_full", "raw_route")


def factorial(values):
    """values[a,b,...]: prompt/source state a and history/edge state b, 0 then 1."""
    baseline = values[0, 0]
    direct = values[1, 0] - baseline
    history = values[0, 1] - baseline
    interaction = values[1, 1] - values[1, 0] - values[0, 1] + baseline
    return dict(direct=direct, history=history, interaction=interaction,
                context_shapley=direct + .5 * interaction,
                history_shapley=history + .5 * interaction,
                total=values[1, 1] - baseline)


def native_scores(worlds):
    components = factorial(np.asarray(worlds, dtype=float))
    scores = {f"mediated_{name}": -components[name]
              for name in ("direct", "history", "total", "context_shapley", "history_shapley")}
    scores["mediated_interaction_size"] = np.abs(components["interaction"])
    return scores, components


def gate_scores(keep, joint, sham, single):
    """Condition order in saved carrier arrays is source-present, source-absent."""
    keep, joint, sham, single = [np.asarray(value, dtype=float) for value in (keep, joint, sham, single)]
    worlds = np.stack([np.stack([joint[1], keep[1]]), np.stack([joint[0], keep[0]])])
    components = factorial(worlds)
    single_effect = keep[:, None] - single
    head_interaction = single_effect[0] - single_effect[1]
    joint_effect = keep - joint
    coalition = joint_effect - single_effect.sum(axis=1)
    conditional_coalition = coalition[0] - coalition[1]
    head_size = np.abs(head_interaction).sum(axis=0)
    cancellation = np.divide(head_size - np.abs(head_interaction.sum(axis=0)), head_size,
                             out=np.zeros_like(head_size), where=head_size > 0)
    sham_source = sham[0] - sham[1]
    source_keep = keep[0] - keep[1]
    scores = dict(gate_source_shapley=-components["context_shapley"],
                  gate_source_controlled=-components["direct"],
                  gate_interaction_size=np.abs(components["interaction"]),
                  head_coalition_size=np.abs(conditional_coalition),
                  head_cancellation=cancellation,
                  gate_sham_adjusted=-(components["direct"] + source_keep - sham_source))
    components.update(single_effect=single_effect, head_interaction=head_interaction,
                      coalition=coalition, conditional_coalition=conditional_coalition,
                      head_size=head_size, sham_source=sham_source, source_keep=source_keep)
    return scores, components


def add_unit_scores(scores, units, names):
    for name in names:
        scores[name + "_unit_mean"] = aggregate_units(scores[name], units)
    return scores
