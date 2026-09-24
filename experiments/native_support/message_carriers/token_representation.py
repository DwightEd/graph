"""Fixed physical head axes for independent token logp and choice-margin effects."""

import numpy as np

from ..evidence_contrast.aggregation import METHODS as BASE_METHODS, TOKEN_METHODS as BASE_TOKENS
from .representation import GLOBAL_CHANNELS

TOKEN_NEW = ("choice_selected", "choice_sham", "choice_replay",
             "token_source_selected", "token_source_sham", "token_source_replay")
TOKEN_METHODS = (*TOKEN_NEW, *BASE_TOKENS)
METHODS = (*TOKEN_NEW, *(f"{name}_unit_mean" for name in TOKEN_NEW), *BASE_METHODS)
COMPARISONS = tuple(("choice_selected" + suffix, name + suffix)
    for suffix in ("", "_unit_mean") for name in ("choice_replay", "choice_sham", "source_local", "raw_route"))
HEAD_CHANNELS = ("logp_delete_with_source", "logp_delete_without_source", "logp_interaction",
                 "margin_delete_with_source", "margin_delete_without_source", "margin_interaction", "measured")


def schema(layers, heads):
    layers, heads = int(layers), int(heads)
    return dict(version="token-carriers-v2", node="original_answer_token", dtype="float32",
        dimension=2 * len(GLOBAL_CHANNELS) + layers * heads * len(HEAD_CHANNELS),
        global_channels=[f"{metric}_{name.removeprefix('logp_')}" for metric in ("logp", "margin") for name in GLOBAL_CHANNELS],
        head_channels=list(HEAD_CHANNELS), head_shape=[layers, heads, len(HEAD_CHANNELS)],
        flatten_order="C: layer,head,channel", missing="measured=0; padded values are not measured zero effects",
        edge_columns=["layer", "head", "receiver_history_index", "key_history_index"],
        condition_axis=["with_source", "without_source"], metric_axis=["log_probability", "fixed_foil_margin"],
        single_axes=["condition", "edge", "metric"], approximation_axes=["edge", "condition"],
        target_scope="one_independent_token; future_answer_tokens_not_used",
        candidate_scope="history_only; bounded_receiver_screen; not_complete_circuit_or_source_attribution",
        foil="highest_non_target_with_source_logit; frozen_for_both_conditions_and_all_cuts")


def global_values(keep, joint, sham):
    effect, random = keep - joint, keep - sham
    return np.asarray([*keep, keep[0] - keep[1], *effect, effect[0] - effect[1],
                       joint[0] - joint[1], *random, random[0] - random[1], sham[0] - sham[1]])


def token_vector(saved):
    keep, joint, sham = (saved[name].astype(float) for name in ("keep", "joint", "sham"))
    layers, heads = saved["head_shape"]
    global_part = np.concatenate([global_values(keep[:, metric], joint[:, metric], sham[:, metric]) for metric in (0, 1)])
    head_part = np.zeros((layers, heads, len(HEAD_CHANNELS)), dtype=np.float32)
    keys, receivers = np.full((layers, heads), -1), np.full((layers, heads), -1)
    effects = keep[:, None, :] - saved["single"].astype(float)
    for index, (layer, head, receiver, key) in enumerate(saved["edges"]):
        logp, margin = effects[:, index, 0], effects[:, index, 1]
        head_part[layer, head] = [*logp, logp[0] - logp[1], *margin, margin[0] - margin[1], 1]
        keys[layer, head], receivers[layer, head] = key, receiver
    return np.concatenate([global_part, head_part.ravel()]).astype(np.float32), keys, receivers


def assemble_tokens(views, baseline, measurements):
    if len(measurements) != len(views["answer_ids"]):
        raise ValueError("Incomplete independent-token capture")
    vectors, keys, receivers = [], [], []
    scores = {name: value.copy() for name, value in baseline.items()}
    for target, saved in enumerate(measurements):
        if saved["target"] != target or saved["token_id"] != views["answer_ids"][target]:
            raise ValueError("Independent-token target identity differs")
        vector, key, receiver = token_vector(saved)
        vectors.append(vector)
        keys.append(key)
        receivers.append(receiver)
    features = np.stack(vectors)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite token-choice representation")
    for metric, prefix in ((0, "token_source"), (1, "choice")):
        for condition, name in (("joint", "selected"), ("sham", "sham"), ("keep", "replay")):
            value = np.asarray([float(saved[condition][1, metric]) - float(saved[condition][0, metric])
                                for saved in measurements])
            scores[f"{prefix}_{name}"] = value
            scores[f"{prefix}_{name}_unit_mean"] = aggregate_units(value, views["units"])
    scores["selected_count"] = np.asarray([len(saved["edges"]) for saved in measurements])
    return scores, dict(node_features=features, history_index=np.stack(keys), receiver_index=np.stack(receivers),
        foil_id=np.asarray([saved["foil_id"] for saved in measurements]),
        target=scores["target"], token_id=scores["token_id"], unit_id=scores["unit_id"])


def aggregate_units(values, units, beta=0.0):
    """Output-position weighting, not a key-distance prior or an attention edit."""
    result = np.empty(len(values), dtype=float)
    for unit in units:
        start, stop = unit["start"], unit["stop"]
        position = np.linspace(0, 1, stop - start)
        logits = beta * position
        weights = np.exp(logits - logits.max())
        result[start:stop] = np.dot(values[start:stop], weights) / weights.sum()
    return result
