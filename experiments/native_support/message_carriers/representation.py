"""One token vector, fixed physical head axes, and separate sparse edge identities."""

import numpy as np

from ..evidence_contrast.aggregation import METHODS as BASE_METHODS, TOKEN_METHODS as BASE_TOKENS

NEW_TOKENS = ("source_selected", "source_sham", "source_full_replay")
TOKEN_METHODS = (*NEW_TOKENS, *BASE_TOKENS)
METHODS = (*NEW_TOKENS, *(f"{name}_unit_mean" for name in NEW_TOKENS), *BASE_METHODS)
COMPARISONS = tuple(("source_selected" + suffix, name + suffix)
                    for suffix in ("", "_unit_mean")
                    for name in ("source_sham", "source_full_replay", "source_local", "raw_route"))
GLOBAL_CHANNELS = ("logp_with_source", "logp_without_source", "source_gain",
                   "joint_delete_with_source", "joint_delete_without_source", "joint_interaction",
                   "source_gain_after_joint_delete", "sham_delete_with_source",
                   "sham_delete_without_source", "sham_interaction", "source_gain_after_sham")
HEAD_CHANNELS = ("single_delete_with_source", "single_delete_without_source",
                 "single_interaction", "measured")


def feature_schema(layers, heads):
    return dict(version=1, node="original_answer_token", dtype="float32",
        dimension=len(GLOBAL_CHANNELS) + layers * heads * len(HEAD_CHANNELS),
        global_channels=list(GLOBAL_CHANNELS), head_channels=list(HEAD_CHANNELS),
        head_shape=[layers, heads, len(HEAD_CHANNELS)], flatten_order="C: layer,head,channel",
        head_column="len(global_channels)+(layer*heads+head)*len(head_channels)+channel",
        unmeasured="head values padded zero with measured=0; not measured zero effect",
        history_index="separate int array [token,layer,head]; -1 means unmeasured",
        edge_scope="same historical key gated at every query in the saved unit",
        edge_columns=["layer", "head", "history_index"],
        condition_axis=["with_source", "without_source"],
        single_axes=["condition", "selected_edge", "target"],
        approximation_axes=["selected_edge", "condition"],
        token_effect_scope="finite unit-wide intervention read at each token; includes upstream unit effects",
        gradient_scope="selected unit-mean gradient saved with edges, not a per-token derivative",
        feature_units="logp/effects in nats; measured is a binary indicator")


def unit_vectors(measured):
    keep, joint, sham = (measured[name].astype(np.float64) for name in ("keep", "joint", "sham"))
    layers, heads = measured["eligible_counts"].shape
    count = keep.shape[1]
    joint_effect, sham_effect = keep - joint, keep - sham
    global_values = np.stack([*keep, keep[0] - keep[1], *joint_effect,
        joint_effect[0] - joint_effect[1], joint[0] - joint[1], *sham_effect,
        sham_effect[0] - sham_effect[1], sham[0] - sham[1]], axis=1)
    head_values = np.zeros((count, layers, heads, len(HEAD_CHANNELS)), dtype=np.float32)
    history = np.full((count, layers, heads), -1, dtype=np.int64)
    individual = keep[:, None, :] - measured["single"].astype(np.float64)
    for index, (layer, head, key) in enumerate(measured["edges"]):
        effect = individual[:, index]
        head_values[:, layer, head] = np.stack([*effect, effect[0] - effect[1], np.ones(count)], axis=1)
        history[:, layer, head] = key
    vectors = np.concatenate([global_values, head_values.reshape(count, -1)], axis=1).astype(np.float32)
    return vectors, history


def assemble(views, baseline, measured_units):
    """Freeze scores and vectors without annotations; unit aggregation is explicit."""
    scores = {name: value.copy() for name, value in baseline.items()}
    vectors, histories, selected_counts = [], [], []
    for unit, measured in zip(views["units"], measured_units):
        target = np.arange(unit["start"], unit["stop"])
        if not np.array_equal(measured["target"], target) or not np.array_equal(
                measured["token_id"], np.asarray(views["answer_ids"])[target]):
            raise ValueError("Captured unit target identity mismatch")
        features, history = unit_vectors(measured)
        vectors.append(features)
        histories.append(history)
        selected_counts.extend([len(measured["edges"])] * len(target))
    features, history = np.concatenate(vectors), np.concatenate(histories)
    if len(features) != len(views["answer_ids"]) or not np.isfinite(features).all():
        raise ValueError("Missing or nonfinite token representations")
    readouts = dict(source_selected="source_gain_after_joint_delete",
                    source_sham="source_gain_after_sham", source_full_replay="source_gain")
    for name, channel in readouts.items():
        column = GLOBAL_CHANNELS.index(channel)
        scores[name] = -features[:, column].astype(float)
        mean = np.empty(len(features))
        for unit in views["units"]:
            selected = slice(unit["start"], unit["stop"])
            mean[selected] = scores[name][selected].mean()
        scores[f"{name}_unit_mean"] = mean
    scores["selected_count"] = np.asarray(selected_counts, dtype=np.int64)
    return scores, dict(node_features=features, history_index=history,
                        target=scores["target"], token_id=scores["token_id"], unit_id=scores["unit_id"])
