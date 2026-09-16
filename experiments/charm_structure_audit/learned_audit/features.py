"""Measurements of the actual thresholded graph, never reconstructed softmax."""

import numpy as np


CONTEXT_NAMES = ["log_position", "relative_position", "log_response_length",
                 "log_prompt_length", "log_in_rp", "log_in_rr", "log_out_degree",
                 "mean_self", "mean_retained_mass"]


def head_overlap(graph):
    """Per-layer cosine agreement across heads reading the same sources, RP/RR separate."""
    layers, heads = int(graph["layers"]), int(graph["heads"])
    prompt = int(graph["prompt_length"])
    count = len(graph["x"]) - prompt
    source, target = graph["edge_index"]
    overlap = np.zeros((count, layers, 2), dtype=np.float32)
    available = np.zeros_like(overlap, dtype=bool)
    for query in np.unique(target):
        for relation in range(2):
            selected = (target == query) & ((source >= prompt) == bool(relation))
            weights = graph["edge_attr"][selected].reshape(-1, layers, heads).astype(float)
            norms = np.sqrt(np.sum(weights ** 2, axis=0))
            valid_heads = (norms > 0).sum(axis=1)
            normalized = weights / np.maximum(norms, 1e-12)[None]
            cross = np.sum(normalized.sum(axis=2) ** 2, axis=0)
            cross -= np.sum(normalized ** 2, axis=(0, 2))
            denominator = valid_heads * (valid_heads - 1)
            overlap[query - prompt, :, relation] = np.divide(
                cross, denominator, out=np.zeros(layers), where=denominator > 0)
            available[query - prompt, :, relation] = denominator > 0
    return overlap, available


def graph_features(graph):
    """Return [response,features], named column groups, and no gold-derived inputs."""
    prompt = int(graph["prompt_length"])
    source, target = graph["edge_index"]
    channels = graph["x"].shape[1]
    diagonal = graph["x"][prompt:].astype(np.float64)
    count = len(diagonal)
    mass = diagonal.copy()
    log_sum = diagonal * np.log(np.maximum(diagonal, 1e-300))
    rp = np.zeros_like(mass)
    rr = np.zeros_like(mass)
    weights = graph["edge_attr"].astype(np.float64)
    np.add.at(mass, target - prompt, weights)
    np.add.at(log_sum, target - prompt, weights * np.log(np.maximum(weights, 1e-300)))
    np.add.at(rp, target[source < prompt] - prompt, weights[source < prompt])
    np.add.at(rr, target[source >= prompt] - prompt, weights[source >= prompt])
    entropy = np.log(np.maximum(mass, 1e-300)) - log_sum / np.maximum(mass, 1e-300)
    entropy[mass == 0] = 0  # Also expose mass; zero-mass is not a known delta distribution.
    incoming_rp = np.bincount(target[source < prompt] - prompt, minlength=count)
    incoming_rr = np.bincount(target[source >= prompt] - prompt, minlength=count)
    outgoing = np.bincount(source, minlength=len(graph["x"]))[prompt:]
    context = np.column_stack((np.log1p(np.arange(count)), np.arange(count) / max(count - 1, 1),
        np.full(count, np.log1p(count)), np.full(count, np.log1p(prompt)),
        np.log1p(incoming_rp), np.log1p(incoming_rr), np.log1p(outgoing),
        diagonal.mean(axis=1), mass.mean(axis=1)))
    agreement, available = head_overlap(graph)
    features = np.column_stack((context, entropy, diagonal, rp, rr,
                               agreement.reshape(count, -1), available.reshape(count, -1)))
    return features.astype(np.float32), feature_columns(channels, int(graph["layers"]))


def feature_columns(channels, layers):
    context = len(CONTEXT_NAMES)
    entropy_end = context + channels
    marginal_end = context + 4 * channels
    return dict(context=list(range(context)),
                entropy=list(range(context, entropy_end)),
                context_entropy=list(range(entropy_end)),
                channel_marginals=list(range(marginal_end)),
                with_agreement=list(range(marginal_end + 4 * layers)))


def source_descriptors(graph, features):
    """Coarse label-blind matching descriptors; not semantic equivalence."""
    channels = graph["x"].shape[1]
    entropy = features[:, len(CONTEXT_NAMES):len(CONTEXT_NAMES) + channels]
    return np.column_stack((entropy.mean(axis=1), features[:, 7], features[:, 6]))


def feature_names(layers, heads):
    names = list(CONTEXT_NAMES)
    for kind in ("entropy", "self", "prompt_mass", "history_mass"):
        for layer in range(layers):
            for head in range(heads):
                names.append(f"{kind}/L{layer}/H{head}")
    for kind in ("head_agreement", "head_agreement_observed"):
        for layer in range(layers):
            for relation in ("prompt", "history"):
                names.append(f"{kind}/L{layer}/{relation}")
    return names
