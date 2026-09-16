"""Change one kind of channel pairing while keeping the actual message slots."""

import numpy as np

from .features import graph_features, head_overlap, source_descriptors


MODES = ("vector", "layer", "head", "positive_head", "head_names")


def edge_groups(graph):
    """Same target, source role and lag band. lag=1 cannot swap with lag>1."""
    source, target = graph["edge_index"]
    prompt = int(graph["prompt_length"])
    bands = np.ceil(np.log2(target - source)).astype(int)
    keys = np.column_stack((target, source >= prompt, bands))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries) if len(order) else []


def permute_weights(graph, mode, seed):
    """Entropy-preserving controls; no labels, no thresholding after permutation."""
    result = dict(graph)
    weights = graph["edge_attr"].copy()
    layers, heads = int(graph["layers"]), int(graph["heads"])
    rng = np.random.default_rng(seed)
    if mode == "head_names":
        order = np.concatenate([layer * heads + rng.permutation(heads) for layer in range(layers)])
        result["x"] = graph["x"][:, order].copy()
        result["edge_attr"] = weights[:, order]
        return result
    for group in edge_groups(graph):
        if len(group) < 2:
            continue
        if mode == "vector":
            weights[group] = weights[rng.permutation(group)]
        elif mode == "layer":
            for layer in range(layers):
                columns = slice(layer * heads, (layer + 1) * heads)
                weights[group, columns] = weights[rng.permutation(group), columns]
        elif mode in ("head", "positive_head"):
            for channel in range(layers * heads):
                selected = group
                if mode == "positive_head":
                    selected = group[weights[group, channel] > 0]
                weights[selected, channel] = weights[rng.permutation(selected), channel]
        else:
            raise ValueError("unknown pairing control: " + mode)
    result["edge_attr"] = weights
    return result


def compare_inputs(graph, changed):
    """Report perturbation strength and side effects; do not silently assume invariants."""
    base_features, columns = graph_features(graph)
    new_features, _ = graph_features(changed)
    weights, altered = graph["edge_attr"], changed["edge_attr"]
    movable = sum(len(group) for group in edge_groups(graph) if len(group) > 1)
    overlap, observed = head_overlap(graph)
    new_overlap, new_observed = head_overlap(changed)
    common = observed & new_observed
    entropy_delta = new_features[:, columns["entropy"]] - base_features[:, columns["entropy"]]
    return dict(edges=len(weights), movable_edges=movable,
                changed_values=int(np.count_nonzero(weights != altered)),
                changed_edges=int(np.count_nonzero(np.any(weights != altered, axis=1))),
                zero_message_slots=int(np.count_nonzero(~np.any(altered > 0, axis=1))),
                changed_support=int(np.count_nonzero((weights > 0) != (altered > 0))),
                max_entropy_change=float(np.max(abs(entropy_delta))),
                mean_overlap_change=float((new_overlap - overlap)[common].mean()) if common.any() else None,
                overlap_observations=int(common.sum()))


def surface_classes(graph):
    text = str(graph["response"])
    classes = []
    for start, end in graph["offsets"]:
        piece = text[start:end].strip()
        if not piece:
            classes.append(0)
        elif piece.isalpha():
            classes.append(1)
        elif piece.isdigit():
            classes.append(2)
        else:
            classes.append(3)
    return np.asarray(classes)


def match_sources(graph, features):
    """Same-answer prior RR donors; identical eligible edges for both donor classes.

    Match lag, token surface class, current-token-copy status, and coarse
    entropy/self/outdegree. These are observed-state controls, not fact labels.
    """
    prompt = int(graph["prompt_length"])
    source, target = graph["edge_index"]
    labels = graph["gold"].astype(bool)
    descriptors = source_descriptors(graph, features)
    categories = surface_classes(graph)
    token_ids = graph["token_ids"][prompt:]
    same, other = source.copy(), source.copy()
    eligible = np.zeros(len(source), bool)
    for group in edge_groups(graph):
        if len(group) < 3 or source[group[0]] < prompt:
            continue
        local = source[group] - prompt
        query = target[group[0]] - prompt
        is_copy = token_ids[local] == token_ids[query]
        for offset, edge in enumerate(group):
            anchor = local[offset]
            difference = abs(descriptors[local] - descriptors[anchor])
            valid = (difference <= [0.25, 0.10, 1.0]).all(axis=1)
            valid &= (categories[local] == categories[anchor]) & (is_copy == is_copy[offset])
            valid &= local != anchor
            same_candidates = np.flatnonzero(valid & (labels[local] == labels[anchor]))
            other_candidates = np.flatnonzero(valid & (labels[local] != labels[anchor]))
            if not len(same_candidates) or not len(other_candidates):
                continue
            distances = np.sum((difference / [0.25, 0.10, 1.0]) ** 2, axis=1)
            same[edge] = source[group[same_candidates[np.argmin(distances[same_candidates])]]]
            other[edge] = source[group[other_candidates[np.argmin(distances[other_candidates])]]]
            eligible[edge] = True
    return dict(same_label=same, other_label=other, eligible=eligible)
