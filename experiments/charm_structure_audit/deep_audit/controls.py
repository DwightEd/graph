"""Frozen-model graph controls. Only oracle_span_cut may inspect gold labels."""

import numpy as np

from .common import fraction, membership


MODEL_FIELDS = ("x", "edge_index", "edge_attr", "edge_mark", "prompt_length", "layers", "heads")


def model_graph(graph):
    """Keep annotations physically outside the label-blind intervention API."""
    return {name: graph[name] for name in MODEL_FIELDS}


def edge_groups(graph):
    """Fixed query, prompt/history role, and coarse position/distance band."""
    source, target = graph["edge_index"]
    prompt = int(graph["prompt_length"])
    history = source >= prompt
    band = np.floor(np.log2(np.maximum(target - source, 1))).astype(int)
    band[~history] = source[~history] * 8 // max(prompt, 1)
    key = np.stack((target, history.astype(int), band), axis=1)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries) if len(order) else []


def shuffle_endpoints(graph, seed, independent=False):
    """Keep each head's weight multiset in every group exactly unchanged.

    Coupled: whole multi-head edge vectors move together.
    Independent: each head moves separately, breaking cross-head co-location.
    The union graph and divisor stay fixed, even if an edge becomes all-zero.
    """
    rng = np.random.default_rng(seed)
    original = graph["edge_attr"]
    altered = original.copy()
    eligible = informative = 0
    for group in edge_groups(graph):
        if len(group) < 2:
            continue
        block = original[group]
        eligible += len(group)
        informative += int((np.ptp(block, axis=0) > 0).sum())
        if independent:
            order = np.argsort(rng.random(block.shape), axis=0)
            altered[group] = np.take_along_axis(block, order, axis=0)
        else:
            altered[group] = block[rng.permutation(len(group))]
    changed = altered != original
    return dict(graph, edge_attr=altered), dict(
        eligible_edges=eligible, informative_head_groups=informative,
        changed_cells=int(changed.sum()), changed_cell_fraction=fraction(changed.sum(), changed.size),
        changed_edges=int(np.any(changed, axis=1).sum()),
        all_zero_edges=int(np.all(altered == 0, axis=1).sum()),
        retained_topology=True, per_head_group_marginals_preserved=True)


def permute_head_identity(graph, seed, edge_only=False):
    """A separate control for named-head identity, not the co-location test."""
    rng = np.random.default_rng(seed)
    layers, heads = int(graph["layers"]), int(graph["heads"])
    order = np.concatenate([layer * heads + rng.permutation(heads) for layer in range(layers)])
    out = dict(graph, edge_attr=graph["edge_attr"][:, order])
    if not edge_only:
        out["x"] = graph["x"][:, order]
    return out, dict(channel_order=order.tolist(), edge_only=edge_only)


def subset_edges(graph, keep):
    out = dict(graph, edge_index=graph["edge_index"][:, keep])
    for name in ("edge_attr", "edge_mark"):
        out[name] = graph[name][keep]
    return out


def degree_preserving_rewire(graph, seed, attempts_per_edge=2):
    """Causal RR double-edge swaps preserving BOTH in/out degree sequences."""
    source, target = graph["edge_index"]
    changed = source.copy()
    prompt = int(graph["prompt_length"])
    eligible = np.flatnonzero(source >= prompt)
    rng = np.random.default_rng(seed)
    existing = set(zip(source.tolist(), target.tolist()))
    attempts = attempts_per_edge * len(eligible) if len(eligible) > 1 else 0
    accepted = 0
    for _ in range(attempts):
        first, second = rng.choice(eligible, 2, replace=False)
        a, b, u, v = changed[first], changed[second], target[first], target[second]
        if a == b or u == v or b >= u or a >= v:
            continue
        if (b, u) in existing or (a, v) in existing:
            continue
        old_band = (int(u - a).bit_length(), int(v - b).bit_length())
        new_band = (int(u - b).bit_length(), int(v - a).bit_length())
        if old_band != new_band:
            continue
        existing.remove((a, u))
        existing.remove((b, v))
        existing.update(((b, u), (a, v)))
        changed[first], changed[second] = b, a
        accepted += 1
    out = dict(graph, edge_index=np.stack((changed, target)))
    count = int(np.sum(changed != source))
    return out, dict(rr_edges=len(eligible), attempts=attempts, accepted_swaps=accepted,
                     changed_rr_edges=count, changed_rr_fraction=fraction(count, len(eligible)),
                     preserves="RP edges, in/out degrees, target edge vectors, causal order, RR lag bands")


def oracle_span_cut(graph, sample, seed):
    """Gold-defined diagnostic cut and equally sized within-group random cut.

    Neither cut is a detector. The random cut may overlap the oracle cut;
    overlap and exchangeable groups are reported instead of hidden.
    """
    source, target = graph["edge_index"]
    prompt = int(graph["prompt_length"])
    member = np.r_[np.full(prompt, -1), membership(sample)]
    oracle = (source >= prompt) & (member[source] >= 0) & (member[source] == member[target])
    random_cut = np.zeros(len(source), bool)
    rng = np.random.default_rng(seed)
    exchangeable = 0
    for group in edge_groups(graph):
        count = int(oracle[group].sum())
        if count:
            random_cut[rng.choice(group, count, replace=False)] = True
            if count < len(group):
                exchangeable += count
    info = dict(removed_edges=int(oracle.sum()), exchangeable_removed_edges=exchangeable,
                null_overlap_fraction=fraction((oracle & random_cut).sum(), oracle.sum()),
                diagnostic_uses_gold=True, random_cut_matches="target, role, lag-band, removed count")
    return subset_edges(graph, ~oracle), subset_edges(graph, ~random_cut), info
