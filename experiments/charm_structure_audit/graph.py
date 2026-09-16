"""The parent's threshold-union graph and label-blind structure controls."""

import numpy as np


VARIANTS = ("charm_out", "charm_in", "node_only", "local_in", "rewire_in")
PERTURBATIONS = ("real", "no_rp", "no_rr", "zero_edge", "zero_mark", "shuffle_nodes", "rewire_in")


def build_graph(record, tau=.05):
    """Keep all tokens; x=diagonal and e=thresholded [layer,head] vector.

    Input CSR row c*R+r means query P+r, not the next-token prediction row.
    No labels, token text, span boundaries or source identities enter this graph.
    """
    cache = record.sparse
    if cache is None:
        raise ValueError("this reproduction requires the existing canonical CSR NPZ")
    diagonal = cache["attention_diagonal"]
    layers, heads, n = diagonal.shape
    p, channels = record.response_idx, layers * heads
    pointer = cache["response_row_ptr"]
    values = np.asarray(cache["response_values"], dtype=np.float32)
    kept = np.flatnonzero(values > tau)
    rows = np.searchsorted(pointer, kept, side="right") - 1
    source = cache["response_column_indices"][kept].astype(np.int64)
    target, channel = p + rows % (n - p), rows // (n - p)
    if np.any(source < 0) or np.any(source >= target):
        raise ValueError("canonical off-diagonal edges must be source < query")
    pair, inverse = np.unique(target * n + source, return_inverse=True)
    edge_attr = np.zeros((len(pair), channels), np.float32)
    np.maximum.at(edge_attr, (inverse, channel), values[kept].astype(np.float32))
    edge_index = np.stack((pair % n, pair // n))
    rp = edge_index[0] < p
    return dict(x=diagonal.reshape(channels, n).T.astype(np.float32),
                edge_index=edge_index, edge_attr=edge_attr,
                edge_mark=np.stack((rp, ~rp), axis=1).astype(np.float32),
                prompt_length=np.asarray(p), layers=np.asarray(layers), heads=np.asarray(heads))


def transform_graph(graph, kind="real", seed=0):
    """Return a shallow copy with transformed inputs; labels are not consulted.

    local/rewire keep RP, target RR indegree and per-target edge vectors.
    local chooses nearest k history tokens. rewire draws without replacement
    inside each original coarse lag band. Neither preserves source outdegree.
    """
    out = dict(graph)
    edges = np.asarray(graph["edge_index"])
    source, target = edges
    p = int(graph["prompt_length"])
    rng = np.random.default_rng(seed)
    changed = 0
    if kind in ("real", "charm_in", "charm_out"):
        return out, dict(changed_edges=0, edges=len(source), changed_fraction=0.)
    if kind in ("no_rp", "no_rr", "node_only"):
        keep = source >= p if kind == "no_rp" else source < p
        if kind == "node_only":
            keep = np.zeros(len(source), bool)
        for field in ("edge_attr", "edge_mark"):
            out[field] = graph[field][keep]
        out["edge_index"] = edges[:, keep]
        changed = int((~keep).sum())
    elif kind in ("local_in", "rewire_in"):
        new_source = source.copy()
        for q in np.unique(target[source >= p]):
            idx = np.flatnonzero((target == q) & (source >= p))
            if kind == "local_in":
                new_source[idx] = np.arange(q - len(idx), q)
            else:
                bands = np.searchsorted([1, 4, 16, 64], q - source[idx], side="left")
                candidates = np.arange(p, q)
                candidate_bands = np.searchsorted([1, 4, 16, 64], q - candidates, side="left")
                for band in np.unique(bands):
                    group = idx[bands == band]
                    new_source[group] = rng.choice(candidates[candidate_bands == band], len(group), replace=False)
        changed = int(np.sum(source != new_source))
        out["edge_index"] = np.stack((new_source, target))
    elif kind == "zero_edge":
        out["edge_attr"] = np.zeros_like(graph["edge_attr"])
    elif kind == "zero_mark":
        out["edge_mark"] = np.zeros_like(graph["edge_mark"])
    elif kind == "shuffle_nodes":
        order = np.arange(len(graph["x"]))
        # Keep prompt/response role; do not move labels or structural degrees.
        order[:p] = rng.permutation(order[:p])
        order[p:] = rng.permutation(order[p:])
        out["x"] = graph["x"][order]
    else:
        raise ValueError("unknown graph control: " + kind)
    return out, dict(changed_edges=changed, edges=len(source),
                    changed_fraction=changed / len(source) if len(source) else 0.)


def degree(graph, normalization="in"):
    endpoint = 0 if normalization == "out" else 1
    return np.maximum(np.bincount(graph["edge_index"][endpoint], minlength=len(graph["x"])), 1).astype(np.float32)


def prefix_graph(graph, end):
    """Keep tokens [0,end); recompute graph statistics in the model."""
    out = dict(graph)
    keep = graph["edge_index"][1] < end
    out["x"] = graph["x"][:end]
    out["edge_index"] = graph["edge_index"][:, keep]
    for name in ("edge_attr", "edge_mark"):
        out[name] = graph[name][keep]
    return out


def node_statistics(graph):
    n, p = len(graph["x"]), int(graph["prompt_length"])
    source, target = graph["edge_index"]
    rp = source < p
    counts = lambda mask: np.bincount(target[mask], minlength=n).astype(float)
    rr_count = counts(~rp)
    values = dict(in_rp=counts(rp), in_rr=rr_count,
                  out_degree=np.bincount(source, minlength=n).astype(float),
                  local_rr_fraction=np.divide(counts((~rp) & (target - source <= 10)), rr_count,
                                              out=np.zeros(n), where=rr_count > 0),
                  self_attention_mean=graph["x"].mean(axis=1))
    return {k: v[p:] for k, v in values.items()}
