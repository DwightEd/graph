"""Score-blind measurements on the ORIGINAL threshold-union graph.

All channel arrays keep the original layer/head order. Scalar structural
summaries are matching covariates, never replacements for detector inputs.
"""

import numpy as np


STRUCTURE = (
    "internal_density", "internal_edge_share", "internal_mass_share",
    "lag1_share", "hub_concentration", "log_mean_edge_mass", "log_in_degree",
)
CALIPERS = np.asarray([.10, .10, .10, .10, .10, .50, .50])
PROFILE_NAMES = ("self_attention", "incoming_mass", "retained_entropy",
                 "prompt_share", "internal_share")


def window_average(values, length):
    """Average all contiguous equal-length windows; axes after token are kept."""
    prefix = np.concatenate((np.zeros_like(values[:1]), np.cumsum(values, axis=0)), axis=0)
    return (prefix[length:] - prefix[:-length]) / length


def prepare_features(graph):
    """One graph in memory; no N x N x heads reconstruction."""
    prompt = int(graph["prompt_length"])
    count = len(graph["x"]) - prompt
    channels = graph["x"].shape[1]
    source, target = graph["edge_index"]
    target = target - prompt
    incoming = np.zeros((count, channels), float)
    weighted_log = np.zeros_like(incoming)
    prompt_mass = np.zeros_like(incoming)
    edge_mass = np.zeros(len(source), float)
    degree = np.bincount(target, minlength=count)
    nonzero = np.zeros_like(incoming)
    for start in range(0, len(source), 4096):
        stop = start + 4096
        values = np.asarray(graph["edge_attr"][start:stop], dtype=float)
        rows = target[start:stop]
        np.add.at(incoming, rows, values)
        np.add.at(nonzero, rows, values > 0)
        np.add.at(weighted_log, rows, values * np.log(np.maximum(values, 1e-30)))
        selected = source[start:stop] < prompt
        np.add.at(prompt_mass, rows[selected], values[selected])
        edge_mass[start:stop] = values.sum(axis=1)
    entropy = np.log(np.maximum(incoming, 1e-30))
    entropy -= np.divide(weighted_log, incoming, out=np.zeros_like(incoming), where=incoming > 0)
    entropy = np.divide(entropy, np.log(np.maximum(nonzero, 2)),
                        out=np.zeros_like(entropy), where=incoming > 0)
    rr = np.flatnonzero(source >= prompt)
    rr = rr[np.argsort(target[rr], kind="stable")]
    return dict(graph=graph, count=count, prompt=prompt, source=source, target=target,
                incoming=incoming, entropy=entropy, prompt_mass=prompt_mass,
                mass=edge_mass, degree=degree, rr=rr, rr_target=target[rr])


def internal_edges(data, start, length):
    """Actual saved RR edges whose two endpoints lie inside this interval."""
    end = start + length
    left, right = np.searchsorted(data["rr_target"], [start, end])
    indices = data["rr"][left:right]
    return indices[data["source"][indices] >= data["prompt"] + start]


def structure_at(data, start, length):
    indices = internal_edges(data, start, length)
    end = start + length
    left, right = np.searchsorted(data["rr_target"], [start, end])
    rr_incoming = data["rr"][left:right]
    source = data["source"][indices] - data["prompt"]
    target = data["target"][indices]
    count = len(indices)
    mass = data["mass"][indices].sum()
    all_mass = data["incoming"][start:end].sum()
    hub_counts = np.bincount(source - start, minlength=length)
    concentration = float(np.square(hub_counts / count).sum()) if count else 0.
    return np.asarray([
        count / (length * (length - 1) / 2),
        count / len(rr_incoming) if len(rr_incoming) else 0.,
        mass / all_mass if all_mass else 0.,
        float(np.mean(target - source == 1)) if count else 0.,
        concentration,
        np.log(max(mass / count, 1e-12)) if count else np.log(1e-12),
        np.log1p(data["degree"][start:end].mean()),
    ])


def marginal_windows(data, length):
    """Per-head means; no head averaging. Do not include endpoint role here."""
    diagonal = data["graph"]["x"][data["prompt"]:]
    values = np.stack((diagonal, data["incoming"], data["entropy"]), axis=1)
    return window_average(values, length)


def head_profile(data, start, length):
    """Retained fractions use each head's own total, not summed-head routes."""
    end = start + length
    graph = data["graph"]
    indices = internal_edges(data, start, length)
    internal = np.zeros_like(data["incoming"][start:end])
    for left in range(0, len(indices), 4096):
        chosen = indices[left:left + 4096]
        np.add.at(internal, data["target"][chosen] - start, graph["edge_attr"][chosen])
    total = data["incoming"][start:end]
    prompt = np.divide(data["prompt_mass"][start:end], total,
                       out=np.zeros_like(total), where=total > 0)
    inside = np.divide(internal, total, out=np.zeros_like(total), where=total > 0)
    rows = (graph["x"][data["prompt"] + start:data["prompt"] + end],
            total, data["entropy"][start:end], prompt, inside)
    return np.stack([values.mean(axis=0) for values in rows])


def representation_geometry(values, left, right, length):
    """Cosines are descriptive geometry, NOT a trained readout or causal gain."""
    a = np.asarray(values[left:left + length], float)
    b = np.asarray(values[right:right + length], float)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    def within(x):
        return float((np.square(x.sum(axis=0)).sum() - np.square(x).sum()) /
                     (length * (length - 1)))
    wrong, normal = within(a), within(b)
    cross = float(a.mean(axis=0) @ b.mean(axis=0))
    return dict(error_within_cosine=wrong, normal_within_cosine=normal,
                cross_span_cosine=cross, within_minus_cross=(wrong + normal) / 2 - cross)
