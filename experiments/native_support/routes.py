"""Historical routing and carrier geometry from the same head/source ledger."""

import numpy as np

EPS = 1e-12
ROUTE_SCORES = (
    "routing_imbalance", "attention_displacement",
    "prompt_routing_imbalance", "prompt_attention_displacement",
)


def carrier_geometry(mass, mask):
    """f7344e2: normalize each head first, then measure source support and head rank."""
    selected = mass * mask
    total = selected.sum(-1, keepdims=True)
    valid = total[..., 0] > EPS
    probability = selected / np.maximum(total, EPS)
    mixture = (probability * valid[..., None]).sum(1) / np.maximum(valid.sum(1), 1)[:, None]
    source_entropy = -(mixture * np.log(np.maximum(mixture, EPS))).sum(-1)
    gram = probability @ probability.swapaxes(-1, -2)
    trace = np.trace(gram, axis1=-2, axis2=-1)
    rank = trace ** 2 / np.maximum(np.square(gram).sum((-1, -2)), EPS)
    anchors = selected.argmax(-1)
    anchors[~valid] = -1
    return {
        "effective_sources": np.exp(source_entropy), "effective_rank": np.maximum(rank, 1),
        "anchor": anchors, "active_heads": valid.sum(1),
    }


def temporal_support(anchors):
    """Effective anchor identities over current plus at most three past rows."""
    values = np.stack(anchors, axis=1)  # layer, time, head
    result = np.ones(len(values))
    for layer, history in enumerate(values):
        counts = np.unique(history[history >= 0], return_counts=True)[1]
        if len(counts):
            probability = counts / counts.sum()
            result[layer] = np.exp(-(probability * np.log(probability)).sum())
    return result


def route_geometry(mass, prompt, ordinary, previous, family):
    """Keep the historical full-prompt definition separate from ordinary prompt."""
    details, current = {}, {}
    for scope, mask in (("ordinary", ordinary), ("legacy", np.ones(prompt, dtype=bool))):
        mask = mask.copy()
        if mass.shape[-1] == prompt:
            mask[-1] = False  # Historical carrier geometry excludes predictor self.
        name = f"{family}_{scope}"
        geometry = carrier_geometry(mass[..., :prompt], mask)
        anchors = [item[name] for item in previous] + [geometry["anchor"]]
        temporal = temporal_support(anchors)
        details[f"{name}_log_volume"] = (
            np.log(geometry["effective_sources"]) + np.log(geometry["effective_rank"])
            + np.log(temporal)
        )
        for field in ("effective_sources", "effective_rank", "active_heads"):
            details[f"{name}_{field}"] = geometry[field]
        details[f"{name}_anchor"] = geometry["anchor"]
        current[name] = geometry["anchor"]
    return details, current


def route_scores(attention, magnitude, groups, prompt, evidence):
    """079c33a magnitude shares; a2a40fd mean attention displacement. No logit axis."""
    history = np.arange(len(groups)) >= prompt
    ordinary_prompt = groups == 0
    ordinary_history = groups == 1
    total = np.maximum(magnitude.sum((-1, -2)), EPS)
    prompt_difference = ordinary_history.astype(float) - ordinary_prompt
    result = {
        "prompt_routing_imbalance": float(((magnitude * prompt_difference).sum((-1, -2)) / total).mean()),
        "prompt_attention_displacement": float((attention * prompt_difference).sum(-1).mean()),
        "routing_imbalance": np.nan, "attention_displacement": np.nan,
        "self_attention": float(attention[..., -1].mean()),
        "strict_history_attention": float(attention[..., prompt:-1].sum(-1).mean()),
        "special_attention": float(attention[..., groups == 2].sum(-1).mean()),
    }
    if evidence is not None:
        source = np.zeros(len(groups), dtype=bool)
        source[:prompt] = evidence
        source[-1] = False  # The first prediction's own key is a separate self role.
        difference = history.astype(float) - source
        result["routing_imbalance"] = float(((magnitude * difference).sum((-1, -2)) / total).mean())
        result["attention_displacement"] = float((attention * difference).sum(-1).mean())
    return result


def focus_writes(edges, groups, focus, prompt, width, evidence=None):
    """Read and write are measured on the SAME head's current prompt window."""
    width = min(prompt, width)
    positions = focus["focus_start"][..., None] + np.arange(width)
    selected = np.take_along_axis(edges[..., :prompt], positions, axis=-1)
    active = groups[positions] == 0
    if evidence is not None:
        active &= evidence[positions]
    selected = selected * active
    positive = np.maximum(selected, 0).sum(-1)
    negative = np.maximum(-selected, 0).sum(-1)
    return {
        "focus_positive_write": positive, "focus_negative_write": negative,
        "focus_net_write": positive - negative,
    }


def measure_routes(arrays, prompt, evidence, previous, focus, width):
    attention = np.asarray(arrays["attention"], dtype=np.float32)
    magnitude = np.sqrt(np.maximum(arrays["edge_value_energy"], 0))
    groups = arrays["group_ids"]
    scores = route_scores(attention, magnitude, groups, prompt, evidence)
    details, anchors = {}, {}
    for family, mass in (("attention", attention), ("functional", magnitude)):
        geometry, current = route_geometry(mass, prompt, groups[:prompt] == 0, previous, family)
        details.update(geometry)
        anchors.update(current)
    details.update(focus_writes(arrays["edge_logit_write"], groups, focus, prompt, width, evidence))
    for name in ("focus_positive_write", "focus_negative_write", "focus_net_write"):
        scores[name] = float(details[name].sum())
    scores["negative_margin"] = -float(arrays["logit_gap"])
    return scores, details, anchors
