import torch

from data import attention_entries, node_features


def _base(sample, node_feature, hidden_layer):
    return {
        "sample_id": sample["sample_id"],
        "source_id": sample["source_id"],
        "response_idx": sample["response_idx"],
        "token_ids": sample["token_ids"],
        "x": node_features(sample, node_feature, hidden_layer),
    }


def _relation(source, response_idx):
    return (source >= response_idx).to(torch.int8)


def _channel(entries, heads):
    return (entries["layer"] * heads + entries["head"]).to(torch.int32)


def build_multiplex(sample, tau=None, node_feature="diagonal", hidden_layer=-1):
    """Keep retained layer/head attention as directed multi-edges."""
    e = attention_entries(sample)
    keep = torch.ones(len(e["weight"]), dtype=torch.bool) if tau is None else e["weight"].float() > tau
    source, target = e["source"][keep], e["target"][keep]
    graph = _base(sample, node_feature, hidden_layer)
    graph.update({
        "edge_index": torch.stack((source, target)),
        "edge_weight": e["weight"][keep].float(),
        "edge_channel": _channel({key: value[keep] for key, value in e.items()}, sample["attention_diagonal"].shape[1]),
        "edge_type": _relation(source, sample["response_idx"]),
    })
    return graph


def build_original(sample, tau=0.05, node_feature="diagonal", hidden_layer=-1):
    """Original threshold-union graph with dense L*H edge attributes."""
    if tau < sample["attention_floor"]:
        raise ValueError("tau cannot be below the stored attention floor")
    e = attention_entries(sample)
    keep = e["weight"].float() > tau
    source, target = e["source"][keep], e["target"][keep]
    weight = e["weight"][keep].float()

    tokens = len(sample["token_ids"])
    heads = sample["attention_diagonal"].shape[1]
    channels = sample["attention_diagonal"].shape[0] * heads
    channel = (e["layer"][keep] * heads + e["head"][keep]).long()

    pair, inverse = torch.unique(target * tokens + source, sorted=True, return_inverse=True)
    edge_source, edge_target = pair % tokens, pair // tokens
    edge_attr = torch.zeros((len(pair), channels), dtype=torch.float32)
    edge_attr[inverse, channel] = weight

    graph = _base(sample, node_feature, hidden_layer)
    graph.update({
        "edge_index": torch.stack((edge_source, edge_target)),
        "edge_attr": edge_attr,
        "edge_type": _relation(edge_source, sample["response_idx"]),
    })
    return graph


def _support_mask(sample, mass):
    """Smallest observed support whose cumulative mass reaches `mass`."""
    e = attention_entries(sample)
    weight = e["weight"].float()
    if not len(weight):
        return torch.zeros(0, dtype=torch.bool)

    by_weight = torch.argsort(weight, descending=True, stable=True)
    order = by_weight[torch.argsort(e["row"][by_weight], stable=True)]
    row = e["row"][order]
    w = weight[order]

    starts = torch.ones(len(order), dtype=torch.bool)
    starts[1:] = row[1:] != row[:-1]
    cumulative = w.cumsum(0)
    before_group = torch.where(starts, cumulative - w, torch.zeros_like(w))
    before_group = torch.cummax(before_group, dim=0).values
    within_group = cumulative - before_group
    keep_ordered = within_group - w < mass

    keep = torch.zeros(len(order), dtype=torch.bool)
    keep[order] = keep_ordered
    return keep


def build_support(sample, mass=0.8, node_feature="diagonal", hidden_layer=-1):
    """Per layer/head/target mass-cover graph, preserving channel identity."""
    e = attention_entries(sample)
    keep = _support_mask(sample, mass)
    source, target = e["source"][keep], e["target"][keep]
    graph = _base(sample, node_feature, hidden_layer)
    graph.update({
        "edge_index": torch.stack((source, target)),
        "edge_weight": e["weight"][keep].float(),
        "edge_channel": _channel({key: value[keep] for key, value in e.items()}, sample["attention_diagonal"].shape[1]),
        "edge_type": _relation(source, sample["response_idx"]),
    })
    return graph


def _segmented_topk(row, score, relation, k_prompt, k_history):
    group = row * 2 + relation.long()
    by_score = torch.argsort(score, descending=True, stable=True)
    order = by_score[torch.argsort(group[by_score], stable=True)]
    ordered_group = group[order]

    starts = torch.ones(len(order), dtype=torch.bool)
    starts[1:] = ordered_group[1:] != ordered_group[:-1]
    pos = torch.arange(len(order))
    start_pos = torch.where(starts, pos, torch.zeros_like(pos))
    start_pos = torch.cummax(start_pos, dim=0).values
    rank = pos - start_pos

    limits = torch.where(ordered_group % 2 == 0, k_prompt, k_history)
    keep = torch.zeros(len(order), dtype=torch.bool)
    keep[order] = rank < limits
    return keep


def build_relation_topk(sample, k_prompt=8, k_history=8, node_feature="diagonal", hidden_layer=-1):
    """Top-k Prompt and response-history sources for every channel/target row."""
    e = attention_entries(sample)
    relation = _relation(e["source"], sample["response_idx"])
    keep = _segmented_topk(e["row"], e["weight"].float(), relation, k_prompt, k_history)
    source, target = e["source"][keep], e["target"][keep]

    graph = _base(sample, node_feature, hidden_layer)
    graph.update({
        "edge_index": torch.stack((source, target)),
        "edge_weight": e["weight"][keep].float(),
        "edge_channel": _channel({key: value[keep] for key, value in e.items()}, sample["attention_diagonal"].shape[1]),
        "edge_type": relation[keep],
    })
    return graph


def build_hypergraph(sample, tau=0.05, node_feature="diagonal", hidden_layer=-1):
    """One typed hyperedge for each (channel, target, PR-or-RR) group."""
    if tau < sample["attention_floor"]:
        raise ValueError("tau cannot be below the stored attention floor")
    e = attention_entries(sample)
    keep = e["weight"].float() > tau
    source = e["source"][keep]
    target = e["target"][keep]
    weight = e["weight"][keep].float()
    heads = sample["attention_diagonal"].shape[1]
    channel = (e["layer"][keep] * heads + e["head"][keep]).long()
    relation = _relation(source, sample["response_idx"]).long()

    if not len(source):
        groups = torch.empty(0, dtype=torch.long)
        inverse = groups
    else:
        group_key = (channel * len(sample["token_ids"]) + target) * 2 + relation
        groups, inverse = torch.unique(group_key, sorted=True, return_inverse=True)

    hyperedge = torch.arange(len(groups), dtype=torch.long)
    hyperedge_type = (groups % 2).to(torch.int8)
    q = groups // 2
    hyperedge_target = q % len(sample["token_ids"])
    hyperedge_channel = (q // len(sample["token_ids"])).to(torch.int32)

    incidence_node = torch.cat((source, hyperedge_target))
    incidence_edge = torch.cat((inverse, hyperedge))
    diagonal = sample["attention_diagonal"].reshape(-1, len(sample["token_ids"]))
    target_weight = diagonal[hyperedge_channel.long(), hyperedge_target].float()

    graph = _base(sample, node_feature, hidden_layer)
    graph.update({
        "incidence_index": torch.stack((incidence_node, incidence_edge)),
        "incidence_weight": torch.cat((weight, target_weight)),
        "hyperedge_target": hyperedge_target,
        "hyperedge_channel": hyperedge_channel,
        "hyperedge_type": hyperedge_type,
    })
    return graph


def build_graph(sample, kind, **kwargs):
    builders = {
        "original": build_original,
        "multiplex": build_multiplex,
        "support": build_support,
        "relation_topk": build_relation_topk,
        "hypergraph": build_hypergraph,
    }
    return builders[kind](sample, **kwargs)
