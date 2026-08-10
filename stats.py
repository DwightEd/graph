import torch

from data import attention_entries
from graphs import _support_mask


def row_stats(sample, support_mass=0.8):
    """Token-query statistics on retained sparse attention rows."""
    e = attention_entries(sample)
    rows = len(sample["row_ptr"]) - 1
    response_idx = sample["response_idx"]
    weight = e["weight"].float()
    row = e["row"]

    def sum_by_row(values):
        out = torch.zeros(rows, dtype=torch.float32)
        out.index_add_(0, row, values)
        return out

    prompt = e["source"] < response_idx
    history = ~prompt
    total = sum_by_row(weight)
    prompt_mass = sum_by_row(weight * prompt)
    response_mass = sum_by_row(weight * history)
    concentration = sum_by_row(weight.square()) / total.square().clamp_min(1e-12)

    support = _support_mask(sample, support_mass)
    support_size = sum_by_row(support.float())

    lag = (e["target"] - e["source"]).float()
    history_weight = weight * history
    mean_lag = sum_by_row(history_weight * lag) / sum_by_row(history_weight).clamp_min(1e-12)

    return {
        "prompt_mass": prompt_mass,
        "response_mass": response_mass,
        "concentration": concentration,
        "support_size": support_size,
        "response_mean_lag": mean_lag,
    }


def channel_stats(sample, support_mass=0.8):
    """Average row statistics over response positions, returning [L,H] tensors."""
    layers, heads, tokens = sample["attention_diagonal"].shape
    response_tokens = tokens - sample["response_idx"]
    return {
        name: values.view(layers, heads, response_tokens).mean(-1)
        for name, values in row_stats(sample, support_mass).items()
    }


def graph_stats(graph):
    result = {"nodes": int(len(graph["token_ids"]))}

    if "edge_index" in graph:
        edge_count = graph["edge_index"].shape[1]
        result["edges"] = int(edge_count)
        if edge_count:
            edge_type = graph["edge_type"]
            result["prompt_response_edges"] = int((edge_type == 0).sum())
            result["response_response_edges"] = int((edge_type == 1).sum())
            source = graph["edge_index"][0]
            counts = torch.bincount(source, minlength=result["nodes"]).float()
            counts = counts[counts > 0]
            p = counts / counts.sum()
            result["source_hub_concentration"] = float((p.square().sum()).item())
        return result

    result["hyperedges"] = int(len(graph["hyperedge_target"]))
    result["incidences"] = int(graph["incidence_index"].shape[1])
    return result
