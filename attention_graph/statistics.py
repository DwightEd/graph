"""Label-blind full-population diagnostics for attention graphs.

These statistics are the canonical exact token state used by the graph
representation and are also available as retrospective diagnostics.
Evaluation labels are joined only by ``evaluate_statistics`` after feature
records have been frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .graph import GraphBuildConfig, RP, build_attention_graph

TOKEN_FEATURES = (
    "retained_mass",
    "prompt_mass_fraction",
    "history_mass_fraction",
    "normalized_entropy",
    "top1_share",
    "retained_concentration",
    "in_degree",
    "edge_density",
    "history_edge_fraction",
    "history_lag",
    "channel_coverage",
    "mean_edge_strength",
)

DIRECT_FEATURES = ("direct_lookback_anomaly",)


def token_statistics(graph):
    """Return one interpretable diagnostic vector per response token."""
    response_nodes = torch.nonzero(graph.response_mask, as_tuple=False).flatten()
    count = len(response_nodes)
    device = graph.node_attr.device
    output = torch.zeros((count, len(TOKEN_FEATURES)), dtype=torch.float32, device=device)
    if graph.num_edges == 0:
        return output

    source, target = graph.edge_index
    row = target - graph.response_idx
    weight = graph.edge_score.float().clamp_min(0)
    relation = graph.edge_type.long()

    mass = torch.zeros(count, device=device).index_add_(0, row, weight)
    prompt_mass = torch.zeros(count, device=device).index_add_(0, row[relation == RP], weight[relation == RP])
    history_mass = torch.zeros(count, device=device).index_add_(0, row[relation != RP], weight[relation != RP])
    safe_mass = mass.clamp_min(1e-12)
    prompt_fraction = prompt_mass / safe_mass
    history_fraction = history_mass / safe_mass

    probability = weight / safe_mass[row]
    entropy = torch.zeros(count, device=device).index_add_(
        0, row, -probability * probability.clamp_min(1e-12).log()
    )
    degree = torch.bincount(row, minlength=count).float()
    normalized_entropy = torch.zeros_like(entropy)
    multi = degree > 1
    normalized_entropy[multi] = entropy[multi] / degree[multi].log()

    concentration = torch.zeros(count, device=device).index_add_(0, row, probability.square())
    top1 = torch.zeros(count, device=device)
    top1.scatter_reduce_(0, row, probability, reduce="amax", include_self=True)

    absolute_target = graph.response_idx + torch.arange(count, device=device)
    edge_density = degree / absolute_target.float().clamp_min(1)
    history_degree = torch.bincount(row[relation != RP], minlength=count).float()
    history_edge_fraction = history_degree / degree.clamp_min(1)

    history_lag = torch.zeros(count, device=device)
    history = relation != RP
    if bool(history.any()):
        lag = (target[history] - source[history]).float() / max(count - 1, 1)
        lag_mass = torch.zeros(count, device=device).index_add_(
            0, row[history], weight[history] * lag
        )
        history_lag = lag_mass / history_mass.clamp_min(1e-12)

    trace_count = torch.bincount(graph.trace_edge_id, minlength=graph.num_edges).float()
    channel_count = torch.zeros(count, device=device).index_add_(0, row, trace_count)
    channel_coverage = channel_count / (degree.clamp_min(1) * graph.num_channels)

    mean_edge_strength = mass / degree.clamp_min(1)

    output = torch.stack((
        mass,
        prompt_fraction,
        history_fraction,
        normalized_entropy,
        top1,
        concentration,
        degree,
        edge_density,
        history_edge_fraction,
        history_lag,
        channel_coverage,
        mean_edge_strength,
    ), dim=1)
    return torch.nan_to_num(output)


def direct_lookback(attention, *, csr_row_block=4096):
    """Legacy retained-attention Lookback anomaly averaged over layer/head.

    The per-channel ratio is computed before this explicit compatibility
    average. Values below ``attention_floor`` cannot be recovered, so this is
    exact only for the retained cache, not for the original dense attention.
    Channel-preserving representations use ``direct_lookback_channels`` from
    ``token_representation`` instead.
    """
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    if response_count < 1 or prompt_count < 1:
        raise ValueError("direct Lookback requires a non-empty prompt and response")
    device = attention.response_values.device
    rows_count = int(attention.num_channels) * response_count
    row_ptr = attention.response_row_ptr.long()
    prompt_mass = torch.zeros(rows_count, dtype=torch.float32, device=device)
    history_mass = torch.zeros_like(prompt_mass)
    for row_start in range(0, rows_count, int(csr_row_block)):
        row_end = min(row_start + int(csr_row_block), rows_count)
        starts = row_ptr[row_start:row_end]
        lengths = row_ptr[row_start + 1:row_end + 1] - starts
        entry_count = int(lengths.sum())
        if not entry_count:
            continue
        repeated_starts = torch.repeat_interleave(starts, lengths)
        prefix = torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        positions = repeated_starts + torch.arange(entry_count, device=device) - prefix
        local_row = torch.repeat_interleave(
            torch.arange(row_end - row_start, device=device), lengths
        )
        source = attention.response_column_indices[positions].long()
        value = attention.response_values[positions].float().clamp_min(0.0)
        is_prompt = source < prompt_count
        if bool(is_prompt.any()):
            prompt_mass[row_start:row_end].index_add_(
                0, local_row[is_prompt], value[is_prompt]
            )
        if bool((~is_prompt).any()):
            history_mass[row_start:row_end].index_add_(
                0, local_row[~is_prompt], value[~is_prompt]
            )
    diagonal = (
        attention.attention_diagonal.float()[:, :, prompt_count:]
        .reshape(-1).clamp_min(0.0)
    )
    token_row = torch.arange(rows_count, device=device).remainder(response_count)
    prompt_mean = prompt_mass / float(prompt_count)
    generated_mean = (history_mass + diagonal) / (token_row + 1).float()
    denominator = prompt_mean + generated_mean
    ratio = torch.where(
        denominator > 0,
        prompt_mean / denominator,
        torch.zeros_like(denominator),
    )
    ratio = ratio.reshape(
        attention.num_layers, attention.num_heads, response_count
    ).permute(2, 0, 1)
    return torch.nan_to_num(1.0 - ratio.mean((1, 2)))


def collect_statistics(dataset, *, output_path, graph_config=None):
    """Compute diagnostics for every response token and every response."""
    graph_config = GraphBuildConfig() if graph_config is None else graph_config
    token_records = []
    response_records = []
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        attention = sample.attention()
        graph = build_attention_graph(attention, graph_config)
        matrix = token_statistics(graph).detach().cpu().numpy()
        lookback = direct_lookback(attention).detach().cpu().numpy()
        for token_index, vector in enumerate(matrix):
            token_records.append({
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "token_index": token_index,
                **{name: float(value) for name, value in zip(TOKEN_FEATURES, vector)},
                DIRECT_FEATURES[0]: float(lookback[token_index]),
                "task_type": sample.task_type,
                "data_source": sample.data_source,
                "generator_model": sample.generator_model,
            })
        response = {
            "sample_id": sample.sample_id,
            "source_id": sample.source_id,
            "task_type": sample.task_type,
            "data_source": sample.data_source,
            "generator_model": sample.generator_model,
            "response_tokens": len(matrix),
        }
        for column, name in enumerate(TOKEN_FEATURES):
            response[f"mean_{name}"] = float(matrix[:, column].mean())
            response[f"std_{name}"] = float(matrix[:, column].std())
        response[f"mean_{DIRECT_FEATURES[0]}"] = float(lookback.mean())
        response[f"std_{DIRECT_FEATURES[0]}"] = float(lookback.std())
        response_records.append(response)
        sample.release_attention()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "attention-graph-statistics-v1",
        "token_feature_names": list(TOKEN_FEATURES + DIRECT_FEATURES),
        "token_records": token_records,
        "response_records": response_records,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "tokens": len(token_records), "responses": len(response_records)}


def _separation(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "separability": None}
    auc = float(roc_auc_score(labels, values))
    return {
        "auroc": auc,
        "separability": max(auc, 1.0 - auc),
        "median_label_0": float(np.median(values[labels == 0])),
        "median_label_1": float(np.median(values[labels == 1])),
    }


def evaluate_statistics(dataset, *, statistics_path, output_path):
    """Open labels only now and measure all-data feature separability."""
    payload = json.loads(Path(statistics_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "attention-graph-statistics-v1":
        raise ValueError("unsupported statistics file")
    labels = dataset.labels()
    token_label_by_key = {}
    response_label_by_id = {}
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        token_labels = labels.response_labels(sample).cpu().numpy()
        response_label_by_id[sample_id] = int(token_labels.max())
        for token_index, label in enumerate(token_labels):
            token_label_by_key[(sample_id, token_index)] = int(label)
        sample.release_attention()

    token_records = payload["token_records"]
    token_y = np.asarray([
        token_label_by_key[(str(row["sample_id"]), int(row["token_index"]))]
        for row in token_records
    ])
    feature_names = tuple(payload.get("token_feature_names", TOKEN_FEATURES))
    token_result = {
        name: _separation([row[name] for row in token_records], token_y)
        for name in feature_names
    }

    response_records = payload["response_records"]
    response_y = np.asarray([response_label_by_id[str(row["sample_id"])] for row in response_records])
    response_result = {}
    for name in feature_names:
        for summary in ("mean", "std"):
            field = f"{summary}_{name}"
            response_result[field] = _separation(
                [row[field] for row in response_records], response_y
            )

    report = {
        "schema": "attention-graph-statistics-evaluation-v1",
        "labels_read_during": "evaluation_only",
        "token": token_result,
        "response": response_result,
        "token_count": len(token_records),
        "response_count": len(response_records),
        "hallucination_tokens": int(token_y.sum()),
        "hallucinated_responses": int(response_y.sum()),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
