"""Sparse attention structure helpers and saved-sample rendering."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .graph import RP, RR
from .evidence_flow import csr_entries


SCHEMA = "token-graph-representation-v2"

DIRECT_STRUCTURE_NAMES = (
    "retained_prompt_mass",
    "retained_prompt_coverage",
    "retained_prompt_span",
    "prompt_centroid",
    "prompt_centroid_shift",
    "retained_history_mass",
    "retained_history_coverage",
    "history_edge_fraction",
    "history_lag",
    "history_lag_change",
    "history_far_mass_fraction",
)


@dataclass(frozen=True)
class _DisplayConfig:
    provenance_hops: int = 2
    csr_row_block: int = 4096
    display_mass_cover: float = 0.80
    display_edges_per_type: int = 2
    display_max_edges: int = 300

    def validate(self):
        if min(
            self.provenance_hops, self.csr_row_block,
            self.display_edges_per_type, self.display_max_edges,
        ) < 1:
            raise ValueError("display limits must be positive")
        if not 0.0 < float(self.display_mass_cover) <= 1.0:
            raise ValueError("display_mass_cover must be in (0,1]")


def structure_names(hops):
    names = list(DIRECT_STRUCTURE_NAMES)
    for hop in range(1, int(hops) + 1):
        names.extend((
            f"prompt_provenance_log_mass_hop{hop}",
            f"prompt_provenance_centroid_hop{hop}",
            f"prompt_provenance_spread_hop{hop}",
        ))
    return tuple(names)


def _temporal_abs_change(values):
    output = torch.zeros_like(values)
    if values.shape[1] > 1:
        output[:, 1:] = (values[:, 1:] - values[:, :-1]).abs()
    return output


def _channel_prompt_history_mass(attention, *, csr_row_block):
    """Return retained prompt/history mass for every layer-head query row."""
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    channels = int(attention.num_channels)
    if response_count < 1 or prompt_count < 1:
        raise ValueError("Lookback requires non-empty prompt and response")
    rows_count = channels * response_count
    device = attention.response_values.device
    prompt_mass = torch.zeros(rows_count, dtype=torch.float32, device=device)
    history_mass = torch.zeros_like(prompt_mass)
    row_ptr = attention.response_row_ptr.long()
    for row_start in range(0, rows_count, int(csr_row_block)):
        row_end = min(row_start + int(csr_row_block), rows_count)
        rows, source, weight = csr_entries(
            attention, row_start, row_end, row_ptr=row_ptr
        )
        if not len(rows):
            continue
        is_prompt = source < prompt_count
        prompt_mass.index_add_(0, rows[is_prompt], weight[is_prompt])
        history = ~is_prompt
        history_mass.index_add_(0, rows[history], weight[history])
    return (
        prompt_mass.reshape(channels, response_count),
        history_mass.reshape(channels, response_count),
    )


def direct_lookback_channels(attention, *, csr_row_block=4096):
    """Return retained Lookback as ``[token, layer, head]`` without averaging."""
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    channels = int(attention.num_channels)
    prompt_mass, history_mass = _channel_prompt_history_mass(
        attention, csr_row_block=csr_row_block
    )
    device = prompt_mass.device

    diagonal = (
        attention.attention_diagonal.float()[:, :, prompt_count:]
        .reshape(channels, response_count)
    )
    prompt_mass = prompt_mass.reshape(channels, response_count)
    history_mass = history_mass.reshape(channels, response_count)
    token = torch.arange(response_count, dtype=torch.float32, device=device)
    prompt_mean = prompt_mass / float(prompt_count)
    generated_mean = (history_mass + diagonal) / (token[None, :] + 1.0)
    denominator = prompt_mean + generated_mean
    lookback = torch.where(
        denominator > 0,
        prompt_mean / denominator,
        torch.full_like(denominator, float(attention.attention_floor)),
    )
    return lookback.T.reshape(
        response_count, attention.num_layers, attention.num_heads
    )


def _max_grouped_edges(key, weight):
    """Reduce repeated head entries to their strongest retained edge value."""
    order = torch.argsort(key, stable=True)
    key = key[order]
    weight = weight[order]
    group_start = torch.ones(len(key), dtype=torch.bool, device=key.device)
    group_start[1:] = key[1:] != key[:-1]
    group_id = group_start.cumsum(0) - 1
    reduced = torch.full(
        (int(group_start.sum()),), -torch.inf,
        dtype=torch.float32, device=weight.device,
    )
    reduced.scatter_reduce_(0, group_id, weight, reduce="amax", include_self=True)
    return key[group_start], reduced


def _salient_layer_route(attention, *, csr_row_block):
    """Build the union topology and weight each layer edge by its strongest head."""
    response_count = int(attention.num_response_tokens)
    token_count = int(attention.num_tokens)
    heads = int(attention.num_heads)
    layers = int(attention.num_layers)
    route_rows, route_sources, route_weights = [], [], []
    rows_per_layer = heads * response_count
    # Work one layer at a time. This preserves the exact grouping while
    # limiting sort temporaries to roughly 1/L of a full-sample reduction.
    for layer in range(layers):
        keys, weights = [], []
        layer_start = layer * rows_per_layer
        layer_end = layer_start + rows_per_layer
        for row_start in range(layer_start, layer_end, int(csr_row_block)):
            row_end = min(row_start + int(csr_row_block), layer_end)
            rows, source, weight = csr_entries(attention, row_start, row_end)
            if not len(rows):
                continue
            target = rows.remainder(response_count)
            keys.append(target * token_count + source)
            weights.append(weight)
        if not keys:
            continue
        unique_key, reduced = _max_grouped_edges(torch.cat(keys), torch.cat(weights))
        salient = reduced >= float(attention.attention_floor)
        unique_key = unique_key[salient]
        reduced = reduced[salient]
        route_rows.append(
            layer * response_count
            + torch.div(unique_key, token_count, rounding_mode="floor")
        )
        route_sources.append(unique_key.remainder(token_count))
        route_weights.append(reduced)
    if not route_rows:
        empty_long = torch.empty(0, dtype=torch.long, device=attention.response_values.device)
        return empty_long, empty_long, torch.empty_like(empty_long, dtype=torch.float32)
    return torch.cat(route_rows), torch.cat(route_sources), torch.cat(route_weights)


def _layer_route_tensors(attention, *, csr_row_block):
    """Return the exact sparse layer route used by propagation and artifacts."""
    route_row, source, weight = _salient_layer_route(
        attention, csr_row_block=csr_row_block
    )
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    return {
        "layer": torch.div(route_row, response_count, rounding_mode="floor"),
        "source": source,
        "target": prompt_count + route_row.remainder(response_count),
        "weight": weight,
    }

def exact_channel_route(attention, *, csr_row_block=4096):
    """Return every canonical CSR edge with its unmerged layer-head channel."""
    response_count = int(attention.num_response_tokens)
    heads = int(attention.num_heads)
    prompt_count = int(attention.response_idx)
    rows_count = int(attention.num_channels) * response_count
    rows, sources, weights = [], [], []
    for row_start in range(0, rows_count, int(csr_row_block)):
        row_end = min(row_start + int(csr_row_block), rows_count)
        current_rows, source, weight = csr_entries(attention, row_start, row_end)
        if len(current_rows):
            rows.append(current_rows)
            sources.append(source)
            weights.append(weight)
    device = attention.response_values.device
    if not rows:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_weight = torch.empty(0, dtype=torch.float32, device=device)
        return {
            "channel": empty_long, "layer": empty_long, "head": empty_long,
            "source": empty_long, "target": empty_long, "weight": empty_weight,
            "attention_floor": float(attention.attention_floor),
        }
    row = torch.cat(rows)
    channel = torch.div(row, response_count, rounding_mode="floor")
    return {
        "channel": channel,
        "layer": torch.div(channel, heads, rounding_mode="floor"),
        "head": channel.remainder(heads),
        "source": torch.cat(sources),
        "target": prompt_count + row.remainder(response_count),
        "weight": torch.cat(weights),
        "attention_floor": float(attention.attention_floor),
    }


def compact_layer_structure(attention, *, provenance_hops=2,
                            csr_row_block=4096, return_route=False):
    """Return ``[token, mechanism, layer]`` and optionally its exact route COO."""
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    layers = int(attention.num_layers)
    route_tensors = _layer_route_tensors(
        attention, csr_row_block=csr_row_block
    )
    layer_route = route_tensors["layer"]
    source = route_tensors["source"]
    weight = route_tensors["weight"]
    route_row = layer_route * response_count + (
        route_tensors["target"] - prompt_count
    )
    rows_count = layers * response_count
    device = attention.response_values.device
    prompt_mass = torch.zeros(rows_count, dtype=torch.float32, device=device)
    prompt_count_retained = torch.zeros_like(prompt_mass)
    prompt_first = torch.full_like(prompt_mass, float("inf"))
    prompt_last = torch.full_like(prompt_mass, -1.0)
    prompt_moment1 = torch.zeros_like(prompt_mass)
    prompt_moment2 = torch.zeros_like(prompt_mass)
    history_mass = torch.zeros_like(prompt_mass)
    history_count_retained = torch.zeros_like(prompt_mass)
    history_lag_mass = torch.zeros_like(prompt_mass)
    history_far_mass = torch.zeros_like(prompt_mass)
    prompt_scale = float(max(prompt_count - 1, 1))

    is_prompt = source < prompt_count
    if bool(is_prompt.any()):
        row = route_row[is_prompt]
        src = source[is_prompt]
        value = weight[is_prompt]
        normalized_source = src.float() / prompt_scale
        prompt_mass.index_add_(0, row, value)
        prompt_count_retained.index_add_(0, row, torch.ones_like(value))
        prompt_moment1.index_add_(0, row, value * normalized_source)
        prompt_moment2.index_add_(0, row, value * normalized_source.square())
        prompt_first.scatter_reduce_(0, row, src.float(), reduce="amin", include_self=True)
        prompt_last.scatter_reduce_(0, row, src.float(), reduce="amax", include_self=True)
    history = ~is_prompt
    if bool(history.any()):
        row = route_row[history]
        source_relative = source[history] - prompt_count
        target_relative = row.remainder(response_count)
        value = weight[history]
        lag = (target_relative - source_relative).float()
        lag_fraction = lag / target_relative.float().clamp_min(1.0)
        history_mass.index_add_(0, row, value)
        history_count_retained.index_add_(0, row, torch.ones_like(value))
        history_lag_mass.index_add_(0, row, value * lag_fraction)
        history_far_mass.index_add_(
            0, row, value * (lag_fraction >= 0.5).to(value.dtype)
        )

    shape = (layers, response_count)
    prompt_mass = prompt_mass.reshape(shape)
    retained = prompt_count_retained.reshape(shape)
    moment1 = prompt_moment1.reshape(shape)
    moment2 = prompt_moment2.reshape(shape)
    history_mass = history_mass.reshape(shape)
    history_retained = history_count_retained.reshape(shape)
    centroid = torch.where(
        prompt_mass > 0, moment1 / prompt_mass.clamp_min(1e-12),
        torch.zeros_like(prompt_mass),
    )
    first = prompt_first.reshape(shape)
    last = prompt_last.reshape(shape)
    span = torch.where(
        retained > 0, (last - first + 1.0) / float(prompt_count),
        torch.zeros_like(retained),
    )
    history_lag = torch.where(
        history_mass > 0,
        history_lag_mass.reshape(shape) / history_mass.clamp_min(1e-12),
        torch.zeros_like(history_mass),
    )
    history_far = torch.where(
        history_mass > 0,
        history_far_mass.reshape(shape) / history_mass.clamp_min(1e-12),
        torch.zeros_like(history_mass),
    )
    values = {
        "retained_prompt_mass": prompt_mass,
        "retained_prompt_coverage": retained / float(prompt_count),
        "retained_prompt_span": span,
        "prompt_centroid": centroid,
        "prompt_centroid_shift": _temporal_abs_change(centroid),
        "retained_history_mass": history_mass,
        "retained_history_coverage": history_retained / torch.arange(
            response_count, dtype=torch.float32, device=device
        )[None, :].clamp_min(1.0),
        "history_edge_fraction": history_retained / (
            history_retained + retained
        ).clamp_min(1.0),
        "history_lag": history_lag,
        "history_lag_change": _temporal_abs_change(history_lag),
        "history_far_mass_fraction": history_far,
    }

    state = torch.stack((prompt_mass, moment1, moment2), dim=2)
    rr_row = route_row[history]
    rr_source = source[history] - prompt_count
    rr_weight = weight[history]
    for hop in range(1, int(provenance_hops) + 1):
        output = torch.zeros_like(state).reshape(rows_count, 3)
        if len(rr_row):
            layer = torch.div(rr_row, response_count, rounding_mode="floor")
            source_row = layer * response_count + rr_source
            output.index_add_(
                0, rr_row, rr_weight[:, None] * state.reshape(rows_count, 3)[source_row]
            )
        state = output.reshape(layers, response_count, 3)
        mass, first_moment, second_moment = state.unbind(dim=2)
        centroid = torch.where(
            mass > 0, first_moment / mass.clamp_min(1e-12), torch.zeros_like(mass)
        )
        variance = torch.where(
            mass > 0,
            second_moment / mass.clamp_min(1e-12) - centroid.square(),
            torch.zeros_like(mass),
        ).clamp_min(0.0)
        log_mass = torch.where(
            mass > 0, mass.clamp_min(1e-12).log10(), torch.full_like(mass, -12.0)
        )
        values[f"prompt_provenance_log_mass_hop{hop}"] = log_mass
        values[f"prompt_provenance_centroid_hop{hop}"] = centroid
        values[f"prompt_provenance_spread_hop{hop}"] = variance.sqrt()

    names = structure_names(provenance_hops)
    if tuple(values) != names:
        raise RuntimeError("compact structure construction order differs from schema")
    matrix = torch.stack([values[name].T for name in names], dim=1)
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if not return_route:
        return matrix
    return matrix, route_tensors


def _metadata_template():
    return {name: [] for name in (
        "sample_id", "source_id", "token_index", "token_id", "relative_position",
        "task_type", "data_source", "generator_model",
    )}


def _append_metadata(metadata, sample, attention):
    count = int(attention.num_response_tokens)
    metadata["sample_id"].extend([str(sample.sample_id)] * count)
    metadata["source_id"].extend([str(sample.source_id)] * count)
    metadata["token_index"].extend(range(count))
    metadata["token_id"].extend(
        attention.token_ids[attention.response_idx:].detach().cpu().tolist()
    )
    metadata["relative_position"].extend(
        np.arange(count, dtype=np.float32) / max(count - 1, 1)
    )
    for name in ("task_type", "data_source", "generator_model"):
        metadata[name].extend([str(getattr(sample, name))] * count)


def _metadata_arrays(metadata):
    dtype = {
        "sample_id": str, "source_id": str, "token_index": np.int32,
        "token_id": np.int32, "relative_position": np.float32,
        "task_type": str, "data_source": str, "generator_model": str,
    }
    return {name: np.asarray(values, dtype=dtype[name]) for name, values in metadata.items()}


def _safe_filename(value):
    value = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))
    return (value.strip("._") or "sample")[:120]


def _ranking(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    result = {
        "n": int(len(labels)), "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else None,
    }
    if len(np.unique(labels)) < 2:
        return {**result, "auroc": None, "auprc": None}
    return {
        **result,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "correct_median": float(np.median(scores[labels == 0])),
        "hallucination_median": float(np.median(scores[labels == 1])),
    }


def _cluster_bootstrap_difference(
    labels, first, second, sample_ids, *, seed, replicates=200, description=None
):
    """Paired sample-level bootstrap; tokens from a response stay together."""
    labels = np.asarray(labels, dtype=np.int8)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    sample_ids = np.asarray(sample_ids).astype(str)
    unique = np.asarray(list(dict.fromkeys(sample_ids.tolist())))
    rows = [np.flatnonzero(sample_ids == sample_id) for sample_id in unique]
    rng = np.random.default_rng(int(seed))
    auroc, auprc = [], []
    for _ in tqdm(
        range(int(replicates)), desc=description or "paired sample bootstrap",
        unit="replicate",
    ):
        selected = rng.integers(0, len(rows), size=len(rows))
        index = np.concatenate([rows[item] for item in selected])
        current_labels = labels[index]
        if len(np.unique(current_labels)) < 2:
            continue
        auroc.append(
            roc_auc_score(current_labels, first[index])
            - roc_auc_score(current_labels, second[index])
        )
        auprc.append(
            average_precision_score(current_labels, first[index])
            - average_precision_score(current_labels, second[index])
        )

    def summary(values):
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return {"replicates": 0, "mean": None, "ci95": [None, None],
                    "probability_gain_gt_zero": None}
        return {
            "replicates": int(len(values)), "mean": float(values.mean()),
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "probability_gain_gt_zero": float(np.mean(values > 0)),
        }
    return {"auroc_difference": summary(auroc), "auprc_difference": summary(auprc)}
def _open_label_store(evaluation_dataset, description):
    """Open labels, completing the formal-cache seal when it is still closed."""
    try:
        return evaluation_dataset.labels()
    except RuntimeError as error:
        if "only after every attention sample has been processed" not in str(error):
            raise
    for sample_id in tqdm(
        evaluation_dataset.sample_ids, desc=description, unit="sample"
    ):
        sample = evaluation_dataset[sample_id]
        sample.attention()
        sample.release_attention()
    return evaluation_dataset.labels()


def _read_dataset_labels(evaluation_dataset, description):
    store, rows = _open_label_store(evaluation_dataset, description), []
    for sample_id in evaluation_dataset.sample_ids:
        sample = evaluation_dataset[sample_id]
        rows.extend(store.response_labels(sample).cpu().tolist())
        sample.release_attention()
    return np.asarray(rows, dtype=np.int8)


def _collapsed_route_indices(route, token_count, layer=None):
    """Return one maximum-weight layer route for each source-target pair."""
    source = route["source"]
    target = route["target"]
    weight = route["weight"]
    candidates = np.arange(len(weight), dtype=np.int64)
    if layer is not None:
        candidates = candidates[np.asarray(route["layer"]) == int(layer)]
    if not len(candidates):
        return candidates
    pair = (
        source[candidates].astype(np.int64) * int(token_count)
        + target[candidates].astype(np.int64)
    )
    order = np.lexsort((-weight[candidates], pair))
    ranked = candidates[order]
    ranked_pair = pair[order]
    first = np.ones(len(ranked), dtype=bool)
    first[1:] = ranked_pair[1:] != ranked_pair[:-1]
    return ranked[first]


def _display_route_edges(route, response_idx, token_count, config, layer=None):
    """Choose readable edges from the same route used in propagation."""
    source = route["source"]
    target = route["target"]
    weight = route["weight"]
    if not len(weight):
        return np.empty(0, dtype=np.int64)
    candidates = _collapsed_route_indices(route, token_count, layer=layer)
    chosen = []
    for node in range(int(response_idx), int(token_count)):
        incoming = candidates[target[candidates] == node]
        for relation in (RP, RR):
            is_relation = (
                source[incoming] < response_idx
                if relation == RP else source[incoming] >= response_idx
            )
            ids = incoming[is_relation]
            if not len(ids):
                continue
            ranked = ids[np.argsort(-weight[ids], kind="stable")]
            mass = weight[ranked]
            reached = np.flatnonzero(
                np.cumsum(mass) >= config.display_mass_cover * mass.sum()
            )
            count = int(reached[0]) + 1 if len(reached) else len(ranked)
            chosen.extend(ranked[:min(count, config.display_edges_per_type)].tolist())
    chosen = np.asarray(sorted(set(chosen)), dtype=np.int64)
    if len(chosen) > config.display_max_edges:
        order = np.argsort(-weight[chosen], kind="stable")
        chosen = np.sort(chosen[order[:config.display_max_edges]])
    return chosen


def _route_edges_by_relation(route, response_idx, response_count, layer=None):
    """Sparse RP/RR coordinates for plotting without quadratic matrices."""
    token_count = int(response_idx) + int(response_count)
    selected = _collapsed_route_indices(route, token_count, layer=layer)
    source = np.asarray(route["source"])[selected]
    target = np.asarray(route["target"])[selected]
    weight = np.asarray(route["weight"], dtype=np.float32)[selected]
    target_relative = target - int(response_idx)
    prompt = source < int(response_idx)
    history = ~prompt
    return {
        "rp_source": source[prompt],
        "rp_target": target_relative[prompt],
        "rp_weight": weight[prompt],
        "rr_source": source[history] - int(response_idx),
        "rr_target": target_relative[history],
        "rr_weight": weight[history],
        "selected": selected,
    }


def _weight_norm(values):
    """Log normalization which leaves absent edges visually blank."""
    from matplotlib.colors import LogNorm

    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0]
    if not len(positive):
        return None
    lower = max(float(positive.min()), float(np.quantile(positive, .02)))
    upper = float(positive.max())
    if upper <= lower:
        lower = max(upper * .5, np.finfo(np.float32).tiny)
    return LogNorm(vmin=lower, vmax=upper)


def _render_sample(output, graph, route, coordinates, structure, labels,
                   names, config, num_layers, *, layer=None,
                   anomaly_scores=None, anomaly_threshold=None,
                   anomaly_component=None):
    """Render weighted structure without collapsing tokens onto two lines."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    response_idx = graph["response_idx"]
    response_count = graph["end"] - graph["start"]
    token_count = len(graph["token_ids"])
    if layer is not None and not 0 <= int(layer) < int(num_layers):
        raise ValueError(f"display layer must be in [0,{int(num_layers) - 1}]")
    route_edges = _route_edges_by_relation(
        route, response_idx, response_count, layer=layer
    )
    collapsed = route_edges["selected"]
    selected = _display_route_edges(
        route, response_idx, token_count, config, layer=layer
    )
    anomaly_scores = (
        np.zeros(response_count, dtype=np.float32)
        if anomaly_scores is None else np.asarray(anomaly_scores, dtype=np.float32)
    )
    if len(anomaly_scores) != response_count:
        raise ValueError("sample anomaly scores do not align with response nodes")
    anomaly_threshold = (
        1.0 if anomaly_threshold is None else float(anomaly_threshold)
    )
    anomaly_component = (
        np.full(response_count, -1, dtype=np.int32)
        if anomaly_component is None else np.asarray(anomaly_component, dtype=np.int32)
    )
    figure, axes = plt.subplots(2, 2, figsize=(17, 14), constrained_layout=True)
    axes = axes.ravel()
    layer_text = "max over layers" if layer is None else f"layer {int(layer)}"

    # Panel 1: all response nodes in their frozen representation coordinates.
    rr_selected = selected[np.asarray(route["source"])[selected] >= response_idx]
    rr_weight = np.asarray(route["weight"], dtype=np.float32)[rr_selected]
    if len(rr_selected):
        source = np.asarray(route["source"])[rr_selected] - response_idx
        target = np.asarray(route["target"])[rr_selected] - response_idx
        valid = (
            (source >= 0) & (source < response_count)
            & (target >= 0) & (target < response_count)
        )
        source, target, rr_weight = source[valid], target[valid], rr_weight[valid]
        segments = np.stack((coordinates[source], coordinates[target]), axis=1)
        maximum = max(float(rr_weight.max()) if len(rr_weight) else 0.0, 1e-12)
        collection = LineCollection(
            segments, cmap="magma", norm=_weight_norm(rr_weight),
            linewidths=.25 + 2.5 * np.sqrt(rr_weight / maximum), alpha=.55,
            zorder=1,
        )
        collection.set_array(rr_weight)
        axes[0].add_collection(collection)
        figure.colorbar(collection, ax=axes[0], label="salient RR route weight")
    incoming = np.zeros(response_count, dtype=np.float32)
    np.add.at(incoming, route_edges["rp_target"], route_edges["rp_weight"])
    np.add.at(incoming, route_edges["rr_target"], route_edges["rr_weight"])
    node_size = 16.0 + 60.0 * np.sqrt(incoming / max(float(incoming.max()), 1e-12))
    label_edges = np.where(labels == 1, "#00cfe8", "#202020")
    nodes = axes[0].scatter(
        coordinates[:, 0], coordinates[:, 1], c=anomaly_scores, s=node_size,
        cmap="plasma", vmin=0.0, vmax=1.0,
        edgecolors=label_edges, linewidths=np.where(labels == 1, 1.2, .25), zorder=2,
    )
    figure.colorbar(nodes, ax=axes[0], label="label-free evidence-flow anomaly score")
    for index in np.flatnonzero(anomaly_scores >= anomaly_threshold):
        axes[0].annotate(
            str(index), coordinates[index], xytext=(2, 2), textcoords="offset points",
            fontsize=6, color="#8b0000",
        )
    axes[0].set(
        title=(f"All {response_count} response nodes in evidence-flow graph space\n"
               f"{len(rr_selected)} strongest visible RR routes; {layer_text}"),
        xlabel="PCA component 1", ylabel="PCA component 2",
    )
    axes[0].legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#999999",
               markeredgecolor="#202020", label="normal label outline"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#999999",
               markeredgecolor="#00cfe8", markeredgewidth=1.5,
               label="hallucination label outline (evaluation only)"),
        Line2D([], [], color="#7f3c8d", label="RR route; width/color = weight"),
    ], frameon=False)

    # Panels 2/3: sparse adjacency coordinates expose source, target and weight.
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    rp_norm = _weight_norm(route_edges["rp_weight"])
    image = None
    if len(route_edges["rp_weight"]):
        image = axes[1].scatter(
            route_edges["rp_source"], route_edges["rp_target"],
            c=route_edges["rp_weight"], s=3, cmap=cmap, norm=rp_norm,
            marker="s", linewidths=0, rasterized=True,
        )
    for token in np.flatnonzero(labels == 1):
        axes[1].axhline(token, color="#00ffff", lw=.35, alpha=.65)
    # Overlay every reachable hop-1 prompt provenance as a centroid and spread,
    # instead of drawing arbitrary long arrows between two horizontal rows.
    hop = 1
    log_mass = structure[:, names.index(
        f"prompt_provenance_log_mass_hop{hop}"
    )]
    provenance_centroid = structure[:, names.index(
        f"prompt_provenance_centroid_hop{hop}"
    )]
    provenance_spread = structure[:, names.index(
        f"prompt_provenance_spread_hop{hop}"
    )]
    if layer is None:
        strongest_layer = np.argmax(log_mass, axis=1)
        row = np.arange(response_count)
        displayed_log_mass = log_mass[row, strongest_layer]
        displayed_centroid = provenance_centroid[row, strongest_layer]
        displayed_spread = provenance_spread[row, strongest_layer]
    else:
        displayed_log_mass = log_mass[:, int(layer)]
        displayed_centroid = provenance_centroid[:, int(layer)]
        displayed_spread = provenance_spread[:, int(layer)]
    reachable = displayed_log_mass > -11.5
    provenance_rows = np.flatnonzero(reachable)
    if len(provenance_rows):
        prompt_scale = float(max(response_idx - 1, 1))
        center = displayed_centroid[reachable] * prompt_scale
        radius = displayed_spread[reachable] * prompt_scale
        path_mass = np.power(10.0, displayed_log_mass[reachable])
        relative_mass = path_mass / max(float(path_mass.max()), 1e-12)
        axes[1].hlines(
            provenance_rows, np.clip(center - radius, 0, response_idx - 1),
            np.clip(center + radius, 0, response_idx - 1),
            color="#2d6cdf", lw=.55, alpha=.55,
        )
        axes[1].scatter(
            center, provenance_rows, s=7.0 + 40.0 * np.sqrt(relative_mass),
            facecolors="none", edgecolors="#2d6cdf", linewidths=.65,
            label="hop-1 inherited prompt centroid 鍗?spread",
        )
    axes[1].set(
        title=f"Prompt閳姰esponse weighted adjacency ({layer_text})",
        xlabel="prompt source token index", ylabel="response target token index",
    )
    if image is not None:
        figure.colorbar(image, ax=axes[1], label="salient RP route weight (log scale)")
    if len(provenance_rows):
        axes[1].legend(frameon=False, loc="upper right")

    rr_norm = _weight_norm(route_edges["rr_weight"])
    image = None
    if len(route_edges["rr_weight"]):
        image = axes[2].scatter(
            route_edges["rr_source"], route_edges["rr_target"],
            c=route_edges["rr_weight"], s=3, cmap=cmap, norm=rr_norm,
            marker="s", linewidths=0, rasterized=True,
        )
    axes[2].plot(
        np.arange(response_count), np.arange(response_count),
        color="#00b5d8", lw=.8, linestyle="--", label="zero lag",
    )
    for token in np.flatnonzero(labels == 1):
        axes[2].axhline(token, color="#00ffff", lw=.35, alpha=.65)
    axes[2].set(
        title=f"Response閳姰esponse weighted adjacency ({layer_text})",
        xlabel="history source token index", ylabel="response target token index",
        xlim=(-.5, response_count - .5), ylim=(-.5, response_count - .5),
    )
    axes[2].legend(frameon=False)
    if image is not None:
        figure.colorbar(image, ax=axes[2], label="salient RR route weight (log scale)")

    # Panel 4: the detection output and graph-connected anomaly components.
    token = np.arange(response_count)
    axes[3].plot(token, anomaly_scores, color="#7f3c8d", lw=1.1, alpha=.8)
    axes[3].scatter(
        token, anomaly_scores, c=anomaly_scores, cmap="plasma", vmin=0, vmax=1,
        s=15, edgecolors=label_edges,
        linewidths=np.where(labels == 1, 1.0, .2), zorder=3,
    )
    axes[3].axhline(
        anomaly_threshold, color="#d62728", linestyle="--", lw=1.0,
        label=f"train-only threshold={anomaly_threshold:.3f}",
    )
    for component_id in sorted(set(anomaly_component[anomaly_component >= 0].tolist())):
        members = np.flatnonzero(anomaly_component == component_id)
        axes[3].axvspan(
            members.min() - .45, members.max() + .45,
            color="#ff7f0e", alpha=.10,
        )
    if bool((labels == 1).any()):
        axes[3].scatter(
            token[labels == 1], np.full(int(labels.sum()), .02), marker="x",
            c="#00b5d8", s=25, linewidths=.8,
            label="hallucination label (evaluation only)",
        )
    axes[3].set(
        title="Unsupervised node scores and graph-connected anomaly components",
        xlabel="response token index", ylabel="calibrated anomaly score",
        xlim=(-.5, response_count - .5), ylim=(-.02, 1.02),
    )
    axes[3].legend(frameon=False)

    layer_suffix = "all_layers" if layer is None else f"layer_{int(layer)}"
    path = output / (
        f"sample_{_safe_filename(graph['sample_id'])}_attention_structure_"
        f"{layer_suffix}.png"
    )
    figure.suptitle(
        f"Sample {graph['sample_id']}: evidence-flow geometry, weighted adjacency, and detection\n"
        "cyan outlines/marks are labels opened only after scores were frozen",
        fontsize=14,
    )
    figure.savefig(path, dpi=240)
    plt.close(figure)
    return path, {
        "visible_rr_edges": int(len(rr_selected)),
        "collapsed_route_edges": int(len(collapsed)),
        "rp_route_edges": int(len(route_edges["rp_weight"])),
        "rr_route_edges": int(len(route_edges["rr_weight"])),
        "display_layer": None if layer is None else int(layer),
    }


def render_saved_sample(dataset, *, output_dir, sample_id, layer=None):
    """Re-render one saved sample without recomputing train/test features."""
    output = Path(output_dir)
    index_path = output / "token_representations_label_free.npz"
    report_path = output / "label_free_report.json"
    graph_path = output / "sample_graphs" / f"sample_{_safe_filename(sample_id)}.npz"
    for required in (index_path, report_path, graph_path):
        if not required.exists():
            raise FileNotFoundError(f"saved artifact is missing: {required}")
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    if saved_report["schema"] != SCHEMA:
        raise ValueError(
            f"unsupported token graph schema: {saved_report['schema']!r}; "
            f"expected {SCHEMA!r}"
        )
    saved_config = saved_report.get("config", {})
    with np.load(index_path, allow_pickle=False) as index:
        if str(index["schema"]) != SCHEMA:
            raise ValueError(
                f"unsupported token graph index schema: {str(index['schema'])!r}; "
                f"expected {SCHEMA!r}"
            )
        coordinates_all = np.asarray(index["visualization_coordinates"], dtype=np.float32)
        saved_sample_ids = np.asarray(index["sample_id"]).astype(str)
    with np.load(graph_path, allow_pickle=False) as artifact:
        if str(artifact["schema"]) != SCHEMA:
            raise ValueError(
                f"unsupported sample graph schema: {str(artifact['schema'])!r}; "
                f"expected {SCHEMA!r}"
            )
        start = int(artifact["global_row_start"])
        end = int(artifact["global_row_end"])
        graph = {
            "sample_id": str(artifact["sample_id"]),
            "start": start, "end": end,
            "token_ids": artifact["token_ids"],
            "response_idx": int(artifact["response_idx"]),
        }
        exact_route_sha256 = str(artifact["exact_route_sha256"])
        anomaly_score = artifact["true_graph_score"]
        anomaly_threshold = float(artifact["anomaly_threshold"])
        anomaly_component = artifact["anomaly_component"]
    if not np.all(saved_sample_ids[start:end] == str(sample_id)):
        raise ValueError("saved sample rows do not match the requested sample ID")
    label_cache = output / "evaluation_token_labels.npy"
    if label_cache.exists():
        all_labels = np.load(label_cache, mmap_mode="r")
        label_source = "saved_evaluation_cache"
    else:
        all_labels = _read_dataset_labels(
            dataset, "[render] open sealed labels"
        )
        if len(all_labels) != len(coordinates_all):
            raise ValueError("evaluation labels do not align with saved token rows")
        np.save(label_cache, all_labels)
        label_source = "dataset_after_formal_seal"
    if len(all_labels) != len(coordinates_all):
        raise ValueError("cached evaluation labels do not align with saved token rows")
    labels = np.array(all_labels[start:end], dtype=np.int8, copy=True)
    if len(labels) != end - start:
        raise ValueError("saved sample rows do not align with evaluation labels")
    if isinstance(all_labels, np.memmap):
        all_labels._mmap.close()
    config = _DisplayConfig(
        provenance_hops=int(saved_config.get("provenance_hops", 2)),
        csr_row_block=int(saved_config.get("csr_row_block", 4096)),
        display_mass_cover=float(saved_config.get("display_mass_cover", .80)),
        display_edges_per_type=int(saved_config.get("display_edges_per_type", 2)),
        display_max_edges=int(saved_config.get("display_max_edges", 300)),
    )
    config.validate()
    if str(dataset.rows[str(sample_id)]["sha256"]) != exact_route_sha256:
        raise ValueError("canonical attention sample differs from the frozen graph input")
    sample = dataset[str(sample_id)]
    attention = sample.attention()
    structure_tensor, route_tensors = compact_layer_structure(
        attention, provenance_hops=config.provenance_hops,
        csr_row_block=config.csr_row_block, return_route=True,
    )
    if int(attention.num_response_tokens) != end - start:
        raise ValueError("canonical response length differs from the frozen graph rows")
    structure = structure_tensor.detach().cpu().numpy().astype(np.float32)
    route = {
        name: route_tensors[name].detach().cpu().numpy()
        for name in ("layer", "source", "target", "weight")
    }
    names = structure_names(config.provenance_hops)
    sample.release_attention()
    figure, stats = _render_sample(
        output, graph, route, coordinates_all[start:end], structure,
        labels, names, config, int(dataset.manifest["num_layers"]), layer=layer,
        anomaly_scores=anomaly_score, anomaly_threshold=anomaly_threshold,
        anomaly_component=anomaly_component,
    )
    return {
        "sample_id": str(sample_id), "attention_structure_figure": str(figure),
        "hallucination_tokens": int(labels.sum()),
        "response_nodes": int(len(labels)), "visualization_stats": stats,
        "features_recomputed": False,
        "visualization_structure_recomputed": True,
        "label_source": label_source,
        "predicted_anomaly_nodes": int(
            np.count_nonzero(anomaly_score >= anomaly_threshold)
        ),
    }
