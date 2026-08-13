"""Compact, label-free token graph representations from sparse attention.

The primary node state is the complete layer-head Lookback vector. For a
32-layer, 32-head observer this is exactly 1024 dimensions and is never
averaged before it is saved or projected. Graph propagation is deliberately
separate and compact: heads first vote for a sparse layer-level route, then
prompt moments travel over response-to-response edges for multiple hops.

All reference fitting and projection use the unlabeled train split. Test
labels are opened only after representations, scores and graph files freeze.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .graph import GraphBuildConfig, RP, RR, build_attention_graph
from .statistics import DIRECT_FEATURES, TOKEN_FEATURES, token_statistics


SCHEMA = "token-graph-representation-v1"
EXACT_FEATURES = TOKEN_FEATURES + DIRECT_FEATURES

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
class TokenRepresentationConfig:
    position_bins: int = 10
    lookback_window: int = 8
    provenance_hops: int = 2
    route_top_heads: int = 4
    csr_row_block: int = 4096
    reference_size: int = 12_000
    subspace_components: int = 32
    tail_fraction: float = 0.05
    display_mass_cover: float = 0.80
    display_edges_per_type: int = 2
    display_max_edges: int = 300
    sample_ids: tuple[str, ...] = ()
    seed: int = 42

    def validate(self):
        integer = (
            self.position_bins, self.lookback_window, self.provenance_hops,
            self.route_top_heads,
            self.csr_row_block, self.reference_size,
            self.subspace_components, self.display_edges_per_type,
            self.display_max_edges,
        )
        if min(integer) < 1:
            raise ValueError("representation limits must be positive")
        if not 0.0 < float(self.tail_fraction) <= 1.0:
            raise ValueError("tail_fraction must be in (0,1]")
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


def _csr_entries(attention, row_start, row_end):
    """Return global row IDs, columns and values for one CSR row block."""
    row_ptr = attention.response_row_ptr.long()
    starts = row_ptr[row_start:row_end]
    lengths = row_ptr[row_start + 1:row_end + 1] - starts
    entry_count = int(lengths.sum())
    if entry_count == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=row_ptr.device)
        empty_value = torch.empty(
            0, dtype=torch.float32, device=attention.response_values.device
        )
        return empty_long, empty_long, empty_value
    repeated_starts = torch.repeat_interleave(starts, lengths)
    prefix = torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
    positions = repeated_starts + torch.arange(entry_count, device=row_ptr.device) - prefix
    rows = torch.repeat_interleave(
        torch.arange(row_start, row_end, device=row_ptr.device), lengths
    )
    return (
        rows,
        attention.response_column_indices[positions].long(),
        attention.response_values[positions].float().clamp_min(0.0),
    )


def _causal_window_mean(values, width):
    """Causal variable-width mean on a [channel, token] matrix."""
    width = int(width)
    cumulative = torch.cat((
        torch.zeros((values.shape[0], 1), dtype=values.dtype, device=values.device),
        values.cumsum(dim=1),
    ), dim=1)
    token = torch.arange(values.shape[1], device=values.device)
    start = (token + 1 - width).clamp_min(0)
    total = cumulative[:, token + 1] - cumulative[:, start]
    count = (token + 1 - start).to(values.dtype)
    return total / count[None, :]


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
    for row_start in range(0, rows_count, int(csr_row_block)):
        row_end = min(row_start + int(csr_row_block), rows_count)
        rows, source, weight = _csr_entries(attention, row_start, row_end)
        if not len(rows):
            continue
        is_prompt = source < prompt_count
        if bool(is_prompt.any()):
            prompt_mass.index_add_(0, rows[is_prompt], weight[is_prompt])
        history = ~is_prompt
        if bool(history.any()):
            history_mass.index_add_(0, rows[history], weight[history])
    return (
        prompt_mass.reshape(channels, response_count),
        history_mass.reshape(channels, response_count),
    )


def direct_lookback_channels(attention, *, window=1, csr_row_block=4096):
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
        denominator > 0, prompt_mean / denominator, torch.zeros_like(denominator)
    )
    if int(window) > 1:
        lookback = _causal_window_mean(lookback, window)
    return lookback.T.reshape(
        response_count, attention.num_layers, attention.num_heads
    )


def _window_lookback_channels(current, width):
    token_count, layers, heads = current.shape
    windowed = _causal_window_mean(current.reshape(token_count, -1).T, width).T
    return windowed.reshape(token_count, layers, heads)


def _topk_grouped_edges(key, weight, keep_count):
    """Reduce repeated sparse edge keys to their fixed-divisor top-k mean."""
    by_weight = torch.argsort(weight, descending=True, stable=True)
    by_key = torch.argsort(key[by_weight], stable=True)
    order = by_weight[by_key]
    key = key[order]
    weight = weight[order]
    group_start = torch.ones(len(key), dtype=torch.bool, device=key.device)
    group_start[1:] = key[1:] != key[:-1]
    group_id = group_start.cumsum(0) - 1
    starts = torch.nonzero(group_start, as_tuple=False).flatten()
    rank = torch.arange(len(key), device=key.device) - starts[group_id]
    keep = rank < int(keep_count)
    reduced = torch.zeros(len(starts), dtype=torch.float32, device=weight.device)
    reduced.index_add_(0, group_id[keep], weight[keep])
    return key[group_start], reduced / float(keep_count)


def _salient_layer_route(attention, *, top_heads, csr_row_block):
    """Aggregate identical edges with the mean of their strongest heads.

    Missing heads contribute zero because only retained edges exist in CSR.
    The divisor is therefore always ``min(top_heads, num_heads)`` rather than
    the number of observed heads. This rewards cross-head support without
    washing a useful edge through an all-head mean.
    """
    response_count = int(attention.num_response_tokens)
    token_count = int(attention.num_tokens)
    heads = int(attention.num_heads)
    layers = int(attention.num_layers)
    keep_count = min(int(top_heads), heads)
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
            rows, source, weight = _csr_entries(attention, row_start, row_end)
            if not len(rows):
                continue
            target = rows.remainder(response_count)
            keys.append(target * token_count + source)
            weights.append(weight)
        if not keys:
            continue
        unique_key, reduced = _topk_grouped_edges(
            torch.cat(keys), torch.cat(weights), keep_count
        )
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


def compact_layer_structure(attention, *, provenance_hops=2, route_top_heads=4,
                            csr_row_block=4096, return_route=False):
    """Return ``[token, mechanism, layer]`` and optionally its exact route COO."""
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    layers = int(attention.num_layers)
    route_row, source, weight = _salient_layer_route(
        attention, top_heads=route_top_heads, csr_row_block=csr_row_block
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
    route = {
        "layer": torch.div(route_row, response_count, rounding_mode="floor"),
        "source": source,
        "target": prompt_count + route_row.remainder(response_count),
        "weight": weight,
    }
    return matrix, route


def representation_feature_names(num_layers, num_heads):
    return tuple(
        f"lookback_window:L{layer}:H{head}"
        for layer in range(int(num_layers)) for head in range(int(num_heads))
    )


def build_node_representation(lookback, *, num_layers, num_heads):
    """Flatten complete ``[layer,head]`` Lookback without averaging."""
    token_count, layers, heads = lookback.shape
    if layers != int(num_layers) or heads != int(num_heads):
        raise ValueError("Lookback tensor does not match layer-head geometry")
    return lookback.reshape(token_count, layers * heads).float()


class _PositionReservoir:
    """Deterministic per-position-bin reservoir for a bounded train reference."""

    def __init__(self, bins, size, seed):
        self.bins = int(bins)
        self.capacity = max(1, int(math.ceil(size / bins)))
        self.rng = np.random.default_rng(seed)
        self.values = None
        self.filled = np.zeros(self.bins, dtype=np.int64)
        self.seen = np.zeros(self.bins, dtype=np.int64)

    def add(self, values, position):
        values = np.asarray(values, dtype=np.float32)
        position = np.asarray(position, dtype=np.float64)
        if self.values is None:
            self.values = np.empty(
                (self.bins, self.capacity, values.shape[1]), dtype=np.float16
            )
        if values.ndim != 2 or values.shape[1] != self.values.shape[2]:
            raise ValueError("train reference vectors have inconsistent width")
        bins = np.minimum((position * self.bins).astype(int), self.bins - 1)
        for row, bin_id in zip(values, bins):
            seen = int(self.seen[bin_id])
            self.seen[bin_id] += 1
            if self.filled[bin_id] < self.capacity:
                slot = int(self.filled[bin_id])
                self.filled[bin_id] += 1
            else:
                slot = int(self.rng.integers(seen + 1))
                if slot >= self.capacity:
                    continue
            self.values[bin_id, slot] = row.astype(np.float16)

    def matrix(self):
        if self.values is None or not int(self.filled.sum()):
            raise ValueError("train split produced no reference tokens")
        values, bins = [], []
        for bin_id, count in enumerate(self.filled):
            values.append(self.values[bin_id, :count].astype(np.float32))
            bins.extend([bin_id] * int(count))
        return np.concatenate(values), np.asarray(bins, dtype=np.int16)


class _PositionScaler:
    """Median/MAD calibration fitted only on the unlabeled train reservoir."""

    def __init__(self, bins):
        self.bins = int(bins)

    @staticmethod
    def _statistics(values):
        center = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - center), axis=0)
        std = values.std(axis=0)
        scale = np.where(mad > 1e-6, mad, np.where(std > 1e-6, std, 1.0))
        return center.astype(np.float32), scale.astype(np.float32)

    def fit(self, values, bins, *, column_block=128):
        values = np.asarray(values, dtype=np.float32)
        bins = np.asarray(bins, dtype=np.int16)
        self.center = np.empty((self.bins, values.shape[1]), dtype=np.float32)
        self.scale = np.empty_like(self.center)
        for start in range(0, values.shape[1], int(column_block)):
            end = min(start + int(column_block), values.shape[1])
            global_center, global_scale = self._statistics(values[:, start:end])
            for bin_id in range(self.bins):
                selected = values[bins == bin_id, start:end]
                if len(selected) >= 3:
                    center, scale = self._statistics(selected)
                else:
                    center, scale = global_center, global_scale
                self.center[bin_id, start:end] = center
                self.scale[bin_id, start:end] = scale
        self.count = [int((bins == bin_id).sum()) for bin_id in range(self.bins)]
        return self

    def transform(self, values, position):
        bins = np.minimum(
            (np.asarray(position, dtype=np.float64) * self.bins).astype(int),
            self.bins - 1,
        )
        output = (np.asarray(values, dtype=np.float32) - self.center[bins]) / self.scale[bins]
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def report(self):
        return {
            "type": "train_reservoir_position_conditioned_median_mad",
            "position_bins": self.bins,
            "reference_tokens_per_bin": self.count,
            "fit_uses_labels": False,
        }


def _fit_reference_model(reference, reference_bins, config):
    print("[2/8] fitting train-only robust calibration and PCA", flush=True)
    scaler = _PositionScaler(config.position_bins).fit(reference, reference_bins)
    position = (reference_bins.astype(np.float32) + 0.5) / config.position_bins
    standardized = scaler.transform(reference, position)
    components = min(
        int(config.subspace_components), standardized.shape[1],
        max(1, len(standardized) - 1),
    )
    pca = PCA(
        n_components=components, svd_solver="randomized", random_state=config.seed
    ).fit(standardized)
    return scaler, pca


def _score_representation(standardized, pca, tail_fraction):
    absolute = np.abs(standardized)
    keep = max(1, int(math.ceil(absolute.shape[1] * float(tail_fraction))))
    tail = np.partition(absolute, absolute.shape[1] - keep, axis=1)[:, -keep:].mean(1)
    latent = pca.transform(standardized)
    reconstructed = pca.inverse_transform(latent)
    residual = np.mean((standardized - reconstructed) ** 2, axis=1)
    coordinates = np.zeros((len(standardized), 2), dtype=np.float32)
    coordinates[:, :min(2, latent.shape[1])] = latent[:, :2]
    return tail.astype(np.float32), residual.astype(np.float32), coordinates


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


def _exact_features(graph, current_lookback):
    """Legacy scalar diagnostics, deriving the average from frozen channels."""
    scalar = token_statistics(graph)
    lookback = (1.0 - current_lookback.reshape(len(current_lookback), -1).mean(dim=1))[:, None]
    return torch.cat((scalar, lookback), dim=1)


def _graph_record(graph, start, end):
    return {
        "sample_id": str(graph.sample_id), "start": int(start), "end": int(end),
        "token_ids": graph.token_ids.detach().cpu().numpy().astype(np.int32),
        "response_idx": int(graph.response_idx),
    }


def _safe_filename(value):
    value = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))
    return (value.strip("._") or "sample")[:120]


def _save_graph_index(directory, graph, route, representation_file, structure_file):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sample_{_safe_filename(graph['sample_id'])}.npz"
    np.savez_compressed(
        path, schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
        sample_id=np.asarray(graph["sample_id"]), token_ids=graph["token_ids"],
        response_idx=np.asarray(graph["response_idx"], dtype=np.int32),
        compact_route_layer=route["layer"],
        compact_route_source=route["source"],
        compact_route_target=route["target"],
        compact_route_weight=route["weight"],
        global_row_start=np.asarray(graph["start"], dtype=np.int64),
        global_row_end=np.asarray(graph["end"], dtype=np.int64),
        representation_file=np.asarray(representation_file.name),
        compact_layer_structure_file=np.asarray(structure_file.name),
    )
    return path


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


def _separation(labels, values):
    result = _ranking(labels, values)
    raw = result.get("auroc")
    if raw is None:
        return {
            **result, "raw_auroc_higher_for_hallucination": None,
            "separability": None, "post_hoc_direction": None,
        }
    return {
        **result, "raw_auroc_higher_for_hallucination": raw,
        "separability": max(raw, 1.0 - raw),
        "post_hoc_direction": (
            "higher_for_hallucination" if raw >= 0.5 else "lower_for_hallucination"
        ),
    }


def _signal_summary(rows):
    valid = [row for row in rows if row["separability"] is not None]
    if not valid:
        return {
            "dimensions": int(len(rows)), "evaluated_dimensions": 0,
            "median_separability": None, "q90_separability": None,
            "q95_separability": None, "best_separability": None,
            "count_ge_0_60": 0, "count_ge_0_65": 0,
            "count_ge_0_70": 0, "top_dimensions": [],
        }
    separability = np.asarray([row["separability"] for row in valid], dtype=np.float64)
    order = np.argsort(-separability, kind="stable")
    return {
        "dimensions": int(len(rows)),
        "evaluated_dimensions": int(len(valid)),
        "median_separability": float(np.median(separability)),
        "q90_separability": float(np.quantile(separability, 0.90)),
        "q95_separability": float(np.quantile(separability, 0.95)),
        "best_separability": float(separability[order[0]]),
        "count_ge_0_60": int((separability >= 0.60).sum()),
        "count_ge_0_65": int((separability >= 0.65).sum()),
        "count_ge_0_70": int((separability >= 0.70).sum()),
        "top_dimensions": [valid[index] for index in order[:20]],
    }


def _channel_rows(values, labels, num_heads, *, description=None):
    rows = []
    for channel in tqdm(
        range(values.shape[1]), desc=description, unit="channel",
        disable=description is None,
    ):
        metric = _separation(labels, np.asarray(values[:, channel]))
        rows.append({
            "layer": int(channel // num_heads),
            "head": int(channel % num_heads),
            "raw_auroc_higher_for_hallucination": metric[
                "raw_auroc_higher_for_hallucination"
            ],
            "separability": metric["separability"],
            "post_hoc_direction": metric["post_hoc_direction"],
        })
    return rows


def _evaluate_lookback(representation_file, labels, token_index, num_heads, window):
    """Audit all preserved layer-head Lookback coordinates after freezing."""
    values = np.load(representation_file, mmap_mode="r")
    rows = _channel_rows(
        values, labels, num_heads, description="[6/8] AUROC Lookback channels"
    )
    report = {"all_tokens": _signal_summary(rows)}
    full = np.asarray(token_index) >= int(window) - 1
    full_rows = _channel_rows(
        np.asarray(values[full]), np.asarray(labels)[full], num_heads,
        description="[6/8] AUROC Lookback full windows",
    )
    report["full_window_tokens_only"] = {
        **_signal_summary(full_rows),
        "tokens": int(full.sum()),
        "rule": f"response token index >= {int(window) - 1}",
    }
    pooled_labels = _window_any_positive(labels, token_index, window)
    pooled_rows = _channel_rows(
        np.asarray(values[full]), pooled_labels[full], num_heads,
        description="[6/8] AUROC Lookback-Lens window targets",
    )
    report["lookback_lens_any_positive_window_target"] = {
        **_signal_summary(pooled_rows),
        "tokens": int(full.sum()),
        "rule": (
            f"token index >= {int(window) - 1}; positive iff any of the "
            f"previous {int(window)} response tokens is hallucinated"
        ),
    }
    return report


def _evaluate_layer_structure(structure_file, labels, names):
    """Audit each compact mechanism per layer after label-free freezing."""
    array = np.load(structure_file, mmap_mode="r")
    report = {}
    for index, mechanism in enumerate(names):
        values = array[index]
        rows = []
        for layer in tqdm(
            range(values.shape[1]), desc=f"[6/8] AUROC {mechanism}", unit="layer"
        ):
            metric = _separation(labels, np.asarray(values[:, layer]))
            rows.append({
                "layer": int(layer),
                "raw_auroc_higher_for_hallucination": metric[
                    "raw_auroc_higher_for_hallucination"
                ],
                "separability": metric["separability"],
                "post_hoc_direction": metric["post_hoc_direction"],
            })
        report[mechanism] = _signal_summary(rows)
    return report


def _window_any_positive(labels, token_index, window):
    """Lookback-Lens min-pool equivalent when one denotes hallucination."""
    labels = np.asarray(labels, dtype=np.int8)
    token_index = np.asarray(token_index, dtype=np.int64)
    output = np.zeros_like(labels)
    for row, index in enumerate(token_index):
        start = row - min(int(index), int(window) - 1)
        output[row] = int(labels[start:row + 1].max())
    return output


def _read_labels(evaluation_dataset, metadata):
    for sample_id in tqdm(
        evaluation_dataset.sample_ids, desc="[5/8] open sealed labels", unit="sample"
    ):
        sample = evaluation_dataset[sample_id]
        sample.attention()
        sample.release_attention()
    store, rows = evaluation_dataset.labels(), []
    for sample_id in evaluation_dataset.sample_ids:
        sample = evaluation_dataset[sample_id]
        rows.extend(store.response_labels(sample).cpu().tolist())
        sample.release_attention()
    labels = np.asarray(rows, dtype=np.int8)
    if len(labels) != len(metadata["sample_id"]):
        raise ValueError("evaluation labels do not align with frozen token rows")
    return labels


def _display_route_edges(route, response_idx, token_count, config):
    """Choose visible edges from the exact compact route used in propagation."""
    source = route["source"]
    target = route["target"]
    weight = route["weight"]
    if not len(weight):
        return np.empty(0, dtype=np.int64)
    # A 2-D figure cannot show the layer axis. Keep the strongest layer for
    # each source-target pair; the full multilayer COO remains in the NPZ.
    pair = source.astype(np.int64) * int(token_count) + target.astype(np.int64)
    ranked_all = np.argsort(-weight, kind="stable")
    _, first = np.unique(pair[ranked_all], return_index=True)
    candidates = ranked_all[np.sort(first)]
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


def _select_samples(config, graphs, coordinates):
    available = {graph["sample_id"] for graph in graphs}
    if config.sample_ids:
        requested = list(dict.fromkeys(map(str, config.sample_ids)))
        missing = [sample_id for sample_id in requested if sample_id not in available]
        if missing:
            raise ValueError(f"sample IDs are absent from test split: {missing}")
        return requested, "user_requested_before_labels"
    ranked = []
    for graph in graphs:
        values = coordinates[graph["start"]:graph["end"]]
        dispersion = float(np.linalg.norm(values - values.mean(0), axis=1).mean())
        ranked.append((dispersion, len(values), graph["sample_id"]))
    return [max(ranked)[2]], "label_free_max_lookback_embedding_dispersion"


def _render_population(output, coordinates, scores, labels):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for label, color, name, size, alpha in (
        (0, "#2ca02c", "correct", 4, .12),
        (1, "#d62728", "hallucination", 10, .65),
    ):
        selected = labels == label
        axes[0].scatter(
            coordinates[selected, 0], coordinates[selected, 1], c=color,
            s=size, alpha=alpha, label=name, rasterized=True,
        )
    axes[0].set(
        title="Train-only PCA of complete layer-head Lookback vectors",
        xlabel="component 1", ylabel="component 2",
    )
    axes[0].legend(frameon=False)
    for axis, key, title in (
        (axes[1], "robust_tail", "Robust tail deviation"),
        (axes[2], "subspace_residual", "PCA reconstruction residual"),
    ):
        metric = _ranking(labels, scores[key])
        for label, color, name in ((0, "#2ca02c", "correct"), (1, "#d62728", "hallucination")):
            axis.hist(
                scores[key][labels == label], bins=60, density=True,
                alpha=.55, color=color, label=name,
            )
        auc = "N/A" if metric["auroc"] is None else f"{metric['auroc']:.3f}"
        axis.set(
            title=f"{title}; AUROC={auc}",
            xlabel="label-free anomaly score", ylabel="density",
        )
        axis.legend(frameon=False)
    path = output / "population_token_representations.png"
    figure.savefig(path, dpi=240)
    plt.close(figure)
    return path


def _render_sample(output, graph, route, coordinates, lookback, structure, labels,
                   names, config, num_layers, num_heads):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    response_idx = graph["response_idx"]
    selected = _display_route_edges(
        route, response_idx, len(graph["token_ids"]), config
    )
    edge_index = np.stack((route["source"][selected], route["target"][selected]))
    edge_type = np.where(edge_index[0] < response_idx, RP, RR)
    edge_score = route["weight"][selected]
    response_count = graph["end"] - graph["start"]
    response_nodes = np.arange(response_idx, response_idx + response_count)
    prompt_nodes = np.unique(edge_index[0, edge_index[0] < response_idx])
    colors = np.where(labels == 1, "#d62728", "#2ca02c")
    width = max(22, min(38, 18 + response_count * .12))
    figure, axes = plt.subplots(1, 4, figsize=(width, 6), constrained_layout=True)
    position = {
        **{int(node): (float(node), 1.0) for node in prompt_nodes},
        **{int(node): (float(node), 0.0) for node in response_nodes},
    }
    maximum = max(float(edge_score.max()) if len(edge_score) else 0.0, 1e-12)
    for edge, relation, weight in zip(edge_index.T, edge_type, edge_score):
        source, target = map(int, edge)
        if source not in position:
            continue
        axes[0].annotate("", xy=position[target], xytext=position[source], arrowprops={
            "arrowstyle": "->", "color": "#1f77b4" if relation == RP else "#777777",
            "alpha": .15 + .55 * float(weight / maximum),
            "lw": .3 + 1.5 * float(weight / maximum), "connectionstyle": "arc3,rad=.08",
        })
    if len(prompt_nodes):
        axes[0].scatter(prompt_nodes, np.ones(len(prompt_nodes)), marker="s", s=22, c="#4c78a8")
    hop = int(config.provenance_hops)
    log_mass = structure[:, names.index(
        f"prompt_provenance_log_mass_hop{hop}"
    )]
    centroid = structure[:, names.index(
        f"prompt_provenance_centroid_hop{hop}"
    )]
    path_strength = np.quantile(log_mass, .90, axis=1)
    path_centroid = np.zeros(response_count, dtype=np.float32)
    has_path = np.zeros(response_count, dtype=bool)
    for token_index in range(response_count):
        valid_layer = log_mass[token_index] > -11.5
        if valid_layer.any():
            has_path[token_index] = True
            path_centroid[token_index] = float(np.median(
                centroid[token_index, valid_layer]
            ))
    candidates = np.flatnonzero(has_path)
    if len(candidates) > 60:
        candidates = candidates[
            np.argsort(-path_strength[candidates], kind="stable")[:60]
        ]
    inherited_min = float(path_strength[candidates].min()) if len(candidates) else 0.0
    inherited_max = float(path_strength[candidates].max()) if len(candidates) else 1.0
    inherited_scale = max(inherited_max - inherited_min, 1e-6)
    for token_index in candidates:
        source_position = path_centroid[token_index] * max(response_idx - 1, 1)
        axes[0].annotate(
            "", xy=(response_idx + token_index, 0.0), xytext=(source_position, 1.0),
            arrowprops={
                "arrowstyle": "->", "linestyle": ":", "color": "#9467bd",
                "alpha": .12 + .30 * (
                    float(path_strength[token_index]) - inherited_min
                ) / inherited_scale,
                "lw": .6, "connectionstyle": "arc3,rad=.05",
            },
        )
    axes[0].scatter(response_nodes, np.zeros(response_count), c=colors, s=38,
                    edgecolors="black", linewidths=.25)
    axes[0].set(title=f"Compact route edges + strongest hop-{hop} prompt provenance",
                xlabel="absolute token position",
                yticks=(0, 1), yticklabels=("response", "prompt"))

    rr = edge_type == RR
    for source, target in edge_index[:, rr].T:
        source -= response_idx
        target -= response_idx
        axes[1].plot(
            coordinates[[source, target], 0], coordinates[[source, target], 1],
            color="#777777", alpha=.18, lw=.6,
        )
    axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=42,
                    edgecolors="black", linewidths=.25)
    for index, point in enumerate(coordinates):
        axes[1].text(point[0], point[1], str(index), fontsize=5, ha="center", va="bottom")
    axes[1].set(title="Every token in frozen node-representation space",
                xlabel="PCA component 1", ylabel="PCA component 2")
    axes[1].legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#2ca02c", label="correct"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#d62728", label="hallucination"),
        Line2D([], [], color="#777777", label="compact RR route"),
    ], frameon=False)

    lookback = lookback.reshape(response_count, num_layers, num_heads)
    layer_lookback = np.median(lookback, axis=2).T
    image = axes[2].imshow(
        layer_lookback, aspect="auto", cmap="viridis", vmin=0, vmax=1,
        interpolation="nearest",
    )
    axes[2].set(title="Windowed Lookback by layer (head median)",
                xlabel="response token index", ylabel="layer")
    figure.colorbar(image, ax=axes[2], label="Lookback ratio")

    mechanism_median = np.median(structure, axis=2).T
    lower = np.quantile(mechanism_median, .02, axis=1, keepdims=True)
    upper = np.quantile(mechanism_median, .98, axis=1, keepdims=True)
    normalized = (mechanism_median - lower) / np.maximum(upper - lower, 1e-6)
    image = axes[3].imshow(
        np.clip(normalized, 0, 1), aspect="auto", cmap="magma",
        interpolation="nearest",
    )
    axes[3].set(title="Layer-median compact graph mechanisms",
                xlabel="response token index", yticks=np.arange(len(names)),
                yticklabels=names)
    figure.colorbar(image, ax=axes[3], label="within-mechanism normalized value")
    path = output / f"sample_{_safe_filename(graph['sample_id'])}_token_graph.png"
    figure.suptitle(f"Sample {graph['sample_id']}; labels only color frozen nodes")
    figure.savefig(path, dpi=240)
    plt.close(figure)
    return path, int(len(selected)), int(len(candidates))


def _geometry(dataset):
    return {name: dataset.manifest.get(name) for name in (
        "schema", "num_layers", "num_heads", "attention_floor", "observer_model",
    )}


def discover_token_representations(train_dataset, test_dataset, evaluation_dataset,
                                   *, output_dir, config=None):
    config = TokenRepresentationConfig() if config is None else config
    num_layers = int(train_dataset.manifest["num_layers"])
    num_heads = int(train_dataset.manifest["num_heads"])
    config.validate()
    if _geometry(train_dataset) != _geometry(test_dataset):
        raise ValueError("train and test attention geometry differ")
    if list(map(str, test_dataset.sample_ids)) != list(map(str, evaluation_dataset.sample_ids)):
        raise ValueError("evaluation dataset does not match ordered test sample IDs")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("token representation output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    names = structure_names(config.provenance_hops)
    feature_names = representation_feature_names(num_layers, num_heads)
    channels = num_layers * num_heads

    print("[1/8] building bounded unlabeled train reference", flush=True)
    reservoir = _PositionReservoir(config.position_bins, config.reference_size, config.seed)
    train_sources = set()
    train_tokens = 0
    for sample_id in tqdm(train_dataset.sample_ids, desc="train Lookback nodes", unit="sample"):
        sample = train_dataset[sample_id]
        attention = sample.attention()
        lookback = direct_lookback_channels(
            attention, window=config.lookback_window, csr_row_block=config.csr_row_block
        )
        representation = build_node_representation(
            lookback, num_layers=num_layers, num_heads=num_heads
        ).detach().cpu().numpy()
        position = np.arange(len(representation), dtype=np.float32) / max(len(representation) - 1, 1)
        reservoir.add(representation, position)
        train_sources.add(str(sample.source_id))
        train_tokens += len(representation)
        sample.release_attention()
    reference, reference_bins = reservoir.matrix()
    scaler, pca = _fit_reference_model(reference, reference_bins, config)

    print("[3/8] counting test nodes for compact memory-mapped output", flush=True)
    counts = []
    count_sources = set()
    for sample_id in tqdm(test_dataset.sample_ids, desc="count test tokens", unit="sample"):
        sample = test_dataset[sample_id]
        counts.append(int(sample.attention().num_response_tokens))
        count_sources.add(str(sample.source_id))
        sample.release_attention()
    if train_sources & count_sources:
        raise ValueError("train and test source groups overlap")
    total_tokens = int(sum(counts))
    structure_file = output / "compact_layer_structure.float16.npy"
    representation_file = output / "token_node_representations.float16.npy"
    structure_gib = total_tokens * len(names) * num_layers * 2 / (1024 ** 3)
    representation_gib = total_tokens * len(feature_names) * 2 / (1024 ** 3)
    print(
        f"[3/8] test_tokens={total_tokens}; compact_structure≈{structure_gib:.2f} GiB; "
        f"node_file≈{representation_gib:.2f} GiB",
        flush=True,
    )
    required = (structure_gib + representation_gib) * (1024 ** 3) * 1.10
    free = shutil.disk_usage(output).free
    if free < required:
        raise OSError(
            f"insufficient output disk: need about {required / (1024 ** 3):.2f} GiB, "
            f"available {free / (1024 ** 3):.2f} GiB"
        )
    structure_output = np.lib.format.open_memmap(
        structure_file, mode="w+", dtype=np.float16,
        # Mechanism-major layout keeps one [token,layer] audit matrix contiguous.
        shape=(len(names), total_tokens, num_layers),
    )
    representation_output = np.lib.format.open_memmap(
        representation_file, mode="w+", dtype=np.float16,
        shape=(total_tokens, len(feature_names)),
    )
    exact_output = np.empty((total_tokens, len(EXACT_FEATURES)), dtype=np.float32)
    coordinate_output = np.empty((total_tokens, 2), dtype=np.float32)
    robust_tail = np.empty(total_tokens, dtype=np.float32)
    subspace_residual = np.empty(total_tokens, dtype=np.float32)
    metadata = _metadata_template()
    graphs, graph_paths = [], {}
    graph_directory = output / "sample_graphs"
    offset = 0

    print("[4/8] freezing 1024-D Lookback and compact layer graph state", flush=True)
    for sample_id, expected_count in tqdm(
        zip(test_dataset.sample_ids, counts), total=len(counts),
        desc="test Lookback + graph propagation", unit="sample",
    ):
        sample = test_dataset[sample_id]
        attention = sample.attention()
        current_lookback = direct_lookback_channels(
            attention, window=1, csr_row_block=config.csr_row_block
        )
        windowed_lookback = _window_lookback_channels(
            current_lookback, config.lookback_window
        )
        structure, route_tensors = compact_layer_structure(
            attention,
            provenance_hops=config.provenance_hops,
            route_top_heads=config.route_top_heads,
            csr_row_block=config.csr_row_block,
            return_route=True,
        )
        route = {
            "layer": route_tensors["layer"].detach().cpu().numpy().astype(np.int16),
            "source": route_tensors["source"].detach().cpu().numpy().astype(np.int32),
            "target": route_tensors["target"].detach().cpu().numpy().astype(np.int32),
            "weight": route_tensors["weight"].detach().cpu().numpy().astype(np.float32),
        }
        representation = build_node_representation(
            windowed_lookback, num_layers=num_layers, num_heads=num_heads
        ).detach().cpu().numpy()
        if len(representation) != expected_count:
            raise ValueError("test response length changed between passes")
        end = offset + expected_count
        position = np.arange(expected_count, dtype=np.float32) / max(expected_count - 1, 1)
        # Score the exact float16 representation that is persisted so the
        # saved reference model reproduces coordinates and scores.
        representation_saved = representation.astype(np.float16)
        standardized = scaler.transform(representation_saved, position)
        tail, residual, coordinates = _score_representation(
            standardized, pca, config.tail_fraction
        )
        graph = build_attention_graph(attention, GraphBuildConfig(selection="threshold"))
        exact = _exact_features(graph, current_lookback).detach().cpu().numpy()
        structure_output[:, offset:end] = (
            structure.detach().cpu().permute(1, 0, 2).numpy().astype(np.float16)
        )
        representation_output[offset:end] = representation_saved
        exact_output[offset:end] = exact
        coordinate_output[offset:end] = coordinates
        robust_tail[offset:end] = tail
        subspace_residual[offset:end] = residual
        _append_metadata(metadata, sample, attention)
        record = _graph_record(graph, offset, end)
        graphs.append(record)
        graph_paths[record["sample_id"]] = _save_graph_index(
            graph_directory, record, route, representation_file, structure_file
        )
        offset = end
        sample.release_attention()
    structure_output.flush()
    representation_output.flush()
    if offset != total_tokens:
        raise RuntimeError("test token count and frozen arrays do not align")
    metadata = _metadata_arrays(metadata)
    scores = {"robust_tail": robust_tail, "subspace_residual": subspace_residual}
    selected_samples, selection_rule = _select_samples(config, graphs, coordinate_output)

    reference_model_file = output / "train_reference_model.npz"
    np.savez(
        reference_model_file, schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
        position_center=scaler.center, position_scale=scaler.scale,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_explained_variance=pca.explained_variance_.astype(np.float32),
        representation_feature_names=np.asarray(feature_names),
    )

    np.savez(
        output / "token_representations_label_free.npz",
        schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
        structure_names=np.asarray(names),
        representation_feature_names=np.asarray(feature_names),
        exact_feature_names=np.asarray(EXACT_FEATURES),
        exact_token_features=exact_output,
        visualization_coordinates=coordinate_output,
        robust_tail_score=robust_tail,
        subspace_residual_score=subspace_residual,
        compact_layer_structure_file=np.asarray(structure_file.name),
        node_representation_file=np.asarray(representation_file.name),
        sample_id=metadata["sample_id"], source_id=metadata["source_id"],
        token_index=metadata["token_index"], token_id=metadata["token_id"],
        relative_position=metadata["relative_position"],
        task_type=metadata["task_type"], data_source=metadata["data_source"],
        generator_model=metadata["generator_model"],
    )
    label_free_report = {
        "schema": SCHEMA, "labels_read": False,
        "primary_node_state": {
            "name": "causal-window layer-head Lookback",
            "shape_per_token": [num_layers, num_heads],
            "flattened_dimensions": channels,
            "layer_head_averaged": False,
            "window": config.lookback_window,
        },
        "compact_layer_structure": {
            "names": list(names),
            "shape_per_token": [len(names), num_layers],
            "storage_layout": ["mechanism", "token", "layer"],
            "mixed_into_primary_node_vector": False,
        },
        "compression_semantics": {
            "attention_floor": test_dataset.manifest.get("attention_floor"),
            "missing_edges_reconstructed": False,
            "interpretation": "all masses and routes use retained CSR edges only",
        },
        "prompt_provenance": {
            "trainable": False,
            "all_head_mean_used": False,
            "route": (
                "for each layer/query/source edge, mean the strongest "
                f"{min(config.route_top_heads, num_heads)} head weights; absent heads are zero"
            ),
            "route_top_heads": min(config.route_top_heads, num_heads),
            "row_normalized": False,
            "recurrence": "S_h(layer) = A_RR_salient(layer) @ S_(h-1)(layer)",
            "state": ["prompt mass", "prompt position first moment", "prompt position second moment"],
            "hops": config.provenance_hops,
        },
        "unsupervised_scores": {
            "robust_tail": f"mean largest {config.tail_fraction:.3f} fraction of absolute train-MAD deviations",
            "subspace_residual": "train-only PCA reconstruction mean squared error",
        },
        "calibration": scaler.report(),
        "pca": {
            "fit_split": "train", "fit_uses_labels": False,
            "components": int(pca.n_components_),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        "train_tokens": int(train_tokens), "test_tokens": total_tokens,
        "sample_selection": {
            "sample_ids": selected_samples, "rule": selection_rule, "labels_used": False,
        },
        "artifacts": {
            "compact_layer_structure": str(structure_file),
            "node_representations": str(representation_file),
            "index": str(output / "token_representations_label_free.npz"),
            "train_reference_model": str(reference_model_file),
            "sample_graph_directory": str(graph_directory),
        },
        "config": asdict(config),
    }
    (output / "label_free_report.json").write_text(
        json.dumps(label_free_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    labels = _read_labels(evaluation_dataset, metadata)
    representation_read = np.load(representation_file, mmap_mode="r")
    lookback_signals = _evaluate_lookback(
        representation_file, labels, metadata["token_index"],
        num_heads, config.lookback_window,
    )
    structure_signals = _evaluate_layer_structure(structure_file, labels, names)
    exact_signals = {
        name: _separation(labels, exact_output[:, index])
        for index, name in enumerate(EXACT_FEATURES)
    }
    score_metrics = {name: _ranking(labels, value) for name, value in scores.items()}
    report = {
        **label_free_report,
        "labels_read": True,
        "labels_read_during": "evaluation_and_plot_coloring_only",
        "unsupervised_score_evaluation": score_metrics,
        "lookback_layer_head_signal_evaluation": lookback_signals,
        "compact_layer_structure_signal_evaluation": structure_signals,
        "legacy_aggregated_exact_feature_evaluation": exact_signals,
        "protocol_warning": (
            "Per-coordinate separability is post-hoc mechanism discovery on this "
            "test set, not an unsupervised deployable score. Freeze selected "
            "layers/heads on validation before a new held-out test result."
        ),
    }
    report_path = output / "token_representation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("[7/8] rendering population and selected full-token graphs", flush=True)
    population = _render_population(output, coordinate_output, scores, labels)
    graph_by_id = {graph["sample_id"]: graph for graph in graphs}
    structure_read = np.load(structure_file, mmap_mode="r")
    sample_rows = []
    for sample_id in selected_samples:
        graph = graph_by_id[sample_id]
        start, end = graph["start"], graph["end"]
        with np.load(graph_paths[sample_id], allow_pickle=False) as graph_artifact:
            route = {
                "layer": graph_artifact["compact_route_layer"],
                "source": graph_artifact["compact_route_source"],
                "target": graph_artifact["compact_route_target"],
                "weight": graph_artifact["compact_route_weight"],
            }
        structure = np.asarray(
            structure_read[:, start:end], dtype=np.float32
        ).transpose(1, 0, 2)
        lookback = np.asarray(representation_read[start:end], dtype=np.float32)
        figure, edge_count, inherited_count = _render_sample(
            output, graph, route, coordinate_output[start:end], lookback, structure,
            labels[start:end], names, config, num_layers, num_heads,
        )
        detail_path = output / f"sample_{_safe_filename(sample_id)}_graph_state.npz"
        np.savez_compressed(
            detail_path, schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
            sample_id=np.asarray(sample_id), structure_names=np.asarray(names),
            compact_layer_structure=structure.astype(np.float16),
            node_representation=lookback.astype(np.float16),
            visualization_coordinates=coordinate_output[start:end],
        )
        sample_rows.append({
            "sample_id": sample_id, "selection_rule": selection_rule,
            "response_nodes": int(end - start),
            "hallucination_tokens": int(labels[start:end].sum()),
            "display_edges": edge_count, "figure": str(figure),
            "display_multihop_prompt_provenance": inherited_count,
            "label_free_graph": str(graph_paths[sample_id]),
            "label_free_graph_state": str(detail_path),
        })
    report["population_figure"] = str(population)
    report["sample_visualizations"] = sample_rows
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[8/8] complete", flush=True)
    return {
        "output_dir": str(output), "report": str(report_path),
        "label_free_embeddings": str(output / "token_representations_label_free.npz"),
        "node_representations": str(representation_file),
        "compact_layer_structure": str(structure_file),
        "train_reference_model": str(reference_model_file),
        "sample_graph_directory": str(graph_directory),
        "population_figure": str(population), "sample_visualizations": sample_rows,
        "test_nodes": total_tokens, "unsupervised_score_metrics": score_metrics,
    }
