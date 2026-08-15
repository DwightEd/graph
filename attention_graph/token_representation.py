"""Label-free token representations and exact-channel causal evidence flow.

The primary node state is the complete layer-head Lookback vector. For a
32-layer, 32-head observer this is exactly 1024 dimensions. Exact CSR channels
provide prompt and response evidence flows without projecting heads or
binning prompt positions. A lag-matched RR rewire is the topology null.

All reference fitting and projection use the unlabeled train split. Test
labels are opened only after representations, scores and graph files freeze.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .graph import RP, RR
from .evidence_flow import (
    anomaly_components_from_attention,
    csr_entries,
    direct_field_names,
    lookback_evidence_from_attention,
    propagation_field_names,
)


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
class TokenRepresentationConfig:
    position_bins: int = 10
    provenance_hops: int = 2
    bootstrap_replicates: int = 200
    csr_row_block: int = 65536
    reference_size: int = 12_000
    subspace_components: int = 32
    tail_fraction: float = 0.05
    anomaly_quantile: float = 0.95
    display_mass_cover: float = 0.80
    display_edges_per_type: int = 2
    display_max_edges: int = 300
    display_layer: int | None = None
    sample_ids: tuple[str, ...] = ()
    seed: int = 42

    def validate(self):
        integer = (
            self.position_bins, self.provenance_hops,
            self.bootstrap_replicates,
            self.csr_row_block, self.reference_size,
            self.subspace_components, self.display_edges_per_type,
            self.display_max_edges,
        )
        if min(integer) < 1:
            raise ValueError("representation limits must be positive")
        if not 0.0 < float(self.tail_fraction) <= 1.0:
            raise ValueError("tail_fraction must be in (0,1]")
        if not 0.0 < float(self.anomaly_quantile) < 1.0:
            raise ValueError("anomaly_quantile must be in (0,1)")
        if not 0.0 < float(self.display_mass_cover) <= 1.0:
            raise ValueError("display_mass_cover must be in (0,1]")
        if self.display_layer is not None and int(self.display_layer) < 0:
            raise ValueError("display_layer must be nonnegative")


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


def representation_feature_names(num_layers, num_heads):
    return tuple(
        f"lookback:L{layer}:H{head}"
        for layer in range(int(num_layers)) for head in range(int(num_heads))
    )


def build_node_representation(lookback, *, num_layers, num_heads):
    """Flatten complete ``[layer,head]`` Lookback without averaging."""
    token_count, layers, heads = lookback.shape
    if layers != int(num_layers) or heads != int(num_heads):
        raise ValueError("Lookback tensor does not match layer-head geometry")
    return lookback.reshape(token_count, layers * heads).float()


class _PositionReservoir:
    """Train reference with ``size`` shared across all position bins."""

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

    @property
    def maximum_rows(self):
        return self.bins * self.capacity


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


def _robust_tail(standardized, tail_fraction):
    absolute = np.abs(standardized)
    keep = max(1, int(math.ceil(absolute.shape[1] * float(tail_fraction))))
    return np.partition(
        absolute, absolute.shape[1] - keep, axis=1
    )[:, -keep:].mean(1).astype(np.float32)


def _score_representation(standardized, pca, tail_fraction):
    tail = _robust_tail(standardized, tail_fraction)
    latent = pca.transform(standardized)
    reconstructed = pca.inverse_transform(latent)
    residual = np.mean((standardized - reconstructed) ** 2, axis=1)
    coordinates = np.zeros((len(standardized), 2), dtype=np.float32)
    coordinates[:, :min(2, latent.shape[1])] = latent[:, :2]
    return tail.astype(np.float32), residual.astype(np.float32), coordinates


def _empirical_rank(sorted_reference, values):
    reference = np.asarray(sorted_reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if reference.ndim != 1 or not len(reference):
        raise ValueError("empirical calibration requires a one-dimensional reference")
    return (
        (np.searchsorted(reference, values, side="right") + 0.5)
        / (len(reference) + 1.0)
    ).astype(np.float32)


class _ScoreCalibrator:
    """Turn two unsupervised diagnostics into one train-calibrated tail score."""

    def fit(self, tail, residual):
        self.tail = np.sort(np.asarray(tail, dtype=np.float32))
        self.residual = np.sort(np.asarray(residual, dtype=np.float32))
        combined = np.maximum(
            _empirical_rank(self.tail, tail),
            _empirical_rank(self.residual, residual),
        )
        self.combined = np.sort(combined)
        self.reference_scores = self.transform(tail, residual)
        return self

    def transform(self, tail, residual):
        combined = np.maximum(
            _empirical_rank(self.tail, tail),
            _empirical_rank(self.residual, residual),
        )
        return _empirical_rank(self.combined, combined)


class _ViewReference:
    """Position-conditioned robust reference and low-rank normal subspace."""

    def __init__(self, name, config):
        self.name = str(name)
        self.config = config

    def fit(self, values, bins):
        self.scaler = _PositionScaler(self.config.position_bins).fit(values, bins)
        position = (np.asarray(bins, dtype=np.float32) + 0.5) / self.config.position_bins
        standardized = self.scaler.transform(values, position)
        components = min(
            int(self.config.subspace_components), standardized.shape[1],
            max(1, len(standardized) - 1),
        )
        self.pca = PCA(
            n_components=components, svd_solver="randomized",
            random_state=self.config.seed,
        ).fit(standardized)
        tail, residual, _ = _score_representation(
            standardized, self.pca, self.config.tail_fraction
        )
        self.calibrator = _ScoreCalibrator().fit(tail, residual)
        self.reference_rows = int(len(values))
        return self

    def transform(self, values, position):
        standardized = self.scaler.transform(values, position)
        tail, residual, coordinates = _score_representation(
            standardized, self.pca, self.config.tail_fraction
        )
        return {
            "score": self.calibrator.transform(tail, residual),
            "tail": tail, "subspace_residual": residual,
            "coordinates": coordinates,
        }

    @property
    def reference_scores(self):
        return self.calibrator.reference_scores

    def report(self):
        return {
            "calibration": self.scaler.report(),
            "input_dimensions": int(self.scaler.center.shape[1]),
            "components": int(self.pca.n_components_),
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "score": "train-ECDF(max(robust-tail, PCA reconstruction residual))",
            "fit_split": "train", "fit_uses_labels": False,
            "reference_rows": self.reference_rows,
        }


def _save_view_reference(payload, prefix, model):
    payload[f"{prefix}_position_center"] = model.scaler.center
    payload[f"{prefix}_position_scale"] = model.scaler.scale
    payload[f"{prefix}_pca_mean"] = model.pca.mean_.astype(np.float32)
    payload[f"{prefix}_pca_components"] = model.pca.components_.astype(np.float32)
    payload[f"{prefix}_pca_explained_variance"] = (
        model.pca.explained_variance_.astype(np.float32)
    )
    payload[f"{prefix}_tail_reference"] = model.calibrator.tail
    payload[f"{prefix}_residual_reference"] = model.calibrator.residual
    payload[f"{prefix}_combined_reference"] = model.calibrator.combined


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


def _graph_record(sample, attention, start, end):
    return {
        "sample_id": str(sample.sample_id), "start": int(start), "end": int(end),
        "token_ids": attention.token_ids.detach().cpu().numpy().astype(np.int32),
        "response_idx": int(attention.response_idx),
    }


def _safe_filename(value):
    value = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))
    return (value.strip("._") or "sample")[:120]


def _save_graph_index(
    directory, graph, representation_file, *,
    canonical_split, canonical_sha256, rewire_seed,
    true_graph_file=None, anomaly_score=None, anomaly_threshold=None,
    anomaly_mask=None, anomaly_component=None, rewire_audit=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sample_{_safe_filename(graph['sample_id'])}.npz"
    payload = dict(
        schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
        sample_id=np.asarray(graph["sample_id"]), token_ids=graph["token_ids"],
        response_idx=np.asarray(graph["response_idx"], dtype=np.int32),
        global_row_start=np.asarray(graph["start"], dtype=np.int64),
        global_row_end=np.asarray(graph["end"], dtype=np.int64),
        representation_file=np.asarray(representation_file.name),
        canonical_split=np.asarray(str(canonical_split)),
        exact_route_sample_id=np.asarray(graph["sample_id"]),
        exact_route_sha256=np.asarray(str(canonical_sha256)),
        rewire_seed=np.asarray(rewire_seed, dtype=np.int64),
    )
    if true_graph_file is not None:
        payload["true_graph_representation_file"] = np.asarray(true_graph_file.name)
    if anomaly_score is not None:
        payload.update(
            true_graph_score=np.asarray(anomaly_score, dtype=np.float32),
            anomaly_threshold=np.asarray(anomaly_threshold, dtype=np.float32),
            anomaly_mask=np.asarray(anomaly_mask, dtype=bool),
            anomaly_component=np.asarray(anomaly_component, dtype=np.int32),
        )
    if rewire_audit is not None:
        payload.update(
            rewire_rr_edges=np.asarray(rewire_audit["rr_edges"], dtype=np.int64),
            rewire_changed_edges=np.asarray(
                rewire_audit["rewired_changed_edges"], dtype=np.int64
            ),
            rewire_changed_fraction=np.asarray(
                rewire_audit["rewired_changed_fraction"], dtype=np.float32
            ),
        )
    np.savez_compressed(path, **payload)
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


def _threshold_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(scores) >= float(threshold)
    true_positive = int(np.count_nonzero(predicted & (labels == 1)))
    false_positive = int(np.count_nonzero(predicted & (labels == 0)))
    false_negative = int(np.count_nonzero(~predicted & (labels == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "threshold": float(threshold), "predicted_nodes": int(predicted.sum()),
        "predicted_fraction": float(predicted.mean()),
        "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
    }


def _metrics_by_group(labels, scores, groups):
    groups = np.asarray(groups).astype(str)
    return {
        value: _ranking(np.asarray(labels)[groups == value], np.asarray(scores)[groups == value])
        for value in sorted(set(groups.tolist()))
    }


def _positive_runs_from_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask.astype(np.int8), (1, 1))
    change = np.diff(padded)
    return [
        np.arange(start, end, dtype=np.int32)
        for start, end in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))
    ]


def _component_detection_metrics(labels, graphs, graph_paths):
    """Measure whether fixed-threshold graph components localize labeled spans."""
    true_best_iou, predicted_best_iou = [], []
    true_overlap, predicted_overlap = [], []
    predicted_count = 0
    for graph in graphs:
        start, end = graph["start"], graph["end"]
        true_groups = _positive_runs_from_mask(np.asarray(labels)[start:end] == 1)
        with np.load(graph_paths[graph["sample_id"]], allow_pickle=False) as artifact:
            component = artifact["anomaly_component"]
        predicted_groups = [
            np.flatnonzero(component == component_id)
            for component_id in sorted(set(component[component >= 0].tolist()))
        ]
        predicted_count += len(predicted_groups)

        def best_iou(group, candidates):
            current = set(map(int, group))
            values = []
            for candidate in candidates:
                other = set(map(int, candidate))
                values.append(len(current & other) / max(len(current | other), 1))
            return max(values, default=0.0)

        for group in true_groups:
            value = best_iou(group, predicted_groups)
            true_best_iou.append(value)
            true_overlap.append(value > 0)
        for group in predicted_groups:
            value = best_iou(group, true_groups)
            predicted_best_iou.append(value)
            predicted_overlap.append(value > 0)
    return {
        "true_hallucination_spans": int(len(true_best_iou)),
        "predicted_graph_components": int(predicted_count),
        "mean_best_iou_per_true_span": (
            float(np.mean(true_best_iou)) if true_best_iou else None
        ),
        "true_span_any_overlap_rate": (
            float(np.mean(true_overlap)) if true_overlap else None
        ),
        "mean_best_iou_per_predicted_component": (
            float(np.mean(predicted_best_iou)) if predicted_best_iou else None
        ),
        "predicted_component_any_overlap_rate": (
            float(np.mean(predicted_overlap)) if predicted_overlap else None
        ),
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


def _evaluate_lookback(representation_file, labels, num_heads):
    """Audit all preserved layer-head Lookback coordinates after freezing."""
    values = np.load(representation_file, mmap_mode="r")
    rows = _channel_rows(
        values, labels, num_heads, description="[6/8] AUROC Lookback channels"
    )
    return {"all_tokens": _signal_summary(rows)}


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


def _read_labels(evaluation_dataset, metadata):
    labels = _read_dataset_labels(
        evaluation_dataset, "[5/8] open sealed labels"
    )
    if len(labels) != len(metadata["sample_id"]):
        raise ValueError("evaluation labels do not align with frozen token rows")
    return labels


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
    # Sort by pair first and descending weight second; keep the first member.
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
    figure, axes = plt.subplots(1, 4, figsize=(21, 5), constrained_layout=True)
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
        title="Train-only PCA visualization of the raw true graph",
        xlabel="component 1", ylabel="component 2",
    )
    axes[0].legend(frameon=False)
    for axis, key, title in (
        (axes[1], "token_only", "Token-only reference"),
        (axes[2], "true_graph", "True evidence-flow graph"),
        (axes[3], "rewired_graph", "Rewired graph control"),
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
            label="hop-1 inherited prompt centroid ± spread",
        )
    axes[1].set(
        title=f"Prompt→response weighted adjacency ({layer_text})",
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
        title=f"Response→response weighted adjacency ({layer_text})",
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
    config = TokenRepresentationConfig(
        provenance_hops=int(saved_config.get("provenance_hops", 2)),
        csr_row_block=int(saved_config.get("csr_row_block", 4096)),
        display_mass_cover=float(saved_config.get("display_mass_cover", .80)),
        display_edges_per_type=int(saved_config.get("display_edges_per_type", 2)),
        display_max_edges=int(saved_config.get("display_max_edges", 300)),
        display_layer=layer,
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


def _geometry(dataset):
    return {name: dataset.manifest.get(name) for name in (
        "schema", "num_layers", "num_heads", "attention_floor", "observer_model",
    )}


def _manifest_fingerprint(dataset):
    keys = (
        "schema", "index_sha256", "attention_cache_fingerprint",
        "num_layers", "num_heads", "attention_floor", "observer_model",
    )
    result = {
        key: dataset.manifest[key]
        for key in keys if key in dataset.manifest
    }
    inventory = "\n".join(
        f"{sample_id}\t{dataset.rows[str(sample_id)]['sha256']}"
        for sample_id in dataset.sample_ids
    )
    result["sample_inventory_sha256"] = hashlib.sha256(
        inventory.encode("utf-8")
    ).hexdigest()
    return result


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
    direct_names = direct_field_names(num_layers, num_heads)
    propagation_names = propagation_field_names(num_layers, num_heads)
    view_names = (
        "scalar_only", "token_only", "prompt_graph", "response_graph",
        "true_graph", "rewired_graph", "direct_marginals",
    )

    print("[1/8] building train-only exact-channel references", flush=True)
    reservoirs = {
        name: _PositionReservoir(
            config.position_bins, config.reference_size, config.seed
        )
        for name in view_names
    }
    train_sources = set()
    train_tokens = 0
    for sample_id in tqdm(
        train_dataset.sample_ids, desc="train evidence-flow nodes", unit="sample"
    ):
        sample = train_dataset[sample_id]
        attention = sample.attention()
        lookback, direct, propagation, rewired_propagation, _ = (
            lookback_evidence_from_attention(
                attention, csr_row_block=config.csr_row_block,
                sample_id=sample_id, seed=config.seed,
            )
        )
        representation = build_node_representation(
            lookback, num_layers=num_layers, num_heads=num_heads
        )
        prompt_flow, response_flow = propagation[:, :channels], propagation[:, channels:]
        rewired_response_flow = rewired_propagation[:, channels:]
        reference_views = {
            "scalar_only": representation.mean(dim=1, keepdim=True),
            "token_only": representation,
            "prompt_graph": torch.cat((representation, prompt_flow), dim=1),
            "response_graph": torch.cat((representation, response_flow), dim=1),
            "true_graph": torch.cat((representation, prompt_flow, response_flow), dim=1),
            "rewired_graph": torch.cat((representation, prompt_flow, rewired_response_flow), dim=1),
            "direct_marginals": direct,
        }
        position = np.arange(len(representation), dtype=np.float32) / max(
            len(representation) - 1, 1
        )
        for view, values in reference_views.items():
            reservoirs[view].add(values.detach().cpu().numpy(), position)
        train_sources.add(str(sample.source_id))
        train_tokens += len(representation)
        sample.release_attention()
    print("[2/8] fitting seven train-only one-class reference models", flush=True)
    references, reference_bins = {}, None
    for view, reservoir in reservoirs.items():
        values, bins = reservoir.matrix()
        if reference_bins is None:
            reference_bins = bins
        elif not np.array_equal(reference_bins, bins):
            raise RuntimeError("view reservoirs did not retain aligned train tokens")
        references[view] = _ViewReference(view, config).fit(values, bins)
    del reservoirs, values, bins, reference_bins
    anomaly_threshold = float(np.quantile(
        references["true_graph"].reference_scores, config.anomaly_quantile
    ))

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
    representation_file = output / "token_node_representations.float16.npy"
    true_graph_file = output / "true_graph_node_representations.float16.npy"
    representation_gib = total_tokens * len(feature_names) * 2 / (1024 ** 3)
    true_graph_gib = total_tokens * (3 * channels) * 2 / (1024 ** 3)
    print(
        f"[3/8] test_tokens={total_tokens}; node_file~{representation_gib:.2f} GiB; "
        f"raw_true_graph~{true_graph_gib:.2f} GiB",
        flush=True,
    )
    required = (
        representation_gib + true_graph_gib
    ) * (1024 ** 3) * 1.10
    free = shutil.disk_usage(output).free
    if free < required:
        raise OSError(
            f"insufficient output disk: need about {required / (1024 ** 3):.2f} GiB, "
            f"available {free / (1024 ** 3):.2f} GiB"
        )
    representation_output = np.lib.format.open_memmap(
        representation_file, mode="w+", dtype=np.float16,
        shape=(total_tokens, len(feature_names)),
    )
    true_graph_output = np.lib.format.open_memmap(
        true_graph_file, mode="w+", dtype=np.float16,
        shape=(total_tokens, 3 * channels),
    )
    coordinate_output = np.empty((total_tokens, 2), dtype=np.float32)
    scores = {
        name: np.empty(total_tokens, dtype=np.float32)
        for name in (
            *view_names,
        )
    }
    metadata = _metadata_template()
    graphs, graph_paths = [], {}
    rewire_audits = []
    graph_directory = output / "sample_graphs"
    offset = 0

    print("[4/8] freezing raw Lookback and raw true-graph representations", flush=True)
    for sample_id, expected_count in tqdm(
        zip(test_dataset.sample_ids, counts), total=len(counts),
        desc="test Lookback + graph propagation", unit="sample",
    ):
        sample = test_dataset[sample_id]
        attention = sample.attention()
        (
            current_lookback, direct, propagation,
            rewired_propagation, rewire_audit,
        ) = lookback_evidence_from_attention(
            attention, csr_row_block=config.csr_row_block,
            sample_id=sample_id, seed=config.seed,
        )
        representation_tensor = build_node_representation(
            current_lookback, num_layers=num_layers, num_heads=num_heads
        )
        representation = representation_tensor.detach().cpu().numpy()
        if len(representation) != expected_count:
            raise ValueError("test response length changed between passes")
        end = offset + expected_count
        position = np.arange(expected_count, dtype=np.float32) / max(expected_count - 1, 1)
        representation_saved = representation.astype(np.float16)
        prompt_flow, response_flow = propagation[:, :channels], propagation[:, channels:]
        rewired_response_flow = rewired_propagation[:, channels:]
        true_graph = torch.cat((
            representation_tensor,
            prompt_flow, response_flow,
        ), dim=1).detach().cpu().numpy()
        test_views = {
            "scalar_only": representation.mean(axis=1, keepdims=True),
            "token_only": representation,
            "prompt_graph": np.concatenate((representation, prompt_flow.detach().cpu().numpy()), axis=1),
            "response_graph": np.concatenate((representation, response_flow.detach().cpu().numpy()), axis=1),
            "true_graph": true_graph,
            "rewired_graph": np.concatenate((representation, prompt_flow.detach().cpu().numpy(), rewired_response_flow.detach().cpu().numpy()), axis=1),
            "direct_marginals": direct.detach().cpu().numpy(),
        }
        transformed = {
            view: references[view].transform(values, position)
            for view, values in test_views.items()
        }
        for view in test_views:
            scores[view][offset:end] = transformed[view]["score"]
        true_score = transformed["true_graph"]["score"]
        coordinates = transformed["true_graph"]["coordinates"]
        representation_output[offset:end] = representation_saved
        true_graph_output[offset:end] = true_graph.astype(np.float16)
        coordinate_output[offset:end] = coordinates
        _append_metadata(metadata, sample, attention)
        record = _graph_record(sample, attention, offset, end)
        graphs.append(record)
        rewire_audits.append(rewire_audit)
        active, component = anomaly_components_from_attention(
            attention,
            scores=true_score, threshold=anomaly_threshold,
            csr_row_block=config.csr_row_block,
        )
        graph_paths[record["sample_id"]] = _save_graph_index(
            graph_directory, record, representation_file,
            canonical_split=test_dataset.root,
            canonical_sha256=test_dataset.rows[str(sample_id)]["sha256"],
            rewire_seed=config.seed,
            true_graph_file=true_graph_file,
            anomaly_score=true_score, anomaly_threshold=anomaly_threshold,
            anomaly_mask=active, anomaly_component=component,
            rewire_audit=rewire_audit,
        )
        offset = end
        sample.release_attention()
    representation_output.flush()
    true_graph_output.flush()
    representation_output._mmap.close()
    true_graph_output._mmap.close()
    if offset != total_tokens:
        raise RuntimeError("test token count and frozen arrays do not align")
    rewire_rr_edges = sum(audit["rr_edges"] for audit in rewire_audits)
    rewire_changed_edges = sum(
        audit["rewired_changed_edges"] for audit in rewire_audits
    )
    rewire_summary = {
        "samples": len(rewire_audits),
        "rr_edges": rewire_rr_edges,
        "rewired_changed_edges": rewire_changed_edges,
        "rewired_changed_fraction": (
            rewire_changed_edges / rewire_rr_edges if rewire_rr_edges else 0.0
        ),
    }
    metadata = _metadata_arrays(metadata)
    selected_samples, selection_rule = _select_samples(config, graphs, coordinate_output)

    reference_model_file = output / "train_reference_model.npz"
    reference_payload = {
        "schema": np.asarray(SCHEMA), "labels_included": np.asarray(False),
        "representation_feature_names": np.asarray(feature_names),
        "direct_field_names": np.asarray(direct_names),
        "propagation_field_names": np.asarray(propagation_names),
        "anomaly_threshold": np.asarray(anomaly_threshold, dtype=np.float32),
    }
    for view, model in references.items():
        _save_view_reference(reference_payload, view, model)
    np.savez(reference_model_file, **reference_payload)

    index_payload = dict(
        schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
        representation_feature_names=np.asarray(feature_names),
        direct_field_names=np.asarray(direct_names),
        propagation_field_names=np.asarray(propagation_names),
        visualization_coordinates=coordinate_output,
        node_representation_file=np.asarray(representation_file.name),
        true_graph_representation_file=np.asarray(true_graph_file.name),
        exact_route_canonical_split=np.asarray(str(test_dataset.root)),
        exact_route_storage=np.asarray("canonical sparse CSR attention"),
        exact_route_channel_layout=np.asarray("channel=layer*num_heads+head"),
        rewire_seed=np.asarray(config.seed, dtype=np.int64),
        anomaly_threshold=np.asarray(anomaly_threshold, dtype=np.float32),
        sample_id=metadata["sample_id"], source_id=metadata["source_id"],
        token_index=metadata["token_index"], token_id=metadata["token_id"],
        relative_position=metadata["relative_position"],
        task_type=metadata["task_type"], data_source=metadata["data_source"],
        generator_model=metadata["generator_model"],
    )
    for name, values in scores.items():
        index_payload[f"{name}_score"] = values
    np.savez(output / "token_representations_label_free.npz", **index_payload)
    label_free_report = {
        "schema": SCHEMA, "labels_read": False,
        "primary_node_state": {
            "name": "direct layer-head Lookback",
            "shape_per_token": [num_layers, num_heads],
            "flattened_dimensions": channels,
            "layer_head_averaged": False,
        },
        "compact_layer_structure": {
            "names": list(names),
            "shape_per_token": [len(names), num_layers],
            "mixed_into_primary_node_vector": False,
            "role": "computed_on_demand_for_explicit_sample_visualization_only",
            "stored_for_full_population": False,
        },
        "compression_semantics": {
            "undefined_lookback_fill": "attention_floor",
            "attention_floor": float(test_dataset.manifest["attention_floor"]),
            "missing_edges_reconstructed": False,
            "interpretation": "all masses and routes use retained CSR edges only",
        },
        "evidence_flow_graph": {
            "trainable": False,
            "backpropagation": False,
            "all_head_mean_used": False,
            "route": "exact retained CSR layer-head channels; no projection",
            "execution": "one streaming CSR pass computes direct, true, and rewired flows",
            "raw_mass_preserved": True,
            "node_signal": f"complete {num_layers}x{num_heads} Lookback",
            "propagation": "[Fp, Fr] exact-channel residual flows",
            "direct": "[prompt_mass, response_mass] exact-channel marginals",
            "direct_dimensions": len(direct_names),
            "propagation_dimensions": len(propagation_names),
            "randomized_null": (
                "preserves every RR target, layer, head, channel, weight, and "
                "floor(log2(lag)) bucket; each causal source changes whenever "
                "that bucket contains an alternative (parallel routes are allowed)"
            ),
            "rewire_audit": rewire_summary,
        },
        "exact_route_audit": {
            "canonical_train_split": str(train_dataset.root),
            "canonical_test_split": str(test_dataset.root),
            "train_manifest_fingerprint": _manifest_fingerprint(train_dataset),
            "test_manifest_fingerprint": _manifest_fingerprint(test_dataset),
            "stored_again_in_output": False,
            "sample_key": "sample_id",
            "channel_layout": "channel=layer*num_heads+head",
            "target_layout": "target=response_idx+response_row",
            "rewire_seed": config.seed,
            "per_sample_rewire_audit": "sample_graphs/*.npz",
        },
        "unsupervised_scores": {
            "views": list(view_names),
            "primary_score": "true_graph",
            "within_view": (
                "train-ECDF of max(position-conditioned robust-tail, "
                "train-only PCA reconstruction residual)"
            ),
            "scalar_only": "unlabeled position-conditioned reference on mean(X); no label-selected direction",
            "anomaly_quantile": config.anomaly_quantile,
            "anomaly_threshold": anomaly_threshold,
        },
        "reference_models": {
            view: model.report() for view, model in references.items()
        },
        "structural_validation_protocol": {
            "frozen_before_labels": True,
            "comparisons": [
                "true_graph minus token_only",
                "true_graph minus rewired_graph",
                "response_graph minus prompt_graph",
            ],
            "primary_success_rule": (
                "true_graph is the fixed primary score; all comparisons are paired sample bootstraps"
            ),
            "uncertainty": (
                f"paired {config.bootstrap_replicates}-replicate bootstrap over "
                "whole response samples"
            ),
        },
        "train_tokens": int(train_tokens), "test_tokens": total_tokens,
        "sample_selection": {
            "sample_ids": selected_samples, "rule": selection_rule, "labels_used": False,
        },
        "artifacts": {
            "node_representations": str(representation_file),
            "true_graph_representations": str(true_graph_file),
            "visualization_coordinates": "PCA coordinates only; not node representations",
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
    evaluation_labels_file = output / "evaluation_token_labels.npy"
    np.save(evaluation_labels_file, labels)
    lookback_signals = _evaluate_lookback(
        representation_file, labels, num_heads,
    )
    score_metrics = {name: _ranking(labels, value) for name, value in scores.items()}
    comparisons = {
        "true_graph_vs_token_only": _cluster_bootstrap_difference(
            labels, scores["true_graph"], scores["token_only"],
            metadata["sample_id"], seed=config.seed,
            replicates=config.bootstrap_replicates,
            description="[6/8] bootstrap true graph vs token-only",
        ),
        "true_graph_vs_rewired_graph": _cluster_bootstrap_difference(
            labels, scores["true_graph"], scores["rewired_graph"],
            metadata["sample_id"], seed=config.seed + 1,
            replicates=config.bootstrap_replicates,
            description="[6/8] bootstrap true graph vs rewired graph",
        ),
        "response_graph_vs_prompt_graph": _cluster_bootstrap_difference(
            labels, scores["response_graph"], scores["prompt_graph"],
            metadata["sample_id"], seed=config.seed + 2,
            replicates=config.bootstrap_replicates,
            description="[6/8] bootstrap response graph vs prompt graph",
        ),
    }
    topology_validation = {
        "observed_metric_gains": {
            "true_graph_minus_token_only_auroc": (
                score_metrics["true_graph"]["auroc"]
                - score_metrics["token_only"]["auroc"]
            ),
            "true_graph_minus_token_only_auprc": (
                score_metrics["true_graph"]["auprc"]
                - score_metrics["token_only"]["auprc"]
            ),
            "true_graph_minus_rewired_graph_auroc": (
                score_metrics["true_graph"]["auroc"]
                - score_metrics["rewired_graph"]["auroc"]
            ),
            "true_graph_minus_rewired_graph_auprc": (
                score_metrics["true_graph"]["auprc"]
                - score_metrics["rewired_graph"]["auprc"]
            ),
        },
        "paired_sample_bootstrap": comparisons,
        "by_task_type": {
            name: _metrics_by_group(labels, values, metadata["task_type"])
            for name, values in scores.items()
        },
        "fixed_threshold_detection": _threshold_metrics(
            labels, scores["true_graph"], anomaly_threshold
        ),
        "anomaly_component_localization": _component_detection_metrics(
            labels, graphs, graph_paths
        ),
    }
    topology_validation["primary_success_rule_satisfied"] = bool(all(
        gain > 0
        for gain in topology_validation["observed_metric_gains"].values()
    ))
    topology_validation["strong_bootstrap_evidence"] = bool(all(
        comparisons[comparison][metric]["ci95"][0] is not None
        and comparisons[comparison][metric]["ci95"][0] > 0
        for comparison in (
            "true_graph_vs_token_only", "true_graph_vs_rewired_graph"
        )
        for metric in ("auroc_difference", "auprc_difference")
    ))
    report = {
        **label_free_report,
        "labels_read": True,
        "labels_read_during": "evaluation_and_plot_coloring_only",
        "labels_used": {"train": False, "test": "evaluation_only"},
        "evaluation_label_cache": str(evaluation_labels_file),
        "unsupervised_score_evaluation": score_metrics,
        "graph_pattern_score_evaluation": {
            view: score_metrics[view]
            for view in ("prompt_graph", "response_graph")
        },
        "raw_representations": {
            "X": {
                "file": str(representation_file), "dtype": "float16",
                "shape": [total_tokens, channels],
                "meaning": "direct Lookback [L,H] flattened without temporal smoothing",
            },
            "Z": {
                "file": str(true_graph_file), "dtype": "float16",
                "shape": [total_tokens, 3 * channels],
                "meaning": "true_graph=[X,Fp,Fr] exact-channel representation",
            },
        },
        "structural_validation": topology_validation,
        "lookback_layer_head_signal_evaluation": lookback_signals,
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

    print("[7/8] rendering the population summary", flush=True)
    population = _render_population(output, coordinate_output, scores, labels)
    report["population_figure"] = str(population)
    report["sample_visualizations"] = []
    report["sample_visualization"] = {
        "automatic": False,
        "reason": "full routes and dense adjacency plots are explicit post-processing",
        "recommended_sample_ids": selected_samples,
        "command": (
            "python main.py render-token-graph --test-split <test> "
            f"--output-dir {output} --sample-id <sample_id>"
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[8/8] complete", flush=True)
    return {
        "output_dir": str(output), "report": str(report_path),
        "test_nodes": total_tokens, "primary_score": "true_graph",
        "score_evaluation": {
            view: {
                "auroc": score_metrics[view]["auroc"],
                "auprc": score_metrics[view]["auprc"],
            }
            for view in view_names
        },
        "structural_comparisons": comparisons,
    }
