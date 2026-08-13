"""Label-free token representations and multiscale causal evidence flow.

The primary node state is the complete layer-head Lookback vector. For a
32-layer, 32-head observer this is exactly 1024 dimensions.  A fixed graph
filter bank sends projected head signals and prompt-bin evidence over the
actual layer-wise response topology, retaining one/two-hop diffusion
innovations.  An incoming-route/weight-matched randomized topology is processed identically
as a structural null control.

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
from .evidence_flow import (
    anomaly_components,
    direct_field_names,
    evidence_flow_fields,
    fixed_head_projection,
    propagation_block_slices,
    propagation_field_names,
)
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
    prompt_bins: int = 16
    graph_head_components: int = 8
    bootstrap_replicates: int = 200
    csr_row_block: int = 4096
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
            self.position_bins, self.lookback_window, self.provenance_hops,
            self.prompt_bins, self.graph_head_components,
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
            rows, source, weight = _csr_entries(attention, row_start, row_end)
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


class _ScalarCalibrator:
    """Calibrate a fixed combination of already calibrated view scores."""

    def fit(self, values):
        self.reference = np.sort(np.asarray(values, dtype=np.float32))
        self.reference_scores = self.transform(values)
        return self

    def transform(self, values):
        return _empirical_rank(self.reference, values)

    def threshold(self, quantile):
        return float(np.quantile(self.reference_scores, float(quantile)))


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
        return self

    def transform(self, values, position):
        standardized = self.scaler.transform(values, position)
        tail, residual, coordinates = _score_representation(
            standardized, self.pca, self.config.tail_fraction
        )
        latent = self.pca.transform(standardized).astype(np.float32)
        return {
            "score": self.calibrator.transform(tail, residual),
            "tail": tail, "subspace_residual": residual,
            "coordinates": coordinates, "latent": latent,
            "standardized": standardized,
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


def _save_graph_index(
    directory, graph, route, representation_file, structure_file, *,
    graph_embedding_file=None, anomaly_score=None, anomaly_threshold=None,
    anomaly_mask=None, anomaly_component=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sample_{_safe_filename(graph['sample_id'])}.npz"
    payload = dict(
        schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
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
    if graph_embedding_file is not None:
        payload["graph_embedding_file"] = np.asarray(graph_embedding_file.name)
    if anomaly_score is not None:
        payload.update(
            evidence_flow_score=np.asarray(anomaly_score, dtype=np.float32),
            anomaly_threshold=np.asarray(anomaly_threshold, dtype=np.float32),
            anomaly_mask=np.asarray(anomaly_mask, dtype=bool),
            anomaly_component=np.asarray(anomaly_component, dtype=np.int32),
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


def _route_matrices(route, response_idx, response_count, layer=None):
    """Dense RP/RR matrices of maximum salient route weight for plotting."""
    token_count = int(response_idx) + int(response_count)
    selected = _collapsed_route_indices(route, token_count, layer=layer)
    source = np.asarray(route["source"])[selected]
    target = np.asarray(route["target"])[selected]
    weight = np.asarray(route["weight"], dtype=np.float32)[selected]
    target_relative = target - int(response_idx)
    rp = np.zeros((response_count, response_idx), dtype=np.float32)
    rr = np.zeros((response_count, response_count), dtype=np.float32)
    prompt = source < int(response_idx)
    if bool(prompt.any()):
        rp[target_relative[prompt], source[prompt]] = weight[prompt]
    history = ~prompt
    if bool(history.any()):
        rr[target_relative[history], source[history] - int(response_idx)] = weight[history]
    return rp, rr, selected


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
        title="Train-only PCA of true-topology diffusion innovations",
        xlabel="component 1", ylabel="component 2",
    )
    axes[0].legend(frameon=False)
    for axis, key, title in (
        (axes[1], "token_only", "Token-only reference"),
        (axes[2], "evidence_flow", "Complete evidence-flow detector"),
        (axes[3], "randomized_topology_control", "Randomized-topology control"),
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
    rp_matrix, rr_matrix, collapsed = _route_matrices(
        route, response_idx, response_count, layer=layer
    )
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
    incoming = rr_matrix.sum(axis=1) + rp_matrix.sum(axis=1)
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

    # Panels 2/3: adjacency matrices expose source, target, weight and distance.
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    rp_norm = _weight_norm(rp_matrix)
    image = axes[1].imshow(
        np.ma.masked_less_equal(rp_matrix, 0), origin="lower", aspect="auto",
        cmap=cmap, norm=rp_norm, interpolation="nearest",
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
    if rp_norm is not None:
        figure.colorbar(image, ax=axes[1], label="salient RP route weight (log scale)")
    if len(provenance_rows):
        axes[1].legend(frameon=False, loc="upper right")

    rr_norm = _weight_norm(rr_matrix)
    image = axes[2].imshow(
        np.ma.masked_less_equal(rr_matrix, 0), origin="lower", aspect="equal",
        cmap=cmap, norm=rr_norm, interpolation="nearest",
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
    if rr_norm is not None:
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
        "rp_route_edges": int(np.count_nonzero(rp_matrix)),
        "rr_route_edges": int(np.count_nonzero(rr_matrix)),
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
    saved_config = saved_report.get("config", {})
    with np.load(index_path, allow_pickle=False) as index:
        names = tuple(map(str, index["structure_names"].tolist()))
        structure_file = output / str(index["compact_layer_structure_file"])
        coordinates_all = np.asarray(index["visualization_coordinates"], dtype=np.float32)
        saved_sample_ids = np.asarray(index["sample_id"]).astype(str)
    with np.load(graph_path, allow_pickle=False) as artifact:
        start = int(artifact["global_row_start"])
        end = int(artifact["global_row_end"])
        graph = {
            "sample_id": str(artifact["sample_id"]),
            "start": start, "end": end,
            "token_ids": artifact["token_ids"],
            "response_idx": int(artifact["response_idx"]),
        }
        route = {
            "layer": artifact["compact_route_layer"],
            "source": artifact["compact_route_source"],
            "target": artifact["compact_route_target"],
            "weight": artifact["compact_route_weight"],
        }
        anomaly_score = (
            artifact["evidence_flow_score"]
            if "evidence_flow_score" in artifact.files
            else np.zeros(end - start, dtype=np.float32)
        )
        anomaly_threshold = (
            float(artifact["anomaly_threshold"])
            if "anomaly_threshold" in artifact.files else 1.0
        )
        anomaly_component = (
            artifact["anomaly_component"]
            if "anomaly_component" in artifact.files
            else np.full(end - start, -1, dtype=np.int32)
        )
    structure_array = np.load(structure_file, mmap_mode="r")
    structure = np.asarray(
        structure_array[:, start:end], dtype=np.float32
    ).transpose(1, 0, 2)
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
    labels = np.asarray(all_labels[start:end], dtype=np.int8)
    if len(labels) != end - start:
        raise ValueError("saved sample rows do not align with evaluation labels")
    config = TokenRepresentationConfig(
        lookback_window=int(saved_config.get("lookback_window", 8)),
        provenance_hops=int(saved_config.get("provenance_hops", 2)),
        prompt_bins=int(saved_config.get("prompt_bins", 16)),
        graph_head_components=int(saved_config.get("graph_head_components", 8)),
        display_mass_cover=float(saved_config.get("display_mass_cover", .80)),
        display_edges_per_type=int(saved_config.get("display_edges_per_type", 2)),
        display_max_edges=int(saved_config.get("display_max_edges", 300)),
        display_layer=layer,
    )
    config.validate()
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
        "features_recomputed": False, "label_source": label_source,
        "predicted_anomaly_nodes": int(
            np.count_nonzero(anomaly_score >= anomaly_threshold)
        ),
    }


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
    head_components = min(config.graph_head_components, num_heads)
    head_projection = fixed_head_projection(
        num_heads, head_components, config.seed, device="cpu"
    )
    direct_names = direct_field_names(num_layers, config.prompt_bins)
    propagation_names = propagation_field_names(
        num_layers, head_components, config.prompt_bins
    )

    print("[1/8] building token, edge, and topology-null train references", flush=True)
    reservoirs = {
        name: _PositionReservoir(
            config.position_bins, config.reference_size, config.seed
        )
        for name in (
            "token_only", "direct_edges", "true_propagation",
            "randomized_propagation",
        )
    }
    train_sources = set()
    train_tokens = 0
    for sample_id in tqdm(
        train_dataset.sample_ids, desc="train evidence-flow nodes", unit="sample"
    ):
        sample = train_dataset[sample_id]
        attention = sample.attention()
        current_lookback = direct_lookback_channels(
            attention, window=1, csr_row_block=config.csr_row_block
        )
        lookback = _window_lookback_channels(current_lookback, config.lookback_window)
        route_tensors = _layer_route_tensors(
            attention, csr_row_block=config.csr_row_block,
        )
        direct, true_propagation, _ = evidence_flow_fields(
            lookback, route_tensors,
            prompt_count=attention.response_idx,
            prompt_bins=config.prompt_bins,
            head_projection=head_projection,
            sample_id=sample_id, seed=config.seed, randomize_rr=False,
        )
        randomized_direct, randomized_propagation, _ = evidence_flow_fields(
            lookback, route_tensors,
            prompt_count=attention.response_idx,
            prompt_bins=config.prompt_bins,
            head_projection=head_projection,
            sample_id=sample_id, seed=config.seed, randomize_rr=True,
        )
        if not torch.equal(direct, randomized_direct):
            raise RuntimeError("RR randomization changed the direct-edge field")
        representation = build_node_representation(
            lookback, num_layers=num_layers, num_heads=num_heads
        )
        reference_views = {
            "token_only": representation,
            "direct_edges": direct,
            "true_propagation": true_propagation,
            "randomized_propagation": randomized_propagation,
        }
        position = np.arange(len(representation), dtype=np.float32) / max(
            len(representation) - 1, 1
        )
        for view, values in reference_views.items():
            reservoirs[view].add(values.detach().cpu().numpy(), position)
        train_sources.add(str(sample.source_id))
        train_tokens += len(representation)
        sample.release_attention()
    print("[2/8] fitting four train-only one-class reference models", flush=True)
    references, reference_bins, true_reference_values = {}, None, None
    for view, reservoir in reservoirs.items():
        values, bins = reservoir.matrix()
        if reference_bins is None:
            reference_bins = bins
        elif not np.array_equal(reference_bins, bins):
            raise RuntimeError("view reservoirs did not retain aligned train tokens")
        references[view] = _ViewReference(view, config).fit(values, bins)
        if view == "true_propagation":
            true_reference_values = values
    true_combination = np.maximum.reduce([
        references["token_only"].reference_scores,
        references["direct_edges"].reference_scores,
        references["true_propagation"].reference_scores,
    ])
    random_combination = np.maximum.reduce([
        references["token_only"].reference_scores,
        references["direct_edges"].reference_scores,
        references["randomized_propagation"].reference_scores,
    ])
    true_calibrator = _ScalarCalibrator().fit(true_combination)
    random_calibrator = _ScalarCalibrator().fit(random_combination)
    anomaly_threshold = true_calibrator.threshold(config.anomaly_quantile)
    block_slices = propagation_block_slices(
        num_layers, head_components, config.prompt_bins
    )
    reference_position = (
        reference_bins.astype(np.float32) + 0.5
    ) / config.position_bins
    true_reference_standardized = references["true_propagation"].scaler.transform(
        true_reference_values, reference_position
    )
    block_calibrators = {}
    for block, columns in block_slices.items():
        block_tail = _robust_tail(
            true_reference_standardized[:, columns], config.tail_fraction
        )
        block_calibrators[block] = _ScalarCalibrator().fit(block_tail)

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
    graph_embedding_file = output / "evidence_flow_node_embeddings.float16.npy"
    structure_gib = total_tokens * len(names) * num_layers * 2 / (1024 ** 3)
    representation_gib = total_tokens * len(feature_names) * 2 / (1024 ** 3)
    graph_embedding_width = int(references["true_propagation"].pca.n_components_)
    graph_embedding_gib = total_tokens * graph_embedding_width * 2 / (1024 ** 3)
    print(
        f"[3/8] test_tokens={total_tokens}; compact_structure≈{structure_gib:.2f} GiB; "
        f"node_file≈{representation_gib:.2f} GiB; "
        f"graph_embedding≈{graph_embedding_gib:.2f} GiB",
        flush=True,
    )
    required = (
        structure_gib + representation_gib + graph_embedding_gib
    ) * (1024 ** 3) * 1.10
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
    graph_embedding_output = np.lib.format.open_memmap(
        graph_embedding_file, mode="w+", dtype=np.float16,
        shape=(total_tokens, graph_embedding_width),
    )
    exact_output = np.empty((total_tokens, len(EXACT_FEATURES)), dtype=np.float32)
    coordinate_output = np.empty((total_tokens, 2), dtype=np.float32)
    scores = {
        name: np.empty(total_tokens, dtype=np.float32)
        for name in (
            "token_only", "direct_edges", "true_propagation",
            "randomized_propagation", "evidence_flow",
            "randomized_topology_control",
            *block_slices,
        )
    }
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
            csr_row_block=config.csr_row_block,
            return_route=True,
        )
        direct, true_propagation, _ = evidence_flow_fields(
            windowed_lookback, route_tensors,
            prompt_count=attention.response_idx,
            prompt_bins=config.prompt_bins,
            head_projection=head_projection,
            sample_id=sample_id, seed=config.seed, randomize_rr=False,
        )
        randomized_direct, randomized_propagation, _ = evidence_flow_fields(
            windowed_lookback, route_tensors,
            prompt_count=attention.response_idx,
            prompt_bins=config.prompt_bins,
            head_projection=head_projection,
            sample_id=sample_id, seed=config.seed, randomize_rr=True,
        )
        if not torch.equal(direct, randomized_direct):
            raise RuntimeError("RR randomization changed the direct-edge field")
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
        representation_saved = representation.astype(np.float16)
        test_views = {
            "token_only": representation_saved,
            "direct_edges": direct.detach().cpu().numpy().astype(np.float16),
            "true_propagation": (
                true_propagation.detach().cpu().numpy().astype(np.float16)
            ),
            "randomized_propagation": (
                randomized_propagation.detach().cpu().numpy().astype(np.float16)
            ),
        }
        transformed = {
            view: references[view].transform(values, position)
            for view, values in test_views.items()
        }
        for view in test_views:
            scores[view][offset:end] = transformed[view]["score"]
        for block, columns in block_slices.items():
            block_tail = _robust_tail(
                transformed["true_propagation"]["standardized"][:, columns],
                config.tail_fraction,
            )
            scores[block][offset:end] = block_calibrators[block].transform(block_tail)
        true_raw = np.maximum.reduce([
            transformed["token_only"]["score"],
            transformed["direct_edges"]["score"],
            transformed["true_propagation"]["score"],
        ])
        random_raw = np.maximum.reduce([
            transformed["token_only"]["score"],
            transformed["direct_edges"]["score"],
            transformed["randomized_propagation"]["score"],
        ])
        evidence_score = true_calibrator.transform(true_raw)
        randomized_score = random_calibrator.transform(random_raw)
        scores["evidence_flow"][offset:end] = evidence_score
        scores["randomized_topology_control"][offset:end] = randomized_score
        coordinates = transformed["true_propagation"]["coordinates"]
        graph = build_attention_graph(attention, GraphBuildConfig(selection="threshold"))
        exact = _exact_features(graph, current_lookback).detach().cpu().numpy()
        structure_output[:, offset:end] = (
            structure.detach().cpu().permute(1, 0, 2).numpy().astype(np.float16)
        )
        representation_output[offset:end] = representation_saved
        graph_embedding_output[offset:end] = transformed[
            "true_propagation"
        ]["latent"].astype(np.float16)
        exact_output[offset:end] = exact
        coordinate_output[offset:end] = coordinates
        _append_metadata(metadata, sample, attention)
        record = _graph_record(graph, offset, end)
        graphs.append(record)
        active, component = anomaly_components(
            route, prompt_count=attention.response_idx,
            scores=evidence_score, threshold=anomaly_threshold,
        )
        graph_paths[record["sample_id"]] = _save_graph_index(
            graph_directory, record, route, representation_file, structure_file,
            graph_embedding_file=graph_embedding_file,
            anomaly_score=evidence_score, anomaly_threshold=anomaly_threshold,
            anomaly_mask=active, anomaly_component=component,
        )
        offset = end
        sample.release_attention()
    structure_output.flush()
    representation_output.flush()
    graph_embedding_output.flush()
    if offset != total_tokens:
        raise RuntimeError("test token count and frozen arrays do not align")
    metadata = _metadata_arrays(metadata)
    selected_samples, selection_rule = _select_samples(config, graphs, coordinate_output)

    reference_model_file = output / "train_reference_model.npz"
    reference_payload = {
        "schema": np.asarray(SCHEMA), "labels_included": np.asarray(False),
        "head_projection": head_projection.detach().cpu().numpy(),
        "representation_feature_names": np.asarray(feature_names),
        "direct_field_names": np.asarray(direct_names),
        "propagation_field_names": np.asarray(propagation_names),
        "evidence_flow_combination_reference": true_calibrator.reference,
        "randomized_combination_reference": random_calibrator.reference,
        "anomaly_threshold": np.asarray(anomaly_threshold, dtype=np.float32),
    }
    for block, calibrator in block_calibrators.items():
        reference_payload[f"{block}_reference"] = calibrator.reference
    for view, model in references.items():
        _save_view_reference(reference_payload, view, model)
    np.savez(reference_model_file, **reference_payload)

    index_payload = dict(
        schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
        structure_names=np.asarray(names),
        representation_feature_names=np.asarray(feature_names),
        direct_field_names=np.asarray(direct_names),
        propagation_field_names=np.asarray(propagation_names),
        exact_feature_names=np.asarray(EXACT_FEATURES),
        exact_token_features=exact_output,
        visualization_coordinates=coordinate_output,
        compact_layer_structure_file=np.asarray(structure_file.name),
        node_representation_file=np.asarray(representation_file.name),
        graph_embedding_file=np.asarray(graph_embedding_file.name),
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
        "evidence_flow_graph": {
            "trainable": False,
            "backpropagation": False,
            "all_head_mean_used": False,
            "route": (
                "union of retained layer/query/source edges; each route weight "
                "is the strongest retained head value without head averaging"
            ),
            "raw_mass_preserved": True,
            "conditional_message": "M1=(W X)/(W 1); M2=(W^2 X)/(W^2 1)",
            "filter_bank": ["(W1)(X-M1)", "(W^2 1)(M1-M2)"],
            "node_signal": (
                f"complete {num_layers}x{num_heads} Lookback, projected within "
                f"each layer to {head_components} fixed orthonormal channels"
            ),
            "prompt_signal": f"{config.prompt_bins} source-position bins per layer",
            "direct_dimensions": len(direct_names),
            "propagation_dimensions": len(propagation_names),
            "randomized_null": (
                "preserves every RR target, layer, route-entry count, and edge "
                "weight; only causal history source endpoints are randomized "
                "(parallel null routes are allowed)"
            ),
        },
        "unsupervised_scores": {
            "primary_views": [
                "token_only", "direct_edges", "true_propagation",
                "randomized_propagation", "evidence_flow",
                "randomized_topology_control",
            ],
            "pattern_diagnostics": list(block_slices),
            "within_view": (
                "train-ECDF of max(position-conditioned robust-tail, "
                "train-only PCA reconstruction residual)"
            ),
            "evidence_flow": (
                "train-calibrated maximum of token-only, direct-edge, and "
                "true-topology propagation views"
            ),
            "anomaly_quantile": config.anomaly_quantile,
            "anomaly_threshold": anomaly_threshold,
        },
        "reference_models": {
            view: model.report() for view, model in references.items()
        },
        "structural_validation_protocol": {
            "frozen_before_labels": True,
            "comparisons": [
                "evidence_flow minus token_only",
                "true_propagation minus randomized_propagation",
                "evidence_flow minus randomized_topology_control",
            ],
            "primary_success_rule": (
                "positive AUROC and AUPRC gains for evidence_flow over token_only, "
                "and true_propagation over its incoming-route/weight-matched randomized null"
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
            "compact_layer_structure": str(structure_file),
            "node_representations": str(representation_file),
            "evidence_flow_node_embeddings": str(graph_embedding_file),
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
    comparisons = {
        "evidence_flow_vs_token_only": _cluster_bootstrap_difference(
            labels, scores["evidence_flow"], scores["token_only"],
            metadata["sample_id"], seed=config.seed,
            replicates=config.bootstrap_replicates,
            description="[6/8] bootstrap evidence vs token",
        ),
        "true_vs_randomized_propagation": _cluster_bootstrap_difference(
            labels, scores["true_propagation"], scores["randomized_propagation"],
            metadata["sample_id"], seed=config.seed + 1,
            replicates=config.bootstrap_replicates,
            description="[6/8] bootstrap true vs randomized propagation",
        ),
        "evidence_flow_vs_randomized_topology": _cluster_bootstrap_difference(
            labels, scores["evidence_flow"], scores["randomized_topology_control"],
            metadata["sample_id"], seed=config.seed + 2,
            replicates=config.bootstrap_replicates,
            description="[6/8] bootstrap evidence vs randomized topology",
        ),
    }
    topology_validation = {
        "observed_metric_gains": {
            "evidence_flow_minus_token_only_auroc": (
                score_metrics["evidence_flow"]["auroc"]
                - score_metrics["token_only"]["auroc"]
            ),
            "evidence_flow_minus_token_only_auprc": (
                score_metrics["evidence_flow"]["auprc"]
                - score_metrics["token_only"]["auprc"]
            ),
            "true_minus_randomized_propagation_auroc": (
                score_metrics["true_propagation"]["auroc"]
                - score_metrics["randomized_propagation"]["auroc"]
            ),
            "true_minus_randomized_propagation_auprc": (
                score_metrics["true_propagation"]["auprc"]
                - score_metrics["randomized_propagation"]["auprc"]
            ),
        },
        "paired_sample_bootstrap": comparisons,
        "by_task_type": {
            name: _metrics_by_group(labels, values, metadata["task_type"])
            for name, values in scores.items()
        },
        "fixed_threshold_detection": _threshold_metrics(
            labels, scores["evidence_flow"], anomaly_threshold
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
            "evidence_flow_vs_token_only", "true_vs_randomized_propagation"
        )
        for metric in ("auroc_difference", "auprc_difference")
    ))
    report = {
        **label_free_report,
        "labels_read": True,
        "labels_read_during": "evaluation_and_plot_coloring_only",
        "evaluation_label_cache": str(evaluation_labels_file),
        "unsupervised_score_evaluation": score_metrics,
        "graph_pattern_score_evaluation": {
            block: score_metrics[block] for block in block_slices
        },
        "structural_validation": topology_validation,
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
            anomaly_score = graph_artifact["evidence_flow_score"]
            sample_threshold = float(graph_artifact["anomaly_threshold"])
            anomaly_component = graph_artifact["anomaly_component"]
        structure = np.asarray(
            structure_read[:, start:end], dtype=np.float32
        ).transpose(1, 0, 2)
        lookback = np.asarray(representation_read[start:end], dtype=np.float32)
        figure, visualization_stats = _render_sample(
            output, graph, route, coordinate_output[start:end], structure,
            labels[start:end], names, config, num_layers,
            layer=config.display_layer,
            anomaly_scores=anomaly_score, anomaly_threshold=sample_threshold,
            anomaly_component=anomaly_component,
        )
        detail_path = output / f"sample_{_safe_filename(sample_id)}_graph_state.npz"
        np.savez_compressed(
            detail_path, schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
            sample_id=np.asarray(sample_id), structure_names=np.asarray(names),
            compact_layer_structure=structure.astype(np.float16),
            node_representation=lookback.astype(np.float16),
            visualization_coordinates=coordinate_output[start:end],
            evidence_flow_score=anomaly_score,
            anomaly_threshold=np.asarray(sample_threshold, dtype=np.float32),
            anomaly_component=anomaly_component,
        )
        sample_rows.append({
            "sample_id": sample_id, "selection_rule": selection_rule,
            "response_nodes": int(end - start),
            "hallucination_tokens": int(labels[start:end].sum()),
            "predicted_anomaly_tokens": int(
                np.count_nonzero(anomaly_score >= sample_threshold)
            ),
            "predicted_anomaly_components": int(
                len(set(anomaly_component[anomaly_component >= 0].tolist()))
            ),
            "attention_structure_figure": str(figure),
            "visualization_stats": visualization_stats,
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
        "evidence_flow_node_embeddings": str(graph_embedding_file),
        "compact_layer_structure": str(structure_file),
        "train_reference_model": str(reference_model_file),
        "sample_graph_directory": str(graph_directory),
        "population_figure": str(population), "sample_visualizations": sample_rows,
        "test_nodes": total_tokens, "unsupervised_score_metrics": score_metrics,
        "structural_validation": topology_validation,
    }
