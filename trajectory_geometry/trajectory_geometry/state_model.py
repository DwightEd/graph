"""Unsupervised attention-conditioned graph state-transition model.

The model is deliberately linear and closed form.  Attention is used as a
directed operator on a shared hidden-state signal; labels, GNN training, and
backpropagation are excluded.  The primary node representation is the
cross-layer trajectory of graph-conditioned prediction residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .data import SparseAttentionSample, load_attention_sample
from .hidden import AttentionHiddenPair, load_hidden_sample


SCHEMA = "attention-conditioned-graph-state-model-v1"
VIEW_NAMES = ("node_control", "true_graph", "rewired_graph")
CONTROL_NAMES = (
    "relative_response_position",
    "retained_prompt_share",
    "length_normalized_offdiagonal_lookback",
    "self_mass",
    "unresolved_mass",
)


@dataclass(frozen=True)
class StateModelConfig:
    projection_dim: int = 16
    projection_reference_rows: int = 12_000
    head_components: int = 8
    fit_tokens_per_layer: int = 4_096
    fit_fraction: float = 0.8
    trim_fraction: float = 0.9
    ridge: float = 1e-2
    residual_shrinkage: float = 0.1
    minimum_relative_graph_gain: float = 0.01
    bootstrap_replicates: int = 1000
    dct_components: int = 8
    prompt_rewire_bins: int = 8
    csr_row_block: int = 4096
    seed: int = 20260815

    def validate(self) -> None:
        if self.projection_dim < 1 or self.projection_reference_rows < self.projection_dim + 2:
            raise ValueError("projection reference must exceed projection dimension")
        if self.head_components < 1 or self.fit_tokens_per_layer < 8:
            raise ValueError("head components and fit token reservoir must be positive")
        if not 0.5 <= self.fit_fraction < 1.0:
            raise ValueError("fit_fraction must be in [0.5,1)")
        if not 0.5 <= self.trim_fraction <= 1.0:
            raise ValueError("trim_fraction must be in [0.5,1]")
        if self.ridge <= 0 or not 0 <= self.residual_shrinkage <= 1:
            raise ValueError("ridge/shrinkage configuration is invalid")
        if not 0 <= self.minimum_relative_graph_gain < 1:
            raise ValueError("minimum_relative_graph_gain must be in [0,1)")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if self.dct_components < 1 or self.prompt_rewire_bins < 1:
            raise ValueError("DCT components and prompt bins must be positive")
        if self.csr_row_block < 1:
            raise ValueError("csr_row_block must be positive")


@dataclass(frozen=True)
class ProjectionModel:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(self.mean.size)

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, states: np.ndarray) -> np.ndarray:
        if states.shape[-1] != self.input_dim:
            raise ValueError("hidden dimension differs from the frozen projection")
        flat = states.reshape(-1, self.input_dim).astype(np.float32, copy=False)
        projected = (flat - self.mean) @ self.components.T
        projected /= self.scale
        return projected.reshape(*states.shape[:-1], self.output_dim).astype(np.float32)


@dataclass(frozen=True)
class ViewModel:
    feature_median: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    residual_center: np.ndarray
    residual_whitener: np.ndarray

    def raw_residual(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        if features.ndim != 3 or features.shape[0] != self.feature_median.shape[0]:
            raise ValueError("view features must have shape [layer, token, feature]")
        standardized = (features - self.feature_median[:, None, :]) / self.feature_scale[
            :, None, :
        ]
        design = np.concatenate(
            [standardized, np.ones((*standardized.shape[:-1], 1), dtype=np.float32)],
            axis=-1,
        )
        prediction = np.einsum("lrd,ldk->lrk", design, self.coefficients)
        return target - prediction

    def whitened_residual(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        residual = self.raw_residual(features, target) - self.residual_center[:, None, :]
        return np.einsum("lrk,lkj->lrj", residual, self.residual_whitener).astype(
            np.float32
        )


@dataclass(frozen=True)
class GraphStateModel:
    config: StateModelConfig
    projection: ProjectionModel
    attention_layers: int
    attention_layer_offset: int
    heads: int
    head_bucket: np.ndarray
    head_sign_scale: np.ndarray
    views: dict[str, ViewModel]
    calibration: dict[str, object]

    @property
    def transitions(self) -> int:
        return int(self.views["node_control"].coefficients.shape[0])


@dataclass(frozen=True)
class StateFeatures:
    base: np.ndarray
    true_message: np.ndarray
    rewired_message: np.ndarray
    target_update: np.ndarray
    controls: np.ndarray


@dataclass(frozen=True)
class StateEncoding:
    embeddings: dict[str, np.ndarray]
    raw_residual_norm: dict[str, np.ndarray]
    graph_gain: np.ndarray
    rewire_gap: np.ndarray
    controls: np.ndarray


def _fit_projection(
    pairs: list[AttentionHiddenPair], config: StateModelConfig
) -> ProjectionModel:
    from sklearn.decomposition import PCA

    per_sample = max(1, int(np.ceil(config.projection_reference_rows / len(pairs))))
    rows = []
    hidden_dim = None
    for index, pair in enumerate(pairs, start=1):
        attention = load_attention_sample(pair.attention_path)
        hidden = load_hidden_sample(pair.hidden_path)
        states, _ = hidden.align(attention)
        if hidden_dim is None:
            hidden_dim = int(states.shape[-1])
        elif hidden_dim != states.shape[-1]:
            raise ValueError("hidden dimension differs across training samples")
        flattened = states.reshape(-1, states.shape[-1])
        take = min(per_sample, flattened.shape[0])
        rng = np.random.default_rng(config.seed + 104729 * index)
        selected = rng.choice(flattened.shape[0], size=take, replace=False)
        rows.append(flattened[selected].astype(np.float32, copy=False))
        print(
            f"[projection {index}/{len(pairs)}] {pair.sample_id}: sampled {take} hidden states",
            flush=True,
        )
    reference = np.concatenate(rows)
    if reference.shape[0] > config.projection_reference_rows:
        rng = np.random.default_rng(config.seed)
        reference = reference[
            rng.choice(reference.shape[0], config.projection_reference_rows, replace=False)
        ]
    if min(reference.shape) <= config.projection_dim:
        raise ValueError("not enough hidden-state reference rows for projection_dim")
    pca = PCA(
        n_components=config.projection_dim,
        svd_solver="randomized",
        random_state=config.seed,
    ).fit(reference)
    scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-8))
    return ProjectionModel(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        scale=scale.astype(np.float32),
    )


def _head_projection(heads: int, components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if components > heads:
        raise ValueError("head_components cannot exceed the number of attention heads")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(heads)
    bucket = np.empty(heads, dtype=np.int16)
    bucket[permutation] = np.arange(heads) % components
    sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=heads)
    counts = np.bincount(bucket, minlength=components)
    sign /= np.sqrt(counts[bucket]).astype(np.float32)
    return bucket, sign.astype(np.float32)


def _mix_hash(*values: np.ndarray | int) -> np.ndarray:
    result = np.asarray(values[0], dtype=np.uint64)
    for value in values[1:]:
        result = result * np.uint64(6364136223846793005) + np.asarray(
            value, dtype=np.uint64
        ) + np.uint64(1442695040888963407)
    result ^= result >> np.uint64(33)
    result *= np.uint64(0xff51afd7ed558ccd)
    result ^= result >> np.uint64(33)
    return result


def causal_rewire_sources(
    source: np.ndarray,
    target: np.ndarray,
    layer: np.ndarray,
    head: np.ndarray,
    *,
    prompt_count: int,
    prompt_bins: int,
    seed: int,
) -> np.ndarray:
    """Rewire sources while preserving RP/RR type and coarse source distance."""
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    result = source.copy()
    hashed = _mix_hash(source, target, layer, head, seed)
    prompt = source < prompt_count
    if np.any(prompt):
        original = source[prompt]
        bins = np.minimum(original * prompt_bins // prompt_count, prompt_bins - 1)
        start = bins * prompt_count // prompt_bins
        stop = (bins + 1) * prompt_count // prompt_bins
        width = stop - start
        movable = width > 1
        shift = np.zeros_like(width)
        shift[movable] = 1 + (
            hashed[prompt][movable] % (width[movable] - 1).astype(np.uint64)
        ).astype(np.int64)
        result[prompt] = start + np.remainder(original - start + shift, width)

    history = ~prompt
    if np.any(history):
        history_source = source[history]
        history_target = target[history]
        lag = history_target - history_source
        edges = np.asarray([1, 2, 4, 8, 16, 32], dtype=np.int64)
        bucket = np.searchsorted(edges, lag, side="left")
        low = np.where(bucket == 0, 1, edges[np.maximum(bucket - 1, 0)] + 1)
        high = np.where(bucket < edges.size, edges[np.minimum(bucket, edges.size - 1)], lag)
        high = np.minimum(high, history_target - prompt_count)
        low = np.minimum(low, high)
        width = high - low + 1
        movable = width > 1
        shift = np.zeros_like(width)
        shift[movable] = 1 + (
            hashed[history][movable] % (width[movable] - 1).astype(np.uint64)
        ).astype(np.int64)
        new_lag = low + np.remainder(lag - low + shift, width)
        result[history] = history_target - new_lag
    if np.any(result < 0) or np.any(result >= target):
        raise AssertionError("causal rewiring produced an invalid source")
    if np.any((source < prompt_count) != (result < prompt_count)):
        raise AssertionError("causal rewiring changed RP/RR relation type")
    return result


def build_state_features(
    attention: SparseAttentionSample,
    projected_states: np.ndarray,
    *,
    attention_layer_offset: int,
    head_bucket: np.ndarray,
    head_sign_scale: np.ndarray,
    config: StateModelConfig,
) -> StateFeatures:
    transitions = projected_states.shape[0] - 1
    response = attention.response_tokens
    dimension = projected_states.shape[-1]
    components = int(head_bucket.max()) + 1
    if transitions + attention_layer_offset > attention.layers:
        raise ValueError("hidden transitions exceed available attention layers")
    if head_bucket.shape != (attention.heads,) or head_sign_scale.shape != (attention.heads,):
        raise ValueError("frozen head projection does not match attention heads")

    true_message = np.zeros(
        (transitions, response, components, dimension), dtype=np.float32
    )
    rewired_message = np.zeros_like(true_message)
    prompt_mass = np.zeros((transitions, response), dtype=np.float32)
    history_mass = np.zeros_like(prompt_mass)
    known_mass = np.zeros((transitions, attention.heads, response), dtype=np.float32)

    true_flat = true_message.reshape(-1, dimension)
    rewired_flat = rewired_message.reshape(-1, dimension)
    for block in attention.iter_sparse_row_blocks(config.csr_row_block):
        valid = (block.layer >= attention_layer_offset) & (
            block.layer < attention_layer_offset + transitions
        )
        if not np.any(valid):
            continue
        layer = block.layer[valid] - attention_layer_offset
        head = block.head[valid]
        query = block.query[valid]
        target = block.target[valid]
        source = block.source[valid]
        weight = block.weight[valid].astype(np.float32, copy=False)
        is_prompt = source < attention.response_idx
        np.add.at(prompt_mass, (layer[is_prompt], query[is_prompt]), weight[is_prompt])
        np.add.at(history_mass, (layer[~is_prompt], query[~is_prompt]), weight[~is_prompt])
        np.add.at(known_mass, (layer, head, query), weight)

        component = head_bucket[head]
        row = (layer * response + query) * components + component
        scale = weight * head_sign_scale[head]
        contribution = projected_states[layer, source] * scale[:, None]
        np.add.at(true_flat, row, contribution)
        rewired_source = causal_rewire_sources(
            source,
            target,
            block.layer[valid],
            head,
            prompt_count=attention.response_idx,
            prompt_bins=config.prompt_rewire_bins,
            seed=config.seed,
        )
        rewired_contribution = projected_states[layer, rewired_source] * scale[:, None]
        np.add.at(rewired_flat, row, rewired_contribution)

    response_slice = slice(attention.response_idx, attention.token_count)
    target_state = projected_states[:-1, response_slice]
    for transition in range(transitions):
        attention_layer = transition + attention_layer_offset
        for head in range(attention.heads):
            diagonal = attention.diagonal[attention_layer, head, response_slice].astype(
                np.float32, copy=False
            )
            known_mass[transition, head] += diagonal
            component = int(head_bucket[head])
            contribution = (
                target_state[transition]
                * diagonal[:, None]
                * float(head_sign_scale[head])
            )
            true_message[transition, :, component] += contribution
            rewired_message[transition, :, component] += contribution

    overflow = np.maximum(known_mass - 1.0, 0.0)
    if np.any(overflow > 0.02):
        raise ValueError(f"attention row mass exceeds one by {float(overflow.max()):.6f}")
    prompt_mass /= attention.heads
    history_mass /= attention.heads
    self_mass = np.stack(
        [
            attention.diagonal[
                transition + attention_layer_offset, :, response_slice
            ].mean(axis=0)
            for transition in range(transitions)
        ]
    ).astype(np.float32)
    unresolved = np.maximum(1.0 - known_mass, 0.0).mean(axis=1).astype(np.float32)
    retained = prompt_mass + history_mass + self_mass
    prompt_share = np.divide(
        prompt_mass,
        retained,
        out=np.zeros_like(prompt_mass),
        where=retained > 0,
    )
    token_index = np.arange(response, dtype=np.float32)
    relative_position = token_index / max(response - 1, 1)
    prompt_average = prompt_mass / float(attention.response_idx)
    history_average = np.divide(
        history_mass,
        token_index[None, :],
        out=np.zeros_like(history_mass),
        where=token_index[None, :] > 0,
    )
    lookback = np.divide(
        prompt_average,
        prompt_average + history_average,
        out=np.zeros_like(prompt_average),
        where=(prompt_average + history_average) > 0,
    )
    controls = np.stack(
        [
            np.broadcast_to(relative_position, prompt_mass.shape),
            prompt_share,
            lookback,
            self_mass,
            unresolved,
        ],
        axis=-1,
    ).astype(np.float32)
    base = np.concatenate([target_state, controls], axis=-1).astype(np.float32)
    return StateFeatures(
        base=base,
        true_message=true_message.reshape(transitions, response, -1),
        rewired_message=rewired_message.reshape(transitions, response, -1),
        target_update=(projected_states[1:, response_slice] - target_state).astype(np.float32),
        controls=controls,
    )


class _LayerReservoir:
    def __init__(self, layers: int, maximum: int, seed: int):
        self.layers = layers
        self.maximum = maximum
        self.rng = np.random.default_rng(seed)
        self.priority: list[np.ndarray | None] = [None] * layers
        self.rows: list[np.ndarray | None] = [None] * layers

    def update(self, layer: int, values: np.ndarray) -> None:
        priority = self.rng.random(values.shape[0])
        if self.rows[layer] is not None:
            values = np.concatenate([self.rows[layer], values])
            priority = np.concatenate([self.priority[layer], priority])
        if values.shape[0] > self.maximum:
            selected = np.argpartition(priority, self.maximum - 1)[: self.maximum]
            values = values[selected]
            priority = priority[selected]
        self.rows[layer] = values.astype(np.float32, copy=False)
        self.priority[layer] = priority


def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - median), axis=0)
    standard = values.std(axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(standard > 1e-6, standard, 1.0))
    return median.astype(np.float32), scale.astype(np.float32)


def _ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    design = np.concatenate(
        [design, np.ones((design.shape[0], 1), dtype=np.float64)], axis=1
    )
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    return np.linalg.solve(gram + penalty, design.T @ target).astype(np.float32)


def _fit_view_layer(
    train_x: np.ndarray,
    train_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    config: StateModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    median, scale = _robust_scale(train_x)
    standardized = (train_x - median) / scale
    coefficients = _ridge(standardized.astype(np.float64), train_y.astype(np.float64), config.ridge)
    prediction = np.concatenate(
        [standardized, np.ones((standardized.shape[0], 1), dtype=np.float32)], axis=1
    ) @ coefficients
    residual_norm = np.linalg.norm(train_y - prediction, axis=1)
    cutoff = np.quantile(residual_norm, config.trim_fraction)
    keep = residual_norm <= cutoff
    if int(keep.sum()) >= max(train_y.shape[1] + 2, 8):
        coefficients = _ridge(
            standardized[keep].astype(np.float64),
            train_y[keep].astype(np.float64),
            config.ridge,
        )

    calibration_standardized = (calibration_x - median) / scale
    calibration_design = np.concatenate(
        [
            calibration_standardized,
            np.ones((calibration_standardized.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    residual = calibration_y - calibration_design @ coefficients
    center = np.median(residual, axis=0)
    centered = residual - center
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    average_variance = max(float(np.trace(covariance) / covariance.shape[0]), 1e-6)
    covariance = (
        (1.0 - config.residual_shrinkage) * covariance
        + config.residual_shrinkage * average_variance * np.eye(covariance.shape[0])
    )
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    whitener = (eigenvector * (1.0 / np.sqrt(np.maximum(eigenvalue, 1e-6)))) @ eigenvector.T
    mse = float(np.mean(np.square(residual)))
    return median, scale, coefficients, center.astype(np.float32), whitener.astype(np.float32), mse


def _paired_bootstrap_interval(
    differences: np.ndarray, replicates: int, seed: int
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or differences.size < 1:
        raise ValueError("paired bootstrap requires sample-level differences")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, differences.size, size=(replicates, differences.size))
    means = differences[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def fit_graph_state_model(
    pairs: list[AttentionHiddenPair], config: StateModelConfig | None = None
) -> GraphStateModel:
    config = config or StateModelConfig()
    config.validate()
    if len(pairs) < 2:
        raise ValueError("at least two training samples are required for sample-held-out calibration")
    projection = _fit_projection(pairs, config)
    first_attention = load_attention_sample(pairs[0].attention_path)
    first_hidden = load_hidden_sample(pairs[0].hidden_path)
    first_states, attention_offset = first_hidden.align(first_attention)
    transitions = first_states.shape[0] - 1
    head_bucket, head_sign_scale = _head_projection(
        first_attention.heads, config.head_components, config.seed
    )
    base_dim = projection.output_dim + len(CONTROL_NAMES)
    message_dim = config.head_components * projection.output_dim
    row_dim = base_dim + message_dim + message_dim + projection.output_dim
    fit_reservoir = _LayerReservoir(
        transitions, config.fit_tokens_per_layer, config.seed + 1
    )
    calibration_maximum = max(
        512,
        int(
            np.ceil(
                config.fit_tokens_per_layer
                * (1.0 - config.fit_fraction)
                / config.fit_fraction
            )
        ),
    )
    calibration_reservoir = _LayerReservoir(
        transitions, calibration_maximum, config.seed + 2
    )
    split_rng = np.random.default_rng(config.seed + 3)
    order = split_rng.permutation(len(pairs))
    fit_count = min(max(int(len(pairs) * config.fit_fraction), 1), len(pairs) - 1)
    fit_indices = set(int(value) for value in order[:fit_count])
    for index, pair in enumerate(pairs, start=1):
        attention = load_attention_sample(pair.attention_path)
        hidden = load_hidden_sample(pair.hidden_path)
        states, offset = hidden.align(attention)
        if (
            attention.layers != first_attention.layers
            or attention.heads != first_attention.heads
            or states.shape[0] - 1 != transitions
            or offset != attention_offset
        ):
            raise ValueError("state/attention geometry differs across training samples")
        projected = projection.transform(states)
        features = build_state_features(
            attention,
            projected,
            attention_layer_offset=attention_offset,
            head_bucket=head_bucket,
            head_sign_scale=head_sign_scale,
            config=config,
        )
        destination = fit_reservoir if index - 1 in fit_indices else calibration_reservoir
        for layer in range(transitions):
            destination.update(
                layer,
                np.concatenate(
                    [
                        features.base[layer],
                        features.true_message[layer],
                        features.rewired_message[layer],
                        features.target_update[layer],
                    ],
                    axis=1,
                ),
            )
        print(
            f"[state-fit {index}/{len(pairs)}] {pair.sample_id}: "
            f"{attention.response_tokens} response tokens -> "
            f"{'fit' if index - 1 in fit_indices else 'calibration'}",
            flush=True,
        )

    view_parts: dict[str, dict[str, list[np.ndarray]]] = {
        name: {key: [] for key in ("median", "scale", "coefficients", "center", "whitener")}
        for name in VIEW_NAMES
    }
    mse_by_view = {name: [] for name in VIEW_NAMES}
    for layer, fit_rows in enumerate(fit_reservoir.rows):
        calibration_rows = calibration_reservoir.rows[layer]
        if fit_rows is None or fit_rows.shape[0] < 8:
            raise ValueError(f"layer {layer} has too few fit tokens")
        if calibration_rows is None or calibration_rows.shape[0] < 2:
            raise ValueError(f"layer {layer} has too few sample-held-out calibration tokens")

        def unpack(rows: np.ndarray):
            base = rows[:, :base_dim]
            true = rows[:, base_dim : base_dim + message_dim]
            rewired = rows[:, base_dim + message_dim : base_dim + 2 * message_dim]
            target = rows[:, -projection.output_dim :]
            return base, true, rewired, target

        fit_base, fit_true, fit_rewired, fit_target = unpack(fit_rows)
        cal_base, cal_true, cal_rewired, cal_target = unpack(calibration_rows)
        fit_matrices = {
            "node_control": fit_base,
            "true_graph": np.concatenate([fit_base, fit_true], axis=1),
            "rewired_graph": np.concatenate([fit_base, fit_rewired], axis=1),
        }
        calibration_matrices = {
            "node_control": cal_base,
            "true_graph": np.concatenate([cal_base, cal_true], axis=1),
            "rewired_graph": np.concatenate([cal_base, cal_rewired], axis=1),
        }
        for name, matrix in fit_matrices.items():
            fitted = _fit_view_layer(
                matrix,
                fit_target,
                calibration_matrices[name],
                cal_target,
                config,
            )
            for key, value in zip(
                ("median", "scale", "coefficients", "center", "whitener"), fitted[:-1]
            ):
                view_parts[name][key].append(value)
            mse_by_view[name].append(fitted[-1])
        print(
            f"[ridge {layer + 1}/{transitions}] calibration mse: "
            + ", ".join(f"{name}={mse_by_view[name][-1]:.6f}" for name in VIEW_NAMES),
            flush=True,
        )

    views = {
        name: ViewModel(
            feature_median=np.stack(parts["median"]),
            feature_scale=np.stack(parts["scale"]),
            coefficients=np.stack(parts["coefficients"]),
            residual_center=np.stack(parts["center"]),
            residual_whitener=np.stack(parts["whitener"]),
        )
        for name, parts in view_parts.items()
    }
    reservoir_mean_mse = {name: float(np.mean(values)) for name, values in mse_by_view.items()}
    provisional = GraphStateModel(
        config=config,
        projection=projection,
        attention_layers=first_attention.layers,
        attention_layer_offset=attention_offset,
        heads=first_attention.heads,
        head_bucket=head_bucket,
        head_sign_scale=head_sign_scale,
        views=views,
        calibration={},
    )
    sample_mse = {name: [] for name in VIEW_NAMES}
    calibration_indices = [int(value) for value in order[fit_count:]]
    for index, pair_index in enumerate(calibration_indices, start=1):
        pair = pairs[pair_index]
        attention = load_attention_sample(pair.attention_path)
        hidden = load_hidden_sample(pair.hidden_path)
        states, offset = hidden.align(attention)
        encoding = encode_graph_state(attention, states, offset, provisional)
        for name in VIEW_NAMES:
            sample_mse[name].append(float(encoding.raw_residual_norm[name].mean()))
        print(
            f"[graph-gate {index}/{len(calibration_indices)}] {pair.sample_id}: "
            + ", ".join(f"{name}={sample_mse[name][-1]:.6f}" for name in VIEW_NAMES),
            flush=True,
        )
    sample_mse_array = {
        name: np.asarray(values, dtype=np.float64) for name, values in sample_mse.items()
    }
    mean_mse = {name: float(values.mean()) for name, values in sample_mse_array.items()}
    node_gain = sample_mse_array["node_control"] - sample_mse_array["true_graph"]
    rewire_gain = sample_mse_array["rewired_graph"] - sample_mse_array["true_graph"]
    node_interval = _paired_bootstrap_interval(
        node_gain, config.bootstrap_replicates, config.seed + 31
    )
    rewire_interval = _paired_bootstrap_interval(
        rewire_gain, config.bootstrap_replicates, config.seed + 37
    )
    node_denominator = max(mean_mse["node_control"], 1e-12)
    rewired_denominator = max(mean_mse["rewired_graph"], 1e-12)
    relative_over_node = (
        mean_mse["node_control"] - mean_mse["true_graph"]
    ) / node_denominator
    relative_over_rewired = (
        mean_mse["rewired_graph"] - mean_mse["true_graph"]
    ) / rewired_denominator
    calibration: dict[str, object] = {
        "prediction_mse": mean_mse,
        "reservoir_token_prediction_mse": reservoir_mean_mse,
        "graph_gain_over_node_control": mean_mse["node_control"] - mean_mse["true_graph"],
        "true_graph_gain_over_rewired": mean_mse["rewired_graph"] - mean_mse["true_graph"],
        "relative_graph_gain_over_node_control": relative_over_node,
        "relative_graph_gain_over_rewired": relative_over_rewired,
        "minimum_relative_graph_gain": config.minimum_relative_graph_gain,
        "paired_graph_gain_ci95_over_node_control": list(node_interval),
        "paired_graph_gain_ci95_over_rewired": list(rewire_interval),
        "gate_passed": bool(
            relative_over_node >= config.minimum_relative_graph_gain
            and relative_over_rewired >= config.minimum_relative_graph_gain
            and node_interval[0] > 0
            and rewire_interval[0] > 0
        ),
        "fit_samples": fit_count,
        "calibration_samples": len(pairs) - fit_count,
        "split_unit": "sample",
        "labels_read": False,
    }
    return GraphStateModel(
        config=config,
        projection=projection,
        attention_layers=first_attention.layers,
        attention_layer_offset=attention_offset,
        heads=first_attention.heads,
        head_bucket=head_bucket,
        head_sign_scale=head_sign_scale,
        views=views,
        calibration=calibration,
    )


def _dct_basis(length: int, components: int) -> np.ndarray:
    components = min(length, components)
    position = np.arange(length, dtype=np.float64) + 0.5
    frequency = np.arange(components, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * frequency * position[None, :] / length)
    basis[0] *= np.sqrt(1.0 / length)
    if components > 1:
        basis[1:] *= np.sqrt(2.0 / length)
    return basis.astype(np.float32)


def encode_graph_state(
    attention: SparseAttentionSample,
    hidden_states: np.ndarray,
    attention_layer_offset: int,
    model: GraphStateModel,
) -> StateEncoding:
    if attention.layers != model.attention_layers or attention.heads != model.heads:
        raise ValueError("attention geometry differs from the frozen model")
    if attention_layer_offset != model.attention_layer_offset:
        raise ValueError("hidden-state layer convention differs from the frozen model")
    projected = model.projection.transform(hidden_states)
    features = build_state_features(
        attention,
        projected,
        attention_layer_offset=attention_layer_offset,
        head_bucket=model.head_bucket,
        head_sign_scale=model.head_sign_scale,
        config=model.config,
    )
    matrices = {
        "node_control": features.base,
        "true_graph": np.concatenate([features.base, features.true_message], axis=-1),
        "rewired_graph": np.concatenate(
            [features.base, features.rewired_message], axis=-1
        ),
    }
    raw = {
        name: model.views[name].raw_residual(matrix, features.target_update)
        for name, matrix in matrices.items()
    }
    whitened = {
        name: model.views[name].whitened_residual(matrix, features.target_update)
        for name, matrix in matrices.items()
    }
    basis = _dct_basis(model.transitions, model.config.dct_components)
    embeddings = {
        name: np.einsum("dl,lrk->rdk", basis, residual).reshape(
            attention.response_tokens, -1
        ).astype(np.float32)
        for name, residual in whitened.items()
    }
    norm = {
        name: np.mean(np.square(value), axis=-1).T.astype(np.float32)
        for name, value in raw.items()
    }
    graph_gain = (norm["node_control"] - norm["true_graph"]).mean(axis=1)
    rewire_gap = (norm["rewired_graph"] - norm["true_graph"]).mean(axis=1)
    return StateEncoding(
        embeddings=embeddings,
        raw_residual_norm=norm,
        graph_gain=graph_gain.astype(np.float32),
        rewire_gap=rewire_gap.astype(np.float32),
        controls=features.controls.transpose(1, 0, 2).astype(np.float32),
    )


def save_graph_state_model(model: GraphStateModel, path: str | Path) -> None:
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA),
        "config_json": np.asarray(json.dumps(model.config.__dict__, sort_keys=True)),
        "calibration_json": np.asarray(json.dumps(model.calibration, sort_keys=True)),
        "projection_mean": model.projection.mean,
        "projection_components": model.projection.components,
        "projection_scale": model.projection.scale,
        "attention_layers": np.asarray(model.attention_layers, dtype=np.int16),
        "attention_layer_offset": np.asarray(model.attention_layer_offset, dtype=np.int8),
        "heads": np.asarray(model.heads, dtype=np.int16),
        "head_bucket": model.head_bucket,
        "head_sign_scale": model.head_sign_scale,
        "control_names": np.asarray(CONTROL_NAMES),
    }
    for name, view in model.views.items():
        payload[f"{name}_feature_median"] = view.feature_median
        payload[f"{name}_feature_scale"] = view.feature_scale
        payload[f"{name}_coefficients"] = view.coefficients
        payload[f"{name}_residual_center"] = view.residual_center
        payload[f"{name}_residual_whitener"] = view.residual_whitener
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_graph_state_model(path: str | Path) -> GraphStateModel:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["schema"].item()) != SCHEMA:
            raise ValueError("unsupported graph state model schema")
        config = StateModelConfig(**json.loads(str(payload["config_json"].item())))
        views = {
            name: ViewModel(
                feature_median=payload[f"{name}_feature_median"].copy(),
                feature_scale=payload[f"{name}_feature_scale"].copy(),
                coefficients=payload[f"{name}_coefficients"].copy(),
                residual_center=payload[f"{name}_residual_center"].copy(),
                residual_whitener=payload[f"{name}_residual_whitener"].copy(),
            )
            for name in VIEW_NAMES
        }
        return GraphStateModel(
            config=config,
            projection=ProjectionModel(
                payload["projection_mean"].copy(),
                payload["projection_components"].copy(),
                payload["projection_scale"].copy(),
            ),
            attention_layers=int(payload["attention_layers"]),
            attention_layer_offset=int(payload["attention_layer_offset"]),
            heads=int(payload["heads"]),
            head_bucket=payload["head_bucket"].copy(),
            head_sign_scale=payload["head_sign_scale"].copy(),
            views=views,
            calibration=json.loads(str(payload["calibration_json"].item())),
        )
