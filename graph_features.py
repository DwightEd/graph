"""Graph-derived token feature engineering.

This module owns all deterministic graph -> vector transformations.  It does
not load datasets, read labels, choose controls, or draw figures.
"""

from __future__ import annotations

import numpy as np
import torch


BASIC_FEATURE_NAMES = (
    "incoming_mass",
    "prompt_mass_share",
    "normalized_entropy",
    "history_lag",
    "in_degree",
    "prompt_degree",
    "history_degree",
    "in_density",
    "prompt_density",
    "history_density",
    "history_edge_share",
    "channel_edge_density",
)

RESPONSE_FEATURE_NAMES = BASIC_FEATURE_NAMES + (
    "prompt_mass",
    "history_mass",
    "prompt_entropy",
    "history_entropy",
    "top1_share",
    "top3_share",
    "hhi",
    "prompt_top1_share",
    "history_top1_share",
    "history_lag_std",
    "history_near1_share",
    "history_near4_share",
    "history_near8_share",
    "history_far16_share",
    "early_prompt_mass",
    "middle_prompt_mass",
    "late_prompt_mass",
    "early_history_mass",
    "middle_history_mass",
    "late_history_mass",
)

STATIC_BLOCK_NAMES = (
    "grounding",
    "self_history",
    "sparsity",
    "concentration",
    "locality",
    "layer_routing",
)

STATIC_BLOCK_FEATURES = {
    "grounding": (
        "prompt_mass_share",
        "prompt_mass",
        "prompt_degree",
        "prompt_density",
        "prompt_entropy",
        "prompt_top1_share",
    ),
    "self_history": (
        "history_mass",
        "history_degree",
        "history_density",
        "history_edge_share",
        "history_entropy",
        "history_top1_share",
    ),
    "sparsity": (
        "incoming_mass",
        "in_degree",
        "in_density",
        "channel_edge_density",
    ),
    "concentration": (
        "normalized_entropy",
        "top1_share",
        "top3_share",
        "hhi",
    ),
    "locality": (
        "history_lag",
        "history_lag_std",
        "history_bin_1",
        "history_bin_2_4",
        "history_bin_5_8",
        "history_bin_9_16",
        "history_bin_gt16",
    ),
    "layer_routing": (
        "early_prompt_mass",
        "middle_prompt_mass",
        "late_prompt_mass",
        "early_history_mass",
        "middle_history_mass",
        "late_history_mass",
    ),
}

DYNAMIC_FEATURE_NAMES = (
    "delta_total",
    "delta_grounding",
    "delta_self_history",
    "delta_sparsity",
    "delta_concentration",
    "delta_locality",
    "delta_layer_routing",
    "rolling_total",
    "rolling_grounding",
    "rolling_self_history",
    "rolling_sparsity",
    "rolling_concentration",
    "rolling_locality",
    "rolling_layer_routing",
    "source_js",
    "prompt_source_js",
    "history_source_js",
    "neighbor_change",
    "layer_routing_shift",
)


def _normalized_entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[weights > 0]
    if len(weights) <= 1:
        return 0.0
    probabilities = weights / weights.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(probabilities)))


def _top_share(weights: np.ndarray, k: int) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0 or len(weights) == 0:
        return 0.0
    k = min(int(k), len(weights))
    return float(np.partition(weights, len(weights) - k)[-k:].sum() / total)


def _hhi(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return 0.0
    probabilities = weights / total
    return float(np.square(probabilities).sum())


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0 or len(values) == 0:
        return 0.0, 0.0
    mean = float(np.sum(values * weights) / total)
    variance = float(np.sum(weights * np.square(values - mean)) / total)
    return mean, float(np.sqrt(max(variance, 0.0)))


def _layer_groups(num_layers: int) -> tuple[np.ndarray, np.ndarray]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    group = np.minimum(2, np.arange(num_layers, dtype=np.int64) * 3 // num_layers)
    return group, np.bincount(group, minlength=3)


def basic_structural_features(attention, relations) -> torch.Tensor:
    """Return the original 12-D response-token structural descriptor."""
    response_idx = attention.response_idx
    response_count = attention.num_response_tokens
    device = relations["weight"].device
    if relations["weight"].numel() == 0:
        return torch.zeros(
            (response_count, len(BASIC_FEATURE_NAMES)), dtype=torch.float32, device=device
        )

    source = relations["source"].long()
    target = relations["target"].long()
    weight = relations["weight"].float()
    channel_count = relations["channel_count"].float()
    rows = target - response_idx
    if bool(((rows < 0) | (rows >= response_count)).any()):
        raise ValueError("relation targets must be response tokens")

    prompt = source < response_idx
    history = ~prompt
    total_mass = torch.zeros(response_count, dtype=torch.float32, device=device)
    total_mass.index_add_(0, rows, weight)
    prompt_mass = torch.zeros_like(total_mass)
    prompt_mass.index_add_(0, rows[prompt], weight[prompt])
    history_mass = torch.zeros_like(total_mass)
    history_mass.index_add_(0, rows[history], weight[history])

    prompt_share = torch.zeros_like(total_mass)
    nonempty = total_mass > 0
    prompt_share[nonempty] = prompt_mass[nonempty] / total_mass[nonempty]

    in_degree = torch.bincount(rows, minlength=response_count).float()
    prompt_degree = torch.bincount(rows[prompt], minlength=response_count).float()
    history_degree = torch.bincount(rows[history], minlength=response_count).float()

    probabilities = weight / total_mass[rows]
    entropy = torch.zeros_like(total_mass)
    entropy.index_add_(0, rows, -probabilities * probabilities.log())
    normalized_entropy = torch.zeros_like(total_mass)
    multiple = in_degree > 1
    normalized_entropy[multiple] = entropy[multiple] / in_degree[multiple].log()

    history_lag_mass = torch.zeros_like(total_mass)
    if bool(history.any()):
        lag = (target[history] - source[history]).float()
        history_lag_mass.index_add_(
            0,
            rows[history],
            weight[history] * lag / max(response_count - 1, 1),
        )
    history_lag = torch.zeros_like(total_mass)
    has_history_mass = history_mass > 0
    history_lag[has_history_mass] = (
        history_lag_mass[has_history_mass] / history_mass[has_history_mass]
    )

    response_position = torch.arange(response_count, dtype=torch.float32, device=device)
    absolute_target = response_idx + response_position
    in_density = in_degree / absolute_target.clamp_min(1.0)
    prompt_density = prompt_degree / float(max(response_idx, 1))
    history_density = torch.zeros_like(total_mass)
    has_history = response_position > 0
    history_density[has_history] = history_degree[has_history] / response_position[has_history]
    history_edge_share = torch.zeros_like(total_mass)
    has_edges = in_degree > 0
    history_edge_share[has_edges] = history_degree[has_edges] / in_degree[has_edges]

    channel_degree = torch.zeros_like(total_mass)
    channel_degree.index_add_(0, rows, channel_count)
    channel_edge_density = channel_degree / (
        float(attention.num_channels) * absolute_target.clamp_min(1.0)
    )

    features = torch.stack(
        (
            total_mass,
            prompt_share,
            normalized_entropy,
            history_lag,
            in_degree,
            prompt_degree,
            history_degree,
            in_density,
            prompt_density,
            history_density,
            history_edge_share,
            channel_edge_density,
        ),
        dim=1,
    )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("structural graph features must be finite")
    return features


def response_graph_features(sample) -> np.ndarray:
    """Return one causal 32-D graph-derived vector per response token."""
    attention = sample.attention()
    relations = sample.relation_edges()
    raw_edges = sample.attention_edges()
    response_idx = int(attention.response_idx)
    response_count = int(attention.num_response_tokens)

    rel_source = relations["source"].detach().cpu().numpy().astype(np.int64, copy=False)
    rel_target = relations["target"].detach().cpu().numpy().astype(np.int64, copy=False)
    rel_weight = relations["weight"].detach().cpu().numpy().astype(np.float64, copy=False)
    rel_channels = relations["channel_count"].detach().cpu().numpy().astype(np.float64, copy=False)
    raw_source = raw_edges["source"].detach().cpu().numpy().astype(np.int64, copy=False)
    raw_target = raw_edges["target"].detach().cpu().numpy().astype(np.int64, copy=False)
    raw_layer = raw_edges["layer"].detach().cpu().numpy().astype(np.int64, copy=False)
    raw_weight = raw_edges["weight"].detach().cpu().numpy().astype(np.float64, copy=False)

    layer_group, layer_counts = _layer_groups(int(attention.num_layers))
    group_channels = layer_counts * int(attention.num_heads)
    matrix = np.zeros((response_count, len(RESPONSE_FEATURE_NAMES)), dtype=np.float32)
    lag_norm = float(max(response_count - 1, 1))

    for row in range(response_count):
        target_abs = response_idx + row
        mask = rel_target == target_abs
        sources = rel_source[mask]
        weights = rel_weight[mask]
        channels = rel_channels[mask]
        prompt = sources < response_idx
        history = ~prompt
        prompt_weights = weights[prompt]
        history_weights = weights[history]

        total_mass = float(weights.sum())
        prompt_mass = float(prompt_weights.sum())
        history_mass = float(history_weights.sum())
        in_degree = float(len(weights))
        prompt_degree = float(prompt.sum())
        history_degree = float(history.sum())
        absolute_target = float(max(target_abs, 1))
        prompt_share = prompt_mass / total_mass if total_mass > 0 else 0.0
        history_share = history_degree / in_degree if in_degree > 0 else 0.0

        history_lag = history_lag_std = 0.0
        near1 = near4 = near8 = far16 = 0.0
        if history_mass > 0:
            lags = (target_abs - sources[history]).astype(np.float64)
            history_lag, history_lag_std = _weighted_mean_std(
                lags / lag_norm, history_weights
            )
            near1 = float(history_weights[lags <= 1].sum() / history_mass)
            near4 = float(history_weights[lags <= 4].sum() / history_mass)
            near8 = float(history_weights[lags <= 8].sum() / history_mass)
            far16 = float(history_weights[lags > 16].sum() / history_mass)

        raw_mask = raw_target == target_abs
        target_raw_source = raw_source[raw_mask]
        target_raw_layer = raw_layer[raw_mask]
        target_raw_weight = raw_weight[raw_mask]
        grouped = np.zeros((3, 2), dtype=np.float64)
        for group_id in range(3):
            if group_channels[group_id] <= 0:
                continue
            group_mask = layer_group[target_raw_layer] == group_id
            if not group_mask.any():
                continue
            group_source = target_raw_source[group_mask]
            group_weight = target_raw_weight[group_mask]
            grouped[group_id, 0] = (
                group_weight[group_source < response_idx].sum() / group_channels[group_id]
            )
            grouped[group_id, 1] = (
                group_weight[group_source >= response_idx].sum() / group_channels[group_id]
            )

        channel_degree = float(channels.sum())
        values = (
            total_mass,
            prompt_share,
            _normalized_entropy(weights),
            history_lag,
            in_degree,
            prompt_degree,
            history_degree,
            in_degree / absolute_target,
            prompt_degree / float(max(response_idx, 1)),
            history_degree / float(row) if row > 0 else 0.0,
            history_share,
            channel_degree / (float(attention.num_channels) * absolute_target),
            prompt_mass,
            history_mass,
            _normalized_entropy(prompt_weights),
            _normalized_entropy(history_weights),
            _top_share(weights, 1),
            _top_share(weights, 3),
            _hhi(weights),
            _top_share(prompt_weights, 1),
            _top_share(history_weights, 1),
            history_lag_std,
            near1,
            near4,
            near8,
            far16,
            grouped[0, 0],
            grouped[1, 0],
            grouped[2, 0],
            grouped[0, 1],
            grouped[1, 1],
            grouped[2, 1],
        )
        matrix[row] = np.asarray(values, dtype=np.float32)

    if not np.isfinite(matrix).all():
        raise ValueError("response graph features must be finite")
    return matrix


def static_feature_blocks(features: np.ndarray) -> dict[str, np.ndarray]:
    """Convert the 32-D response state into six conceptual, non-overlapping blocks."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(RESPONSE_FEATURE_NAMES):
        raise ValueError("features must have shape [response_tokens, 32]")
    idx = {name: index for index, name in enumerate(RESPONSE_FEATURE_NAMES)}
    history_mass = values[:, idx["history_mass"]]
    near1 = values[:, idx["history_near1_share"]]
    near4 = values[:, idx["history_near4_share"]]
    near8 = values[:, idx["history_near8_share"]]
    far16 = values[:, idx["history_far16_share"]]
    source = {name: values[:, index] for name, index in idx.items()}
    source.update(
        {
            "history_bin_1": np.clip(near1, 0.0, 1.0),
            "history_bin_2_4": np.clip(near4 - near1, 0.0, 1.0),
            "history_bin_5_8": np.clip(near8 - near4, 0.0, 1.0),
            "history_bin_9_16": np.where(
                history_mass > 0,
                np.clip(1.0 - near8 - far16, 0.0, 1.0),
                0.0,
            ),
            "history_bin_gt16": np.clip(far16, 0.0, 1.0),
        }
    )
    return {
        block: np.stack([source[name] for name in STATIC_BLOCK_FEATURES[block]], axis=1)
        .astype(np.float32, copy=False)
        for block in STATIC_BLOCK_NAMES
    }


def robust_center_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    scale = 1.4826 * mad
    q25, q75 = np.percentile(matrix, [25, 75], axis=0)
    scale = np.where(scale > 1e-6, scale, (q75 - q25) / 1.349)
    scale = np.where(scale > 1e-6, scale, matrix.std(axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def block_slices(blocks: dict[str, np.ndarray]) -> dict[str, slice]:
    output = {}
    start = 0
    for name in STATIC_BLOCK_NAMES:
        width = blocks[name].shape[1]
        output[name] = slice(start, start + width)
        start += width
    return output


def block_balanced_state(
    blocks: dict[str, np.ndarray], *, reference_blocks: dict[str, np.ndarray] | None = None
) -> np.ndarray:
    """Robust-scale blocks and give every structural concept equal L2 budget."""
    transformed = []
    for name in STATIC_BLOCK_NAMES:
        block = np.asarray(blocks[name], dtype=np.float32)
        reference = block if reference_blocks is None else np.asarray(
            reference_blocks[name], dtype=np.float32
        )
        center, scale = robust_center_scale(reference)
        standardized = np.nan_to_num((block - center) / scale)
        transformed.append(standardized / np.sqrt(max(block.shape[1], 1)))
    return np.concatenate(transformed, axis=1).astype(np.float32, copy=False)


def collect_position_reference(
    control_feature_blocks: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pooled = {name: [] for name in STATIC_BLOCK_NAMES}
    positions = []
    for blocks in control_feature_blocks:
        count = len(next(iter(blocks.values())))
        if count == 0:
            continue
        positions.append(np.linspace(0.0, 1.0, count, dtype=np.float32))
        for name in STATIC_BLOCK_NAMES:
            pooled[name].append(blocks[name])
    if not positions:
        raise ValueError("no correct control nodes are available for position baseline")
    return np.concatenate(positions), {
        name: np.concatenate(rows, axis=0) for name, rows in pooled.items()
    }


def position_residual_blocks(
    blocks: dict[str, np.ndarray],
    *,
    control_positions: np.ndarray,
    control_blocks: dict[str, np.ndarray],
    bandwidth: float = 0.08,
    min_points: int = 48,
) -> dict[str, np.ndarray]:
    """Subtract a position-matched correct-control median and divide by robust scale."""
    count = len(next(iter(blocks.values())))
    target_positions = np.linspace(0.0, 1.0, count, dtype=np.float32)
    control_positions = np.asarray(control_positions, dtype=np.float32)
    output = {name: np.zeros_like(blocks[name], dtype=np.float32) for name in STATIC_BLOCK_NAMES}
    for row, position in enumerate(target_positions):
        distance = np.abs(control_positions - position)
        selected = np.flatnonzero(distance <= float(bandwidth))
        if len(selected) < int(min_points):
            take = min(int(min_points), len(control_positions))
            selected = np.argpartition(distance, take - 1)[:take]
        for name in STATIC_BLOCK_NAMES:
            center, scale = robust_center_scale(control_blocks[name][selected])
            output[name][row] = (blocks[name][row] - center) / scale
    return output


def _probability_vector(source_ids, weights, support) -> np.ndarray:
    vector = np.zeros(len(support), dtype=np.float64)
    mapping = {int(value): index for index, value in enumerate(support)}
    for source, weight in zip(source_ids, weights):
        vector[mapping[int(source)]] += float(weight)
    total = vector.sum()
    if total > 0:
        vector /= total
    return vector


def _js_divergence(source_a, weight_a, source_b, weight_b) -> float:
    support = sorted(set(map(int, source_a)) | set(map(int, source_b)))
    if not support:
        return 0.0
    p = _probability_vector(source_a, weight_a, support)
    q = _probability_vector(source_b, weight_b, support)
    if p.sum() == 0 and q.sum() == 0:
        return 0.0
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def source_transition_features(sample, response_features: np.ndarray) -> np.ndarray:
    """Return source-distribution and neighborhood changes between consecutive targets."""
    attention = sample.attention()
    relations = sample.relation_edges()
    response_idx = int(attention.response_idx)
    count = int(attention.num_response_tokens)
    source = relations["source"].detach().cpu().numpy().astype(np.int64, copy=False)
    target = relations["target"].detach().cpu().numpy().astype(np.int64, copy=False)
    weight = relations["weight"].detach().cpu().numpy().astype(np.float64, copy=False)
    idx = {name: index for index, name in enumerate(RESPONSE_FEATURE_NAMES)}
    routing_names = (
        "early_prompt_mass",
        "middle_prompt_mass",
        "late_prompt_mass",
        "early_history_mass",
        "middle_history_mass",
        "late_history_mass",
    )
    routing = response_features[:, [idx[name] for name in routing_names]]
    output = np.zeros((count, 5), dtype=np.float32)
    previous_sources = previous_weights = None
    for row in range(count):
        mask = target == response_idx + row
        current_sources = source[mask]
        current_weights = weight[mask]
        if row == 0:
            previous_sources, previous_weights = current_sources, current_weights
            continue
        output[row, 0] = _js_divergence(
            previous_sources, previous_weights, current_sources, current_weights
        )
        prev_prompt = previous_sources < response_idx
        curr_prompt = current_sources < response_idx
        output[row, 1] = _js_divergence(
            previous_sources[prev_prompt], previous_weights[prev_prompt],
            current_sources[curr_prompt], current_weights[curr_prompt],
        )
        prev_history = previous_sources >= response_idx
        curr_history = current_sources >= response_idx
        output[row, 2] = _js_divergence(
            previous_sources[prev_history], previous_weights[prev_history],
            current_sources[curr_history], current_weights[curr_history],
        )
        previous_set = set(map(int, previous_sources))
        current_set = set(map(int, current_sources))
        union = previous_set | current_set
        output[row, 3] = 1.0 - (
            len(previous_set & current_set) / len(union) if union else 1.0
        )
        output[row, 4] = float(np.linalg.norm(routing[row] - routing[row - 1]))
        previous_sources, previous_weights = current_sources, current_weights
    return output


def dynamic_state(
    static_state: np.ndarray,
    slices: dict[str, slice],
    source_changes: np.ndarray,
    *,
    rolling_window: int = 8,
) -> np.ndarray:
    """Return 19-D local graph-state transition features."""
    state = np.asarray(static_state, dtype=np.float32)
    delta = np.zeros_like(state)
    if len(state) > 1:
        delta[1:] = state[1:] - state[:-1]
    rolling_z = np.zeros_like(state)
    for row in range(1, len(state)):
        start = max(0, row - int(rolling_window))
        center, scale = robust_center_scale(state[start:row])
        rolling_z[row] = (state[row] - center) / scale

    rows = []
    for row in range(len(state)):
        delta_norms = [float(np.linalg.norm(delta[row]))]
        rolling_norms = [float(np.linalg.norm(rolling_z[row]))]
        for name in STATIC_BLOCK_NAMES:
            delta_norms.append(float(np.linalg.norm(delta[row, slices[name]])))
            rolling_norms.append(float(np.linalg.norm(rolling_z[row, slices[name]])))
        rows.append(delta_norms + rolling_norms + source_changes[row].tolist())
    return np.asarray(rows, dtype=np.float32)
