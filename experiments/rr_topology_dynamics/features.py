"""Label-free topology and convergence features for causal RR attention graphs.

The functions in this module consume only :class:`ResearchSample` views. They
never open labels and never parse raw NPZ/PT files. The representation separates
three questions:

1. Does the current response-history routing collapse from many channel/lag
   modes to a smaller set of routes?
2. Is that convergence prompt-grounded or a response-only feedback loop?
3. Which layer/head/source/lag coordinates account for escape from the frozen
   RR spectral subspace?

Attention concentration is treated as a structural routing property, not as a
claim about output confidence. The current cache does not contain logits/NLL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.spectral_feasibility.experiment import load_spectral_reference
from experiments.spectral_feasibility.representations import (
    SpectralConfig,
    prefix_causal_attention_modes,
    response_position_bin,
)


SCALAR_FEATURE_NAMES = (
    "rr_retained_mass",
    "rr_retained_edge_count",
    "active_channel_fraction",
    "route_effective_rank",
    "route_participation_rank",
    "route_stable_rank",
    "route_spectral_entropy",
    "route_top1_energy_share",
    "cross_head_route_consensus",
    "source_effective_number",
    "source_entropy",
    "source_top1_share",
    "source_mean_lag",
    "source_lag_std",
    "source_far_mass_fraction",
    "channel_route_velocity",
    "source_route_velocity",
    "anchor_turnover",
    "offline_route_distance_to_final",
    "offline_source_distance_to_final",
    "direct_prompt_share",
    "prompt_groundedness",
    "grounded_rr_relay",
    "ungrounded_rr_feedback",
    "spectral_residual_energy",
    "spectral_embedding_velocity",
    "residual_effective_channels",
    "residual_channel_entropy",
    "residual_channel_top1_share",
    "residual_channel_top5pct_share",
    "residual_weighted_lag",
    "residual_recent_lag_share",
    "residual_mid_lag_share",
    "residual_far_lag_share",
    "residual_grounded_source_share",
    "residual_source_effective_number",
    "residual_source_top1_share",
)


@dataclass(frozen=True)
class TopologyDynamicsConfig:
    lag_bins: int = 8
    spectral_top_k: int = 5
    block_rows: int = 8192
    position_bins: int = 8
    top_source_count: int = 8
    recent_lag_max: int = 4
    mid_lag_max: int = 16
    far_lag_fraction: float = 0.5
    epsilon: float = 1e-8

    def validate(self) -> None:
        integer_fields = (
            self.lag_bins,
            self.spectral_top_k,
            self.block_rows,
            self.position_bins,
            self.top_source_count,
            self.recent_lag_max,
            self.mid_lag_max,
        )
        if min(map(int, integer_fields)) < 1:
            raise ValueError("topology-dynamics integer settings must be positive")
        if self.mid_lag_max < self.recent_lag_max:
            raise ValueError("mid_lag_max must be >= recent_lag_max")
        if not 0.0 < float(self.far_lag_fraction) <= 1.0:
            raise ValueError("far_lag_fraction must be in (0,1]")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")


def load_rr_reference(path):
    """Load and minimally validate the frozen RR spectral reference."""
    reference = load_spectral_reference(path)
    required = {
        "num_layers",
        "num_heads",
        "top_k",
        "position_bins",
        "rr_center",
        "rr_scale",
        "rr_pca_mean",
        "rr_pca_components",
        "rr_pca_whiten_scale",
    }
    missing = required.difference(reference)
    if missing:
        raise ValueError(f"RR spectral reference misses fields: {sorted(missing)}")
    return reference


def _entropy_effective(probability: torch.Tensor, *, dimension: int, epsilon: float):
    probability = probability.clamp_min(0)
    total = probability.sum(dim=dimension, keepdim=True)
    normalized = probability / total.clamp_min(epsilon)
    entropy = -(normalized * normalized.clamp_min(epsilon).log()).sum(dim=dimension)
    effective = entropy.exp()
    valid = total.squeeze(dimension) > epsilon
    entropy = torch.where(valid, entropy, torch.zeros_like(entropy))
    effective = torch.where(valid, effective, torch.zeros_like(effective))
    return normalized, entropy, effective


def _batched_route_spectrum(route_probability: torch.Tensor, epsilon: float):
    """Return effective-rank summaries of ``[token, channel, lag_bin]`` routes."""
    gram = torch.einsum("tcb,tcd->tbd", route_probability, route_probability)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = eigenvalues.sum(dim=1)
    normalized = eigenvalues / total[:, None].clamp_min(epsilon)
    entropy = -(
        normalized * normalized.clamp_min(epsilon).log()
    ).sum(dim=1)
    effective_rank = entropy.exp()
    participation = total.square() / eigenvalues.square().sum(dim=1).clamp_min(epsilon)
    maximum = eigenvalues.max(dim=1).values
    stable_rank = total / maximum.clamp_min(epsilon)
    top1 = maximum / total.clamp_min(epsilon)
    entropy_normalized = entropy / float(np.log(max(route_probability.shape[-1], 2)))
    valid = total > epsilon
    output = tuple(
        torch.where(valid, value, torch.zeros_like(value))
        for value in (
            effective_rank,
            participation,
            stable_rank,
            entropy_normalized,
            top1,
        )
    )
    return output


def _mean_pairwise_cosine(route_probability: torch.Tensor, active: torch.Tensor, epsilon: float):
    norm = route_probability.norm(dim=2)
    unit = route_probability / norm[:, :, None].clamp_min(epsilon)
    unit = torch.where(active[:, :, None], unit, torch.zeros_like(unit))
    count = active.sum(dim=1).to(route_probability.dtype)
    summed = unit.sum(dim=1)
    numerator = summed.square().sum(dim=1) - count
    denominator = count * (count - 1.0)
    consensus = numerator / denominator.clamp_min(1.0)
    return torch.where(count >= 2, consensus.clamp(-1.0, 1.0), torch.zeros_like(consensus))


def _consecutive_cosine_distance(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros(len(values), dtype=np.float32)
    if len(values) < 2:
        return result
    left = values[:-1]
    right = values[1:]
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(
        numerator,
        np.maximum(denominator, epsilon),
        out=np.ones_like(numerator),
        where=denominator > epsilon,
    )
    result[1:] = (1.0 - np.clip(cosine, -1.0, 1.0)).astype(np.float32)
    return result


def _distance_to_final(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Offline-only cosine distance to the final route state."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    final = values[-1]
    final_norm = float(np.linalg.norm(final))
    norms = np.linalg.norm(values, axis=1)
    numerator = values @ final
    denominator = norms * final_norm
    cosine = np.divide(
        numerator,
        np.maximum(denominator, epsilon),
        out=np.ones_like(numerator),
        where=denominator > epsilon,
    )
    return (1.0 - np.clip(cosine, -1.0, 1.0)).astype(np.float32)


def _anchor_turnover(source_mass: np.ndarray, top_count: int) -> np.ndarray:
    response_count = len(source_mass)
    result = np.zeros(response_count, dtype=np.float32)
    previous: set[int] = set()
    for token in range(response_count):
        current_values = source_mass[token, :token]
        positive = np.flatnonzero(current_values > 0)
        if len(positive):
            keep = min(int(top_count), len(positive))
            chosen = positive[
                np.argpartition(current_values[positive], len(positive) - keep)[-keep:]
            ]
            current = set(map(int, chosen.tolist()))
        else:
            current = set()
        if token > 0:
            union = previous | current
            result[token] = 0.0 if not union else 1.0 - len(previous & current) / len(union)
        previous = current
    return result


def _prompt_groundedness(prompt_mass: np.ndarray, source_mass: np.ndarray, epsilon: float):
    response_count = len(prompt_mass)
    grounded = np.zeros(response_count, dtype=np.float32)
    direct_share = np.zeros(response_count, dtype=np.float32)
    relay = np.zeros(response_count, dtype=np.float32)
    feedback = np.zeros(response_count, dtype=np.float32)
    for token in range(response_count):
        rr = source_mass[token, :token]
        rr_total = float(rr.sum())
        prompt = float(prompt_mass[token])
        direct_share[token] = prompt / max(prompt + rr_total, epsilon)
        if rr_total > epsilon:
            probability = rr / rr_total
            relay[token] = float(probability @ grounded[:token])
            feedback[token] = float(probability @ (1.0 - grounded[:token]))
        grounded[token] = direct_share[token] + (
            1.0 - direct_share[token]
        ) * relay[token]
    return direct_share, grounded, relay, feedback


def _source_distribution_features(source_mass: np.ndarray, config: TopologyDynamicsConfig):
    response_count = len(source_mass)
    effective = np.zeros(response_count, dtype=np.float32)
    entropy = np.zeros(response_count, dtype=np.float32)
    top1 = np.zeros(response_count, dtype=np.float32)
    mean_lag = np.zeros(response_count, dtype=np.float32)
    std_lag = np.zeros(response_count, dtype=np.float32)
    far_share = np.zeros(response_count, dtype=np.float32)
    normalized = np.zeros_like(source_mass, dtype=np.float32)
    for token in range(response_count):
        weights = source_mass[token, :token].astype(np.float64, copy=False)
        total = float(weights.sum())
        if total <= config.epsilon:
            continue
        probability = weights / total
        normalized[token, :token] = probability.astype(np.float32)
        positive = probability > 0
        h = float(-(probability[positive] * np.log(probability[positive])).sum())
        entropy[token] = h / float(np.log(max(token, 2)))
        effective[token] = float(np.exp(h))
        top1[token] = float(probability.max())
        lag = token - np.arange(token, dtype=np.float64)
        mean = float(probability @ lag)
        mean_lag[token] = mean
        std_lag[token] = float(np.sqrt(probability @ np.square(lag - mean)))
        threshold = max(1.0, float(token) * config.far_lag_fraction)
        far_share[token] = float(probability[lag >= threshold].sum())
    return {
        "source_effective_number": effective,
        "source_entropy": entropy,
        "source_top1_share": top1,
        "source_mean_lag": mean_lag,
        "source_lag_std": std_lag,
        "source_far_mass_fraction": far_share,
        "source_probability": normalized,
    }


def _residual_source_attribution(
    residual_coordinate_energy: np.ndarray,
    source_index: np.ndarray,
    lag: np.ndarray,
    groundedness: np.ndarray,
    config: TopologyDynamicsConfig,
):
    response_count = residual_coordinate_energy.shape[0]
    weighted_lag = np.zeros(response_count, dtype=np.float32)
    recent = np.zeros(response_count, dtype=np.float32)
    middle = np.zeros(response_count, dtype=np.float32)
    far = np.zeros(response_count, dtype=np.float32)
    grounded = np.zeros(response_count, dtype=np.float32)
    effective_source = np.zeros(response_count, dtype=np.float32)
    source_top1 = np.zeros(response_count, dtype=np.float32)

    for token in range(response_count):
        weight = residual_coordinate_energy[token].reshape(-1).astype(np.float64)
        source = source_index[token].reshape(-1)
        current_lag = lag[token].reshape(-1)
        valid = (source >= 0) & (source < token) & (current_lag > 0) & (weight > 0)
        if not bool(valid.any()):
            continue
        weight = weight[valid]
        source = source[valid]
        current_lag = current_lag[valid].astype(np.float64)
        total = float(weight.sum())
        probability = weight / total
        weighted_lag[token] = float(probability @ current_lag)
        recent[token] = float(probability[current_lag <= config.recent_lag_max].sum())
        middle[token] = float(
            probability[
                (current_lag > config.recent_lag_max)
                & (current_lag <= config.mid_lag_max)
            ].sum()
        )
        far[token] = float(probability[current_lag > config.mid_lag_max].sum())
        grounded[token] = float(probability @ groundedness[source])
        by_source = np.bincount(source, weights=weight, minlength=max(token, 1))
        source_probability = by_source / max(float(by_source.sum()), config.epsilon)
        positive = source_probability > 0
        effective_source[token] = float(
            np.exp(-(source_probability[positive] * np.log(source_probability[positive])).sum())
        )
        source_top1[token] = float(source_probability.max(initial=0.0))

    return {
        "residual_weighted_lag": weighted_lag,
        "residual_recent_lag_share": recent,
        "residual_mid_lag_share": middle,
        "residual_far_lag_share": far,
        "residual_grounded_source_share": grounded,
        "residual_source_effective_number": effective_source,
        "residual_source_top1_share": source_top1,
    }


def extract_sample_topology_dynamics(
    sample,
    spectral_reference,
    *,
    config: TopologyDynamicsConfig | None = None,
):
    """Extract causal routing, grounding, and spectral-residual topology.

    The returned scalar matrix follows :data:`SCALAR_FEATURE_NAMES`. Layer and
    selected-spectral-rank profiles remain separate arrays so later analyses do
    not hide where a global residual originates.
    """
    config = TopologyDynamicsConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    channels = int(attention.num_channels)
    prompt_count = int(attention.response_idx)
    if response_count < 1 or prompt_count < 1:
        raise ValueError("topology dynamics requires non-empty prompt and response")
    if layers != int(spectral_reference["num_layers"]) or heads != int(
        spectral_reference["num_heads"]
    ):
        raise ValueError("sample attention geometry differs from RR spectral reference")
    if int(spectral_reference["top_k"]) != int(config.spectral_top_k):
        raise ValueError("spectral_top_k differs from frozen RR reference")

    device = attention.response_values.device
    route = torch.zeros(
        (response_count, channels, config.lag_bins),
        dtype=torch.float32,
        device=device,
    )
    channel_mass = torch.zeros(
        (response_count, channels), dtype=torch.float32, device=device
    )
    channel_edges = torch.zeros_like(channel_mass)
    prompt_channel_mass = torch.zeros_like(channel_mass)
    source_mass = torch.zeros(
        (response_count, response_count), dtype=torch.float32, device=device
    )

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        channel = (block.layer * heads + block.head).long()
        query = block.query.long()
        prompt = block.source < prompt_count
        if bool(prompt.any()):
            prompt_channel_mass.index_put_(
                (query[prompt], channel[prompt]),
                block.weight[prompt].float(),
                accumulate=True,
            )
        history = ~prompt
        if not bool(history.any()):
            continue
        history_query = query[history]
        history_channel = channel[history]
        history_source = (block.source[history] - prompt_count).long()
        history_weight = block.weight[history].float()
        lag = history_query - history_source
        if bool((lag <= 0).any()):
            raise ValueError("RR cache contains a non-causal retained edge")
        lag_bin = torch.floor(torch.log2(lag.float())).long().clamp_max(
            config.lag_bins - 1
        )
        route.index_put_(
            (history_query, history_channel, lag_bin),
            history_weight,
            accumulate=True,
        )
        channel_mass.index_put_(
            (history_query, history_channel), history_weight, accumulate=True
        )
        channel_edges.index_put_(
            (history_query, history_channel),
            torch.ones_like(history_weight),
            accumulate=True,
        )
        source_mass.index_put_(
            (history_query, history_source), history_weight, accumulate=True
        )

    active = channel_mass > config.epsilon
    route_probability = route / channel_mass[:, :, None].clamp_min(config.epsilon)
    route_probability = torch.where(
        active[:, :, None], route_probability, torch.zeros_like(route_probability)
    )
    (
        route_effective_rank,
        route_participation_rank,
        route_stable_rank,
        route_spectral_entropy,
        route_top1_share,
    ) = _batched_route_spectrum(route_probability, config.epsilon)
    route_consensus = _mean_pairwise_cosine(
        route_probability, active, config.epsilon
    )

    layer_route = route_probability.reshape(
        response_count, layers, heads, config.lag_bins
    )
    layer_gram = torch.einsum("tlhb,tlhd->tlbd", layer_route, layer_route)
    layer_eigen = torch.linalg.eigvalsh(layer_gram).clamp_min(0)
    layer_total = layer_eigen.sum(dim=2)
    layer_probability = layer_eigen / layer_total[:, :, None].clamp_min(config.epsilon)
    layer_entropy = -(
        layer_probability * layer_probability.clamp_min(config.epsilon).log()
    ).sum(dim=2)
    layer_effective_rank = torch.where(
        layer_total > config.epsilon,
        layer_entropy.exp(),
        torch.zeros_like(layer_entropy),
    )
    layer_active = active.reshape(response_count, layers, heads)
    layer_consensus = _mean_pairwise_cosine(
        layer_route.reshape(response_count * layers, heads, config.lag_bins),
        layer_active.reshape(response_count * layers, heads),
        config.epsilon,
    ).reshape(response_count, layers)

    route_numpy = route_probability.detach().cpu().numpy().astype(np.float32)
    route_flat = route_numpy.reshape(response_count, -1)
    source_mass_numpy = source_mass.detach().cpu().numpy().astype(np.float32)
    source_features = _source_distribution_features(source_mass_numpy, config)
    prompt_mass = prompt_channel_mass.sum(dim=1).detach().cpu().numpy()
    direct_prompt, groundedness, grounded_relay, ungrounded_feedback = (
        _prompt_groundedness(prompt_mass, source_mass_numpy, config.epsilon)
    )

    spectral_config = SpectralConfig(
        top_k=config.spectral_top_k,
        block_rows=config.block_rows,
    )
    modes = prefix_causal_attention_modes(sample, config=spectral_config)
    values = modes.values.reshape(response_count, -1)
    position_bins = int(spectral_reference["position_bins"])
    bins = np.asarray(
        [
            response_position_bin(token, response_count, position_bins)
            for token in range(response_count)
        ],
        dtype=np.int16,
    )
    standardized = (
        values - spectral_reference["rr_center"][bins]
    ) / spectral_reference["rr_scale"][bins]
    centered = standardized - spectral_reference["rr_pca_mean"]
    scores = centered @ spectral_reference["rr_pca_components"].T
    embedding = scores / spectral_reference["rr_pca_whiten_scale"]
    reconstructed = (
        scores @ spectral_reference["rr_pca_components"]
        + spectral_reference["rr_pca_mean"]
    )
    residual_vector = standardized - reconstructed
    residual_coordinate_energy = np.square(residual_vector).reshape(
        response_count, channels, config.spectral_top_k
    )
    residual_energy = residual_coordinate_energy.mean(axis=(1, 2)).astype(np.float32)
    channel_residual = residual_coordinate_energy.mean(axis=2)
    residual_total = channel_residual.sum(axis=1)
    residual_probability = np.divide(
        channel_residual,
        np.maximum(residual_total[:, None], config.epsilon),
        out=np.zeros_like(channel_residual),
        where=residual_total[:, None] > config.epsilon,
    )
    residual_positive = residual_probability > 0
    residual_entropy_raw = -np.where(
        residual_positive,
        residual_probability * np.log(np.maximum(residual_probability, config.epsilon)),
        0.0,
    ).sum(axis=1)
    residual_effective_channels = np.exp(residual_entropy_raw).astype(np.float32)
    residual_channel_entropy = (
        residual_entropy_raw / float(np.log(max(channels, 2)))
    ).astype(np.float32)
    residual_top1 = residual_probability.max(axis=1).astype(np.float32)
    tail_count = max(1, int(np.ceil(channels * 0.05)))
    residual_top5 = np.partition(
        residual_probability, channels - tail_count, axis=1
    )[:, -tail_count:].sum(axis=1).astype(np.float32)
    layer_residual = channel_residual.reshape(response_count, layers, heads).mean(axis=2)
    rank_residual = residual_coordinate_energy.mean(axis=1)
    source_attribution = _residual_source_attribution(
        residual_coordinate_energy,
        modes.source_index,
        modes.lag,
        groundedness,
        config,
    )

    scalar = {
        "rr_retained_mass": channel_mass.sum(dim=1).detach().cpu().numpy(),
        "rr_retained_edge_count": channel_edges.sum(dim=1).detach().cpu().numpy(),
        "active_channel_fraction": active.float().mean(dim=1).detach().cpu().numpy(),
        "route_effective_rank": route_effective_rank.detach().cpu().numpy(),
        "route_participation_rank": route_participation_rank.detach().cpu().numpy(),
        "route_stable_rank": route_stable_rank.detach().cpu().numpy(),
        "route_spectral_entropy": route_spectral_entropy.detach().cpu().numpy(),
        "route_top1_energy_share": route_top1_share.detach().cpu().numpy(),
        "cross_head_route_consensus": route_consensus.detach().cpu().numpy(),
        **{name: source_features[name] for name in (
            "source_effective_number",
            "source_entropy",
            "source_top1_share",
            "source_mean_lag",
            "source_lag_std",
            "source_far_mass_fraction",
        )},
        "channel_route_velocity": _consecutive_cosine_distance(
            route_flat, config.epsilon
        ),
        "source_route_velocity": _consecutive_cosine_distance(
            source_features["source_probability"], config.epsilon
        ),
        "anchor_turnover": _anchor_turnover(
            source_mass_numpy, config.top_source_count
        ),
        "offline_route_distance_to_final": _distance_to_final(
            route_flat, config.epsilon
        ),
        "offline_source_distance_to_final": _distance_to_final(
            source_features["source_probability"], config.epsilon
        ),
        "direct_prompt_share": direct_prompt,
        "prompt_groundedness": groundedness,
        "grounded_rr_relay": grounded_relay,
        "ungrounded_rr_feedback": ungrounded_feedback,
        "spectral_residual_energy": residual_energy,
        "spectral_embedding_velocity": _consecutive_cosine_distance(
            embedding, config.epsilon
        ),
        "residual_effective_channels": residual_effective_channels,
        "residual_channel_entropy": residual_channel_entropy,
        "residual_channel_top1_share": residual_top1,
        "residual_channel_top5pct_share": residual_top5,
        **source_attribution,
    }
    matrix = np.column_stack(
        [np.asarray(scalar[name], dtype=np.float32) for name in SCALAR_FEATURE_NAMES]
    )
    return {
        "feature_names": np.asarray(SCALAR_FEATURE_NAMES, dtype=str),
        "features": matrix.astype(np.float32, copy=False),
        "position_bin": np.asarray(
            [
                response_position_bin(token, response_count, config.position_bins)
                for token in range(response_count)
            ],
            dtype=np.int16,
        ),
        "layer_route_effective_rank": layer_effective_rank.detach().cpu().numpy().astype(np.float32),
        "layer_route_consensus": layer_consensus.detach().cpu().numpy().astype(np.float32),
        "layer_residual_energy": layer_residual.astype(np.float32),
        "spectral_rank_residual_energy": rank_residual.astype(np.float32),
        "rr_embedding": embedding.astype(np.float32),
    }
