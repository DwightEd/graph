"""RR-only source-persistence and routing-collapse feature extraction.

This module uses only :class:`ResearchSample` views.  It decomposes the
historical mixed RR coordinate into interpretable terms and builds current-row
collapse diagnostics without reading token labels.

Missing CSR entries remain censored.  Every quantity below is computed only
from exact retained off-diagonal RR edges plus the separately stored exact
attention diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


TOPK_BLOCKS = (
    "mixed_topk",
    "received_topk",
    "diagonal_topk",
    "ratio_topk",
)
SIGNAL_BLOCKS = (*TOPK_BLOCKS, "collapse_channel")

COLLAPSE_FEATURE_NAMES = (
    "log_rr_mass",
    "log_rr_edge_count",
    "source_entropy",
    "log_source_effective_number",
    "source_top1_share",
    "log_source_mean_lag",
    "source_local_mass_share",
    "anchor_turnover",
    "lag_route_velocity",
    "active_channel_fraction",
    "route_effective_rank",
    "cross_channel_consensus",
)

# Directions are preregistered from the premature-local-collapse hypothesis.
# Zero means "diagnostic only" and excludes the feature from the composite.
COLLAPSE_DIRECTIONS = np.asarray(
    [
        0,   # log_rr_mass
        0,   # log_rr_edge_count
        -1,  # source_entropy: lower is more collapsed
        -1,  # effective source count: lower
        +1,  # source top-1 share: higher
        -1,  # mean lag: lower / more local
        +1,  # local mass share: higher
        -1,  # anchor turnover: lower
        -1,  # route velocity: lower
        0,   # active channels: diagnostic
        -1,  # route rank: lower
        0,   # consensus: unconfirmed diagnostic
    ],
    dtype=np.int8,
)


@dataclass(frozen=True)
class RRSignalConfig:
    """Extraction controls shared by fit and held-out scoring."""

    top_k: int = 5
    lag_bins: int = 8
    local_lag_max: int = 4
    anchor_count: int = 8
    block_rows: int = 8192
    causal_position_bins: int = 10
    epsilon: float = 1e-8

    def validate(self) -> None:
        integer_fields = (
            self.top_k,
            self.lag_bins,
            self.local_lag_max,
            self.anchor_count,
            self.block_rows,
            self.causal_position_bins,
        )
        if min(map(int, integer_fields)) < 1:
            raise ValueError("RR signal integer settings must be positive")
        if not np.isfinite(self.epsilon) or float(self.epsilon) <= 0:
            raise ValueError("epsilon must be positive and finite")


@dataclass(frozen=True)
class RRSignalFeatures:
    """Aligned per-token blocks produced from one causal response."""

    blocks: dict[str, np.ndarray]
    collapse_global: np.ndarray
    token_index: np.ndarray
    relative_position: np.ndarray
    causal_position_bucket: np.ndarray
    num_layers: int
    num_heads: int

    @property
    def response_count(self) -> int:
        return int(len(self.token_index))

    @property
    def num_channels(self) -> int:
        return int(self.num_layers) * int(self.num_heads)

    def validate(self) -> "RRSignalFeatures":
        if self.response_count < 1:
            raise ValueError("RR signal feature set is empty")
        if tuple(self.blocks) != SIGNAL_BLOCKS:
            raise ValueError("RR signal block order changed")
        if self.collapse_global.shape != (
            self.response_count,
            len(COLLAPSE_FEATURE_NAMES),
        ):
            raise ValueError("collapse_global has the wrong shape")
        if self.token_index.shape != (self.response_count,):
            raise ValueError("token_index has the wrong shape")
        if self.relative_position.shape != (self.response_count,):
            raise ValueError("relative_position has the wrong shape")
        if self.causal_position_bucket.shape != (self.response_count,):
            raise ValueError("causal_position_bucket has the wrong shape")
        expected = {
            "mixed_topk": self.num_channels * 1,
            "received_topk": self.num_channels * 1,
            "diagonal_topk": self.num_channels * 1,
            "ratio_topk": self.num_channels * 1,
            "collapse_channel": self.num_channels * 6,
        }
        for name, values in self.blocks.items():
            if values.ndim != 2 or len(values) != self.response_count:
                raise ValueError(f"{name} is not a token matrix")
            if name in TOPK_BLOCKS:
                if values.shape[1] % self.num_channels:
                    raise ValueError(f"{name} does not preserve channel blocks")
                if values.shape[1] // self.num_channels < 1:
                    raise ValueError(f"{name} has no per-channel coordinates")
            elif values.shape[1] != expected[name]:
                raise ValueError(f"{name} has the wrong feature width")
            if not bool(np.isfinite(values).all()):
                raise FloatingPointError(f"{name} contains non-finite values")
        if not bool(np.isfinite(self.collapse_global).all()):
            raise FloatingPointError("collapse_global contains non-finite values")
        return self


def causal_position_bucket(token_index: int, bins: int) -> int:
    """Return an online-causal log2 position bucket."""

    token_index = int(token_index)
    bins = int(bins)
    if token_index < 0 or bins < 1:
        raise ValueError("invalid causal position bucket input")
    return min(int(np.floor(np.log2(token_index + 1))), bins - 1)


def _retained_rr_edges(sample, *, block_rows: int):
    """Collect exact retained causal RR edges through the central data view."""

    attention = sample.attention()
    query_parts: list[torch.Tensor] = []
    source_parts: list[torch.Tensor] = []
    channel_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    for block in sample.iter_sparse_attention_blocks(block_rows=block_rows):
        response = block.source >= attention.response_idx
        if not bool(response.any()):
            continue
        query = block.query[response].long()
        source = (block.source[response] - attention.response_idx).long()
        causal = source < query
        if not bool(causal.any()):
            continue
        query_parts.append(query[causal])
        source_parts.append(source[causal])
        channel_parts.append(
            (
                block.layer[response][causal] * attention.num_heads
                + block.head[response][causal]
            ).long()
        )
        weight_parts.append(block.weight[response][causal].float())

    device = attention.response_values.device
    if not query_parts:
        empty_i = torch.empty(0, dtype=torch.long, device=device)
        empty_v = torch.empty(0, dtype=torch.float32, device=device)
        return empty_i, empty_i, empty_i, empty_v
    return (
        torch.cat(query_parts),
        torch.cat(source_parts),
        torch.cat(channel_parts),
        torch.cat(weight_parts),
    )


def _topk(
    values: torch.Tensor,
    *,
    count: int,
    magnitude: bool,
) -> torch.Tensor:
    """Select fixed-width per-channel values with zero padding."""

    channels, width = values.shape
    keep = min(int(count), int(width))
    output = torch.zeros(
        (channels, int(count)),
        dtype=values.dtype,
        device=values.device,
    )
    if keep < 1:
        return output
    ranking = values.abs() if magnitude else values
    indices = torch.topk(
        ranking,
        k=keep,
        dim=1,
        largest=True,
        sorted=True,
    ).indices
    output[:, :keep] = torch.gather(values, 1, indices)
    return output


def _effective_rank_and_consensus(
    channel_lag: np.ndarray,
    *,
    epsilon: float,
) -> tuple[float, float]:
    active = channel_lag.sum(axis=1) > epsilon
    matrix = channel_lag[active]
    if len(matrix) == 0:
        return 0.0, 0.0
    matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), epsilon)
    gram = matrix.T @ matrix
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    total = float(eigenvalues.sum())
    if total <= epsilon:
        effective_rank = 0.0
    else:
        probability = eigenvalues / total
        positive = probability > 0
        effective_rank = float(
            np.exp(-(probability[positive] * np.log(probability[positive])).sum())
        )
        effective_rank /= float(max(1, min(matrix.shape)))
    if len(matrix) < 2:
        consensus = 0.0
    else:
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        unit = matrix / np.maximum(norm, epsilon)
        summed = unit.sum(axis=0)
        numerator = float(summed @ summed) - len(matrix)
        denominator = len(matrix) * (len(matrix) - 1)
        consensus = float(np.clip(numerator / denominator, -1.0, 1.0))
    return effective_rank, consensus


def _current_row_features(
    *,
    response_count: int,
    num_channels: int,
    query: torch.Tensor,
    source: torch.Tensor,
    channel: torch.Tensor,
    weight: torch.Tensor,
    config: RRSignalConfig,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """Build channel-preserving and global RR-collapse states."""

    shape = (response_count, num_channels)
    mass = torch.zeros(shape, dtype=torch.float32, device=device)
    count = torch.zeros(shape, dtype=torch.float32, device=device)
    wlogw = torch.zeros(shape, dtype=torch.float32, device=device)
    maximum = torch.zeros(shape, dtype=torch.float32, device=device)
    lag_sum = torch.zeros(shape, dtype=torch.float32, device=device)
    channel_lag = torch.zeros(
        (response_count, num_channels, config.lag_bins),
        dtype=torch.float32,
        device=device,
    )
    source_mass = torch.zeros(
        (response_count, response_count),
        dtype=torch.float32,
        device=device,
    )

    if query.numel():
        lag = query - source
        lag_bin = torch.floor(torch.log2(lag.float())).long().clamp_max(
            config.lag_bins - 1
        )
        mass.index_put_((query, channel), weight, accumulate=True)
        count.index_put_(
            (query, channel),
            torch.ones_like(weight),
            accumulate=True,
        )
        wlogw.index_put_(
            (query, channel),
            weight * torch.log(weight.clamp_min(config.epsilon)),
            accumulate=True,
        )
        flat = query * num_channels + channel
        maximum.view(-1).scatter_reduce_(
            0,
            flat,
            weight,
            reduce="amax",
            include_self=True,
        )
        lag_sum.index_put_(
            (query, channel),
            weight * lag.float(),
            accumulate=True,
        )
        channel_lag.index_put_(
            (query, channel, lag_bin),
            weight,
            accumulate=True,
        )
        source_mass.index_put_((query, source), weight, accumulate=True)

    raw_entropy = torch.zeros_like(mass)
    active = mass > config.epsilon
    raw_entropy[active] = (
        torch.log(mass[active])
        - wlogw[active] / mass[active]
    )
    normalized_entropy = torch.zeros_like(mass)
    multiple = count > 1
    normalized_entropy[multiple] = (
        raw_entropy[multiple] / torch.log(count[multiple])
    )
    top1 = torch.where(
        active,
        maximum / mass.clamp_min(config.epsilon),
        torch.zeros_like(mass),
    )
    effective = torch.where(
        active,
        torch.exp(raw_entropy),
        torch.zeros_like(raw_entropy),
    )
    mean_lag = torch.where(
        active,
        lag_sum / mass.clamp_min(config.epsilon),
        torch.zeros_like(lag_sum),
    )
    channel_features = torch.stack(
        (
            torch.log1p(mass),
            normalized_entropy,
            top1,
            torch.log1p(effective),
            torch.log1p(mean_lag),
            active.float(),
        ),
        dim=2,
    ).reshape(response_count, -1)

    source_mass_np = source_mass.detach().cpu().numpy().astype(np.float64)
    channel_lag_np = channel_lag.detach().cpu().numpy().astype(np.float64)
    total_mass_np = mass.sum(dim=1).detach().cpu().numpy().astype(np.float64)
    total_count_np = count.sum(dim=1).detach().cpu().numpy().astype(np.float64)
    active_fraction_np = (
        active.float().mean(dim=1).detach().cpu().numpy().astype(np.float64)
    )

    global_features = np.zeros(
        (response_count, len(COLLAPSE_FEATURE_NAMES)),
        dtype=np.float32,
    )
    previous_anchors: set[int] = set()
    previous_lag_distribution: np.ndarray | None = None
    for token in range(response_count):
        source_weights = source_mass_np[token, :token]
        total = float(source_weights.sum())
        entropy_normalized = 0.0
        effective_sources = 0.0
        source_top1 = 0.0
        mean_lag_value = 0.0
        local_share = 0.0
        anchors: set[int] = set()
        if total > config.epsilon:
            probability = source_weights / total
            positive = probability > 0
            raw = float(
                -(probability[positive] * np.log(probability[positive])).sum()
            )
            entropy_normalized = raw / float(np.log(max(int(positive.sum()), 2)))
            effective_sources = float(np.exp(raw))
            source_top1 = float(probability.max(initial=0.0))
            lag_values = token - np.arange(token, dtype=np.float64)
            mean_lag_value = float(probability @ lag_values)
            local_share = float(
                probability[lag_values <= config.local_lag_max].sum()
            )
            positive_source = np.flatnonzero(source_weights > 0)
            keep = min(config.anchor_count, len(positive_source))
            if keep:
                selected = positive_source[
                    np.argsort(source_weights[positive_source], kind="stable")[-keep:]
                ]
                anchors = set(map(int, selected.tolist()))

        turnover = 0.0
        if token > 0:
            union = anchors | previous_anchors
            turnover = (
                0.0
                if not union
                else 1.0 - len(anchors & previous_anchors) / len(union)
            )
        previous_anchors = anchors

        lag_distribution = channel_lag_np[token].sum(axis=0)
        lag_total = float(lag_distribution.sum())
        if lag_total > config.epsilon:
            lag_distribution = lag_distribution / lag_total
        else:
            lag_distribution = np.zeros(config.lag_bins, dtype=np.float64)
        velocity = 0.0
        if previous_lag_distribution is not None:
            denominator = (
                np.linalg.norm(previous_lag_distribution)
                * np.linalg.norm(lag_distribution)
            )
            if denominator > config.epsilon:
                velocity = 1.0 - float(
                    np.clip(
                        previous_lag_distribution @ lag_distribution / denominator,
                        -1.0,
                        1.0,
                    )
                )
            elif bool(np.any(previous_lag_distribution) != np.any(lag_distribution)):
                velocity = 1.0
        previous_lag_distribution = lag_distribution

        route_rank, consensus = _effective_rank_and_consensus(
            channel_lag_np[token],
            epsilon=config.epsilon,
        )
        global_features[token] = np.asarray(
            (
                np.log1p(total_mass_np[token]),
                np.log1p(total_count_np[token]),
                entropy_normalized,
                np.log1p(effective_sources),
                source_top1,
                np.log1p(mean_lag_value),
                local_share,
                turnover,
                velocity,
                active_fraction_np[token],
                route_rank,
                consensus,
            ),
            dtype=np.float32,
        )

    return (
        channel_features.detach().cpu().numpy().astype(np.float32, copy=False),
        global_features,
    )


def extract_rr_signal_features(
    sample,
    *,
    config: RRSignalConfig | None = None,
) -> RRSignalFeatures:
    """Extract decomposition blocks and local-collapse variables for all tokens."""

    config = RRSignalConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    num_layers = int(attention.num_layers)
    num_heads = int(attention.num_heads)
    num_channels = int(attention.num_channels)
    if response_count < 1:
        raise ValueError("RR signal extraction requires a non-empty response")

    device = attention.response_values.device
    diagonal = (
        attention.attention_diagonal[:, :, attention.response_idx :]
        .float()
        .reshape(num_channels, response_count)
    )
    query, source, channel, weight = _retained_rr_edges(
        sample,
        block_rows=config.block_rows,
    )
    collapse_channel, collapse_global = _current_row_features(
        response_count=response_count,
        num_channels=num_channels,
        query=query,
        source=source,
        channel=channel,
        weight=weight,
        config=config,
        device=device,
    )

    outputs = {
        name: torch.zeros(
            (response_count, num_channels, config.top_k),
            dtype=torch.float32,
            device=device,
        )
        for name in TOPK_BLOCKS
    }
    future_received = torch.zeros_like(diagonal)

    if query.numel():
        order = torch.argsort(query, stable=True)
        sorted_query = query[order]
        stops = torch.searchsorted(
            sorted_query,
            torch.arange(response_count, device=device),
            right=True,
        )
    else:
        order = torch.empty(0, dtype=torch.long, device=device)
        stops = torch.zeros(response_count, dtype=torch.long, device=device)

    start = 0
    for token, stop in enumerate(stops.tolist()):
        selected = order[start:stop]
        if selected.numel():
            future_received.index_put_(
                (channel[selected], source[selected]),
                weight[selected],
                accumulate=True,
            )
        start = stop

        active = token + 1
        age = (
            float(token)
            - torch.arange(active, dtype=torch.float32, device=device)
            + 1.0
        )
        future = future_received[:, :active]
        current_diagonal = diagonal[:, :active]
        received_age = future / age.unsqueeze(0)
        diagonal_age = (
            current_diagonal
            * ((age - 1.0) / age).unsqueeze(0)
        )
        mixed = received_age - diagonal_age
        future_opportunities = (age - 1.0).clamp_min(1.0)
        future_mean = future / future_opportunities.unsqueeze(0)
        ratio = torch.log1p(
            future_mean
            / current_diagonal.clamp_min(config.epsilon)
        )
        ratio[:, age <= 1.0] = 0.0

        outputs["mixed_topk"][token] = _topk(
            mixed,
            count=config.top_k,
            magnitude=True,
        )
        outputs["received_topk"][token] = _topk(
            received_age,
            count=config.top_k,
            magnitude=False,
        )
        outputs["diagonal_topk"][token] = _topk(
            diagonal_age,
            count=config.top_k,
            magnitude=False,
        )
        outputs["ratio_topk"][token] = _topk(
            ratio,
            count=config.top_k,
            magnitude=False,
        )

    blocks = {
        name: outputs[name].reshape(response_count, -1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
        for name in TOPK_BLOCKS
    }
    blocks["collapse_channel"] = collapse_channel

    token_index = np.arange(response_count, dtype=np.int32)
    relative_position = token_index.astype(np.float32) / float(
        max(response_count - 1, 1)
    )
    causal_bucket = np.asarray(
        [
            causal_position_bucket(token, config.causal_position_bins)
            for token in token_index
        ],
        dtype=np.int16,
    )
    return RRSignalFeatures(
        blocks=blocks,
        collapse_global=collapse_global,
        token_index=token_index,
        relative_position=relative_position,
        causal_position_bucket=causal_bucket,
        num_layers=num_layers,
        num_heads=num_heads,
    ).validate()


def features_per_channel(block: str, config: RRSignalConfig) -> int:
    if block in TOPK_BLOCKS:
        return int(config.top_k)
    if block == "collapse_channel":
        return 6
    raise KeyError(block)
