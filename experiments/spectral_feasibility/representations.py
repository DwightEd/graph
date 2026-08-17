"""Causal response-history coordinates for sparse attention graphs.

The active detector in this package is deliberately RR-only. For every
layer/head channel and response prefix we treat retained response-to-response
attention as an artificial age-normalized triangular operator. Its signed
diagonal coordinates are used directly, without a standard graph Laplacian,
eigendecomposition, or eigenvector rotation. Strongest-magnitude coordinates
are kept per channel without averaging layer/head axes.

Prompt routing belongs to the separate topology-dynamics audit. Missing CSR
entries are cache-censored (<= ``attention_floor``); they are never
reconstructed as exact zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SpectralConfig:
    top_k: int = 5
    block_rows: int = 8192
    position_bins: int = 4
    pca_dim: int = 32
    reference_per_sample: int = 6
    trim_fraction: float = 0.90
    calibration_fraction: float = 0.25
    split_seed: int = 20260815
    channel_tail_fraction: float = 0.05
    attribution_topk: int = 8
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.block_rows < 1:
            raise ValueError("block_rows must be positive")
        if self.position_bins < 1:
            raise ValueError("position_bins must be positive")
        if self.pca_dim < 1:
            raise ValueError("pca_dim must be positive")
        if self.reference_per_sample < 1:
            raise ValueError("reference_per_sample must be positive")
        if not 0.5 <= float(self.trim_fraction) <= 1.0:
            raise ValueError("trim_fraction must be in [0.5, 1]")
        if not 0.0 < float(self.calibration_fraction) < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1)")
        if int(self.split_seed) < 0:
            raise ValueError("split_seed must be non-negative")
        if not 0.0 < float(self.channel_tail_fraction) <= 1.0:
            raise ValueError("channel_tail_fraction must be in (0, 1]")
        if self.attribution_topk < 1:
            raise ValueError("attribution_topk must be positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")


@dataclass(frozen=True)
class PrefixCausalAttentionModes:
    """Selected causal-attention coordinates and their source identities.

    Arrays have shape ``[requested_token, layer_head_channel, top_k]``.
    ``source_index`` is response-relative and uses ``-1`` for padding when the
    prefix contains fewer than ``top_k`` source nodes. ``lag`` is measured from
    the requested response token to that selected source and uses ``-1`` for
    padding. Values retain their original signs.
    """

    positions: np.ndarray
    values: np.ndarray
    source_index: np.ndarray
    lag: np.ndarray


def response_position_bin(position: int, response_count: int, bins: int) -> int:
    """Map a response-relative token index to a relative-position bin."""
    if response_count < 1:
        raise ValueError("response_count must be positive")
    if not 0 <= int(position) < response_count:
        raise IndexError("response position is outside the response")
    if bins < 1:
        raise ValueError("bins must be positive")
    if response_count == 1:
        return 0
    fraction = float(position) / float(response_count - 1)
    return min(int(fraction * bins), bins - 1)


def reference_positions(response_count: int, count: int) -> np.ndarray:
    """Choose deterministic approximately-uniform train reference tokens."""
    if response_count < 1:
        return np.empty(0, dtype=np.int64)
    count = min(int(count), response_count)
    if count < 1:
        raise ValueError("reference position count must be positive")
    if count == response_count:
        return np.arange(response_count, dtype=np.int64)
    quantiles = (np.arange(count, dtype=np.float64) + 0.5) / float(count)
    positions = np.rint(quantiles * (response_count - 1)).astype(np.int64)
    return np.unique(np.clip(positions, 0, response_count - 1))


def _requested_positions(response_count: int, positions) -> np.ndarray:
    if positions is None:
        return np.arange(response_count, dtype=np.int64)
    requested = np.asarray(list(positions), dtype=np.int64)
    if requested.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if requested.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(requested < 0) or np.any(requested >= response_count):
        raise IndexError("requested response position is outside the response")
    return np.unique(requested)


def _retained_response_edges(sample, *, block_rows: int):
    """Collect retained RR entries exclusively through ``ResearchSample``."""
    attention = sample.attention()
    channels: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for block in sample.iter_sparse_attention_blocks(block_rows=block_rows):
        response = block.source >= attention.response_idx
        if not bool(response.any()):
            continue
        channels.append(
            (block.layer[response] * attention.num_heads + block.head[response]).long()
        )
        queries.append(block.query[response].long())
        sources.append((block.source[response] - attention.response_idx).long())
        weights.append(block.weight[response].float())

    device = attention.response_values.device
    if not channels:
        empty_i = torch.empty(0, dtype=torch.long, device=device)
        empty_v = torch.empty(0, dtype=torch.float32, device=device)
        return empty_i, empty_i, empty_i, empty_v
    return (
        torch.cat(channels),
        torch.cat(queries),
        torch.cat(sources),
        torch.cat(weights),
    )


def causal_prefix_edge_batches(
    edge_query: torch.Tensor,
    positions: np.ndarray,
):
    """Yield each retained edge once, at its first requested causal prefix.

    The returned edge indices are stably ordered by query position. This lets a
    prefix scan increment ``received`` with only the rows newly admitted since
    the preceding requested token, instead of repeatedly masking every stored
    RR edge. Edges after the final requested prefix are intentionally omitted.
    """
    if edge_query.ndim != 1:
        raise ValueError("edge_query must be one-dimensional")
    requested = torch.as_tensor(
        positions,
        dtype=edge_query.dtype,
        device=edge_query.device,
    )
    if requested.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if edge_query.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=edge_query.device)
        for prefix in requested.tolist():
            yield int(prefix), empty
        return

    edge_order = torch.argsort(edge_query, stable=True)
    sorted_query = edge_query[edge_order]
    stops = torch.searchsorted(sorted_query, requested, right=True)
    start = 0
    for prefix, stop in zip(requested.tolist(), stops.tolist()):
        yield int(prefix), edge_order[start:stop]
        start = stop


def prefix_causal_attention_modes(
    sample,
    *,
    positions=None,
    config: SpectralConfig | None = None,
) -> PrefixCausalAttentionModes:
    """Return age-normalized RR coordinates with their causal sources.

    For channel ``c=(layer, head)``, response prefix ``t`` and response source
    node ``j <= t``::

        d_age[c,t,j] = sum_{u=j..t} A_c[u,j] / (t-j+1) - A_c[j,j]

    This is an artificial age-normalized triangular attention operator, not a
    standard graph Laplacian. The diagonal coordinates above are used directly;
    no eigendecomposition or eigenvector rotation is performed. ``top_k``
    entries with largest absolute magnitude are selected independently for every
    layer/head. Unlike :func:`prefix_causal_attention_spectrum`, this method
    retains the response-relative source index and lag of each selected mode.
    """
    config = SpectralConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    requested = _requested_positions(response_count, positions)
    shape = (len(requested), int(attention.num_channels), int(config.top_k))
    if response_count < 1 or len(requested) == 0:
        return PrefixCausalAttentionModes(
            positions=requested,
            values=np.zeros(shape, dtype=np.float32),
            source_index=np.full(shape, -1, dtype=np.int32),
            lag=np.full(shape, -1, dtype=np.int32),
        )

    device = attention.response_values.device
    diagonal = (
        attention.attention_diagonal[:, :, attention.response_idx :]
        .float()
        .reshape(attention.num_channels, response_count)
    )
    received = diagonal.clone()
    edge_channel, edge_query, edge_source, edge_weight = _retained_response_edges(
        sample, block_rows=config.block_rows
    )

    output = torch.zeros(shape, dtype=torch.float32, device=device)
    output_source = torch.full(shape, -1, dtype=torch.long, device=device)
    output_lag = torch.full(shape, -1, dtype=torch.long, device=device)
    for output_index, (prefix, new_edge) in enumerate(
        causal_prefix_edge_batches(edge_query, requested)
    ):
        if new_edge.numel():
            received.index_put_(
                (edge_channel[new_edge], edge_source[new_edge]),
                edge_weight[new_edge],
                accumulate=True,
            )

        active = prefix + 1
        source = torch.arange(active, dtype=torch.float32, device=device)
        denominator = (float(prefix) - source + 1.0).clamp_min(1.0)
        coordinates = (
            received[:, :active] / denominator.unsqueeze(0)
            - diagonal[:, :active]
        )
        keep = min(config.top_k, active)
        indices = torch.topk(
            coordinates.abs(), k=keep, dim=1, largest=True, sorted=True
        ).indices
        output[output_index, :, :keep] = torch.gather(coordinates, 1, indices)
        output_source[output_index, :, :keep] = indices
        output_lag[output_index, :, :keep] = int(prefix) - indices

    return PrefixCausalAttentionModes(
        positions=requested,
        values=output.detach().cpu().numpy().astype(np.float32, copy=False),
        source_index=output_source.detach().cpu().numpy().astype(np.int32, copy=False),
        lag=output_lag.detach().cpu().numpy().astype(np.int32, copy=False),
    )


def prefix_causal_attention_spectrum(
    sample,
    *,
    positions=None,
    config: SpectralConfig | None = None,
) -> np.ndarray:
    """Signed strongest-magnitude age-normalized RR coordinates per channel."""
    modes = prefix_causal_attention_modes(
        sample, positions=positions, config=config
    )
    return modes.values.reshape(len(modes.positions), -1)


def rr_spectral_dimension(num_layers: int, num_heads: int, top_k: int) -> int:
    return int(num_layers) * int(num_heads) * int(top_k)
