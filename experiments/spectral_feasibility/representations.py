"""Causal response-history spectral states for sparse attention graphs.

The active detector in this package is deliberately RR-only.  For every
layer/head channel and response prefix we treat retained response-to-response
attention as a causal directed adjacency operator.  The corresponding
Laplacian is triangular, so its diagonal is its spectrum.  Strongest-magnitude
spectral coordinates are kept per channel without averaging layer/head axes.

Prompt transport remains available as a separate diagnostic utility, but it is
not concatenated into the active RR subspace detector.  Missing CSR entries are
cache-censored (<= ``attention_floor``); they are never reconstructed as exact
zeros.
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
        if not 0.0 < float(self.channel_tail_fraction) <= 1.0:
            raise ValueError("channel_tail_fraction must be in (0, 1]")
        if self.attribution_topk < 1:
            raise ValueError("attribution_topk must be positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")


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


def prefix_laplacian_spectrum(
    sample,
    *,
    positions=None,
    config: SpectralConfig | None = None,
) -> np.ndarray:
    """Signed strongest-magnitude causal RR-Laplacian spectrum per channel.

    For channel ``c=(layer, head)``, response prefix ``t`` and response source
    node ``j <= t``::

        d[c,t,j]      = sum_{u=j..t} A_c[u,j] / (t-j+1)
        lambda[c,t,j] = d[c,t,j] - A_c[j,j]

    The causal Laplacian is triangular, therefore its diagonal is its spectrum.
    ``top_k`` entries with largest absolute magnitude are retained independently
    for every layer/head and their original signs are preserved.
    """
    config = SpectralConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    if response_count < 1:
        return np.empty(
            (0, attention.num_channels * config.top_k), dtype=np.float32
        )

    if positions is None:
        requested = np.arange(response_count, dtype=np.int64)
    else:
        requested = np.asarray(list(positions), dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        if requested.size == 0:
            return np.empty(
                (0, attention.num_channels * config.top_k), dtype=np.float32
            )
        if np.any(requested < 0) or np.any(requested >= response_count):
            raise IndexError("requested response position is outside the response")
        requested = np.unique(requested)

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

    output = torch.zeros(
        (requested.size, attention.num_channels, config.top_k),
        dtype=torch.float32,
        device=device,
    )
    previous_prefix = -1
    for output_index, prefix in enumerate(requested.tolist()):
        if edge_query.numel():
            new_edge = (edge_query > previous_prefix) & (edge_query <= prefix)
            if bool(new_edge.any()):
                received.index_put_(
                    (edge_channel[new_edge], edge_source[new_edge]),
                    edge_weight[new_edge],
                    accumulate=True,
                )

        active = prefix + 1
        source = torch.arange(active, dtype=torch.float32, device=device)
        denominator = (float(prefix) - source + 1.0).clamp_min(1.0)
        eigenvalues = (
            received[:, :active] / denominator.unsqueeze(0)
            - diagonal[:, :active]
        )
        keep = min(config.top_k, active)
        indices = torch.topk(
            eigenvalues.abs(), k=keep, dim=1, largest=True, sorted=True
        ).indices
        output[output_index, :, :keep] = torch.gather(eigenvalues, 1, indices)
        previous_prefix = prefix

    return (
        output.reshape(requested.size, attention.num_channels * config.top_k)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def prompt_transport_profile(
    sample,
    *,
    positions=None,
    prompt_bins: int = 8,
    block_rows: int = 8192,
) -> np.ndarray:
    """Return an RP routing profile for post-hoc diagnostics only.

    Prompt query rows are unavailable, so this does not fabricate a prompt
    Laplacian.  Retained response-to-prompt weights are accumulated into fixed
    relative prompt-position bins independently per layer/head.  The active RR
    detector never consumes this profile.
    """
    prompt_bins = int(prompt_bins)
    block_rows = int(block_rows)
    if prompt_bins < 2:
        raise ValueError("prompt_bins must be at least two")
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    if prompt_count < 1:
        raise ValueError("response_idx must leave at least one prompt token")
    device = attention.response_values.device
    profile = torch.zeros(
        (response_count, attention.num_channels, prompt_bins),
        dtype=torch.float32,
        device=device,
    )

    for block in sample.iter_sparse_attention_blocks(block_rows=block_rows):
        prompt = block.source < attention.response_idx
        if not bool(prompt.any()):
            continue
        query = block.query[prompt].long()
        channel = (
            block.layer[prompt] * attention.num_heads + block.head[prompt]
        ).long()
        source = block.source[prompt].long()
        weight = block.weight[prompt].float()
        bucket = torch.div(
            source * prompt_bins,
            prompt_count,
            rounding_mode="floor",
        ).clamp_(0, prompt_bins - 1)
        profile.index_put_((query, channel, bucket), weight, accumulate=True)

    if positions is None:
        selected = profile
    else:
        requested = np.asarray(list(positions), dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        if requested.size == 0:
            return np.empty(
                (0, attention.num_channels * prompt_bins), dtype=np.float32
            )
        if np.any(requested < 0) or np.any(requested >= response_count):
            raise IndexError("requested response position is outside the response")
        requested = np.unique(requested)
        selected = profile[
            torch.as_tensor(requested, dtype=torch.long, device=device)
        ]

    return (
        selected.reshape(selected.shape[0], -1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def rr_spectral_dimension(num_layers: int, num_heads: int, top_k: int) -> int:
    return int(num_layers) * int(num_heads) * int(top_k)
