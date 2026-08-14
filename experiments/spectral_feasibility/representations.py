"""Causal channel-preserving spectral states for sparse attention graphs.

The representation is built only from information that is exactly available in
``ResearchSample`` response-query caches:

* RR (response-history) topology: for every layer/head and causal response
  prefix, use the triangular directed Laplacian spectrum.  We keep the signed
  eigenvalues with largest absolute magnitude so strong positive and negative
  departures are both retained.
* RP (response-to-prompt) transport: prompt query rows are unavailable, so a
  prompt-node Laplacian would be fabricated.  Instead retained prompt attention
  is accumulated into fixed relative prompt-position bins independently for
  every layer/head.  The bin sum is the exact retained prompt mass.

Layer/head channels are never averaged before the raw state is formed.  Missing
CSR entries remain cache-censored (<= ``attention_floor``), not reconstructed
zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SpectralConfig:
    top_k: int = 5
    prompt_bins: int = 8
    block_rows: int = 8192
    position_bins: int = 4
    pca_dim: int = 32
    reference_per_sample: int = 6
    trim_fraction: float = 0.90
    neighbors: int = 10
    spectral_window: int = 8
    dynamic_lags: int = 3
    dynamic_ridge: float = 1e-2
    logdet_alpha: float = 1e-3
    attribution_topk: int = 8
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.prompt_bins < 2:
            raise ValueError("prompt_bins must be at least two")
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
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")
        if self.spectral_window < 2:
            raise ValueError("spectral_window must be at least two")
        if self.dynamic_lags < 1:
            raise ValueError("dynamic_lags must be positive")
        if not np.isfinite(self.dynamic_ridge) or self.dynamic_ridge <= 0:
            raise ValueError("dynamic_ridge must be positive and finite")
        if not np.isfinite(self.logdet_alpha) or self.logdet_alpha <= 0:
            raise ValueError("logdet_alpha must be positive and finite")
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
    """Signed strongest-magnitude causal RR-Laplacian eigenvalues per channel.

    For channel ``c=(layer, head)``, response prefix ``t`` and response source
    node ``j <= t``::

        d[c,t,j]      = sum_{u=j..t} A_c[u,j] / (t-j+1)
        lambda[c,t,j] = d[c,t,j] - A_c[j,j]

    The causal Laplacian is triangular, therefore its diagonal is its spectrum.
    We keep the ``top_k`` entries with largest absolute magnitude and preserve
    their signs.  This avoids discarding strong negative spectral departures.
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
        strongest = torch.gather(eigenvalues, 1, indices)
        output[output_index, :, :keep] = strongest
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
    config: SpectralConfig | None = None,
) -> np.ndarray:
    """Channel-preserving RP routing profiles in fixed relative prompt bins.

    Each retained prompt source ``p`` is assigned to a deterministic relative
    prompt-position bin.  For every response token and layer/head channel,
    weights are accumulated without normalization.  Therefore the sum across
    bins is exactly the retained prompt mass while the bin pattern preserves
    coarse source location without CountSketch collisions.
    """
    config = SpectralConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    if prompt_count < 1:
        raise ValueError("response_idx must leave at least one prompt token")
    device = attention.response_values.device
    profile = torch.zeros(
        (response_count, attention.num_channels, config.prompt_bins),
        dtype=torch.float32,
        device=device,
    )

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
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
            source * config.prompt_bins,
            prompt_count,
            rounding_mode="floor",
        ).clamp_(0, config.prompt_bins - 1)
        profile.index_put_((query, channel, bucket), weight, accumulate=True)

    if positions is None:
        selected = profile
    else:
        requested = np.asarray(list(positions), dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        if requested.size == 0:
            return np.empty(
                (0, attention.num_channels * config.prompt_bins), dtype=np.float32
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


def prompt_channel_volume(
    prompt_profile: np.ndarray,
    *,
    num_channels: int,
    prompt_bins: int,
    alpha: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """EigenScore-inspired LogDet of RP routing diversity across channels."""
    values = np.asarray(prompt_profile, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != num_channels * prompt_bins:
        raise ValueError("prompt profile shape does not match channel geometry")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    reshaped = values.reshape(len(values), num_channels, prompt_bins)
    centered = reshaped - reshaped.mean(axis=1, keepdims=True)
    covariance = np.einsum(
        "tcm,tcn->tmn", centered, centered, optimize=True
    ) / max(num_channels, 1)
    diagonal = np.arange(prompt_bins)
    covariance[:, diagonal, diagonal] += alpha
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
        eigenvalues = np.linalg.eigvalsh(covariance)
        logdet = np.log(np.maximum(eigenvalues, epsilon)).sum(axis=1)
    return (logdet / float(prompt_bins)).astype(np.float32, copy=False)


def causal_spectral_state(
    sample,
    *,
    positions=None,
    config: SpectralConfig | None = None,
):
    """Return raw causal RR-spectrum + RP-transport state and RP LogDet."""
    config = SpectralConfig() if config is None else config
    config.validate()
    rr = prefix_laplacian_spectrum(sample, positions=positions, config=config)
    rp = prompt_transport_profile(sample, positions=positions, config=config)
    if len(rr) != len(rp):
        raise RuntimeError("RR spectrum and RP transport views are misaligned")
    state = np.concatenate((rr, rp), axis=1).astype(np.float32, copy=False)
    attention = sample.attention()
    volume = prompt_channel_volume(
        rp,
        num_channels=int(attention.num_channels),
        prompt_bins=config.prompt_bins,
        alpha=config.logdet_alpha,
        epsilon=config.epsilon,
    )
    return state, volume


def spectral_volume(
    embeddings: np.ndarray,
    *,
    window: int = 8,
    alpha: float = 1e-3,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Causal LogDet volume of the recent learned spectral trajectory."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("embeddings must have shape [tokens, dimensions]")
    if window < 2:
        raise ValueError("window must be at least two")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    tokens, dimensions = values.shape
    result = np.zeros(tokens, dtype=np.float64)
    for token in range(tokens):
        start = max(0, token - window + 1)
        current = values[start : token + 1]
        if len(current) < 2 or dimensions == 0:
            result[token] = np.log(alpha)
            continue
        centered = current - current.mean(axis=0, keepdims=True)
        gram = (centered @ centered.T) / max(dimensions, 1)
        gram.flat[:: len(gram) + 1] += alpha
        sign, logdet = np.linalg.slogdet(gram)
        if sign <= 0 or not np.isfinite(logdet):
            eigenvalues = np.linalg.eigvalsh(gram)
            logdet = np.log(np.maximum(eigenvalues, epsilon)).sum()
        result[token] = logdet / float(len(current))
    return result.astype(np.float32, copy=False)


def spectral_state_dimension(
    num_layers: int,
    num_heads: int,
    top_k: int,
    prompt_bins: int,
) -> int:
    channels = int(num_layers) * int(num_heads)
    return channels * (int(top_k) + int(prompt_bins))
