"""Causal channel-preserving spectral states for sparse attention graphs.

The representation uses two spectral objects that are exactly supportable by
the response-query cache:

1. Response-history (RR) topology: for every layer/head and response prefix,
   construct the directed response-subgraph Laplacian diagonal following the
   LapEigvals degree convention. Causality makes the Laplacian triangular, so
   its diagonal is its spectrum. Keep top-k eigenvalues per channel.

2. Prompt-routing (RP) transport: prompt query rows are not present in the
   cache, so we do not fabricate a prompt-node Laplacian. Instead each
   response-query prompt-routing row is mapped by a deterministic CountSketch
   that preserves an exact DC coordinate (retained prompt mass) and signed
   source geometry. Channel-wise sketches are kept, and their covariance
   LogDet provides an EigenScore-inspired spectral-volume diagnostic.

No hallucination labels are used and layer/head channels are never averaged
before the raw spectral state is formed. Values below ``attention_floor`` remain
censored; this is the spectrum of the retained cache, not a reconstruction of
unknown full-precision attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SpectralConfig:
    top_k: int = 5
    prompt_sketch_dim: int = 4
    prompt_sketch_seed: int = 20260814
    block_rows: int = 8192
    position_bins: int = 4
    pca_dim: int = 32
    reference_per_sample: int = 4
    neighbors: int = 10
    spectral_window: int = 8
    logdet_alpha: float = 1e-3
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.prompt_sketch_dim < 2:
            raise ValueError("prompt_sketch_dim must be at least two")
        if self.block_rows < 1:
            raise ValueError("block_rows must be positive")
        if self.position_bins < 1:
            raise ValueError("position_bins must be positive")
        if self.pca_dim < 1:
            raise ValueError("pca_dim must be positive")
        if self.reference_per_sample < 1:
            raise ValueError("reference_per_sample must be positive")
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")
        if self.spectral_window < 2:
            raise ValueError("spectral_window must be at least two")
        if not np.isfinite(self.logdet_alpha) or self.logdet_alpha <= 0:
            raise ValueError("logdet_alpha must be positive and finite")
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
    """Choose deterministic, approximately uniform train reference tokens."""
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
    """Collect retained RR edges exclusively through the ResearchSample API."""
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
    """Top-k causal RR-Laplacian eigenvalues for every layer/head.

    For prefix t and response source j<=t,

        d[c,t,j] = sum_{u=j..t} A_c[u,j] / (t-j+1)
        lambda[c,t,j] = d[c,t,j] - A_c[j,j].

    This matches the no-vertical-edge degree convention used by LapEigvals,
    restricted to the response subgraph that the cache can reconstruct without
    inventing prompt-query rows.
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
        largest = torch.topk(
            eigenvalues, k=keep, dim=1, largest=True, sorted=True
        ).values
        output[output_index, :, :keep] = largest
        previous_prefix = prefix

    return (
        output.reshape(requested.size, attention.num_channels * config.top_k)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _prompt_hash(source: torch.Tensor, config: SpectralConfig):
    """Deterministic source hash for the signed CountSketch coordinates."""
    modulus = 2_147_483_647
    seed = int(config.prompt_sketch_seed) % modulus
    hashed = torch.remainder(source.long() * 1_103_515_245 + seed, modulus)
    bucket = 1 + torch.remainder(hashed, config.prompt_sketch_dim - 1)
    sign_hash = torch.remainder(
        source.long() * 214_013 + seed * 2_531_011, modulus
    )
    sign = torch.where(
        torch.remainder(sign_hash, 2) == 0,
        torch.ones_like(sign_hash, dtype=torch.float32),
        -torch.ones_like(sign_hash, dtype=torch.float32),
    )
    return bucket.long(), sign


def prompt_transport_sketch(
    sample,
    *,
    positions=None,
    config: SpectralConfig | None = None,
) -> np.ndarray:
    """Return channel-preserving prompt-routing sketches [tokens, C*m].

    Coordinate zero is exact retained prompt mass. Remaining coordinates are a
    signed CountSketch over prompt source positions, preserving source geometry
    without materializing a dense [L,H,R,P] tensor.
    """
    config = SpectralConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    device = attention.response_values.device
    sketch = torch.zeros(
        (response_count, attention.num_channels, config.prompt_sketch_dim),
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
        sketch.index_put_(
            (query, channel, torch.zeros_like(query)),
            weight,
            accumulate=True,
        )
        bucket, sign = _prompt_hash(source, config)
        sketch.index_put_(
            (query, channel, bucket),
            weight * sign.to(device=weight.device),
            accumulate=True,
        )

    if positions is None:
        selected = sketch
    else:
        requested = np.asarray(list(positions), dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        if requested.size == 0:
            return np.empty(
                (0, attention.num_channels * config.prompt_sketch_dim),
                dtype=np.float32,
            )
        if np.any(requested < 0) or np.any(requested >= response_count):
            raise IndexError("requested response position is outside the response")
        requested = np.unique(requested)
        selected = sketch[
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
    prompt_sketch: np.ndarray,
    *,
    num_channels: int,
    sketch_dim: int,
    alpha: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """EigenScore-inspired LogDet of prompt-routing diversity across channels."""
    values = np.asarray(prompt_sketch, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != num_channels * sketch_dim:
        raise ValueError("prompt sketch shape does not match channel geometry")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    reshaped = values.reshape(len(values), num_channels, sketch_dim)
    centered = reshaped - reshaped.mean(axis=1, keepdims=True)
    covariance = np.einsum(
        "tcm,tcn->tmn", centered, centered, optimize=True
    ) / max(num_channels, 1)
    diagonal = np.arange(sketch_dim)
    covariance[:, diagonal, diagonal] += alpha
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
        eigenvalues = np.linalg.eigvalsh(covariance)
        logdet = np.log(np.maximum(eigenvalues, epsilon)).sum(axis=1)
    return (logdet / float(sketch_dim)).astype(np.float32, copy=False)


def causal_spectral_state(
    sample,
    *,
    positions=None,
    config: SpectralConfig | None = None,
):
    """Return raw causal spectral state plus prompt channel-volume diagnostic."""
    config = SpectralConfig() if config is None else config
    config.validate()
    rr = prefix_laplacian_spectrum(sample, positions=positions, config=config)
    rp = prompt_transport_sketch(sample, positions=positions, config=config)
    if len(rr) != len(rp):
        raise RuntimeError("RR spectrum and RP transport views are misaligned")
    state = np.concatenate((rr, rp), axis=1).astype(np.float32, copy=False)
    attention = sample.attention()
    volume = prompt_channel_volume(
        rp,
        num_channels=int(attention.num_channels),
        sketch_dim=config.prompt_sketch_dim,
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
    """EigenScore-inspired causal LogDet of recent learned spectral states.

    This is not the original EigenScore. EigenScore applies LogDet to the
    covariance of sentence embeddings from multiple sampled responses. Here
    consecutive graph-spectral embeddings form a causal trajectory window.
    """
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
    prompt_sketch_dim: int,
) -> int:
    channels = int(num_layers) * int(num_heads)
    return channels * (int(top_k) + int(prompt_sketch_dim))
