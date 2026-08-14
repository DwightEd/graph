"""Channel-preserving causal Laplacian spectra for attention graphs.

For a response-prefix ending at response-relative position ``t`` and channel
``c=(layer, head)``, let ``A_c[:t+1, :t+1]`` be the cache-censored
response-to-response attention submatrix, including the stored diagonal.
Following the directed-Laplacian construction in LapEigvals, define

    d[c,t,j] = sum_{u=j..t} A_c[u,j] / (t-j+1)
    lambda[c,t,j] = d[c,t,j] - A_c[j,j].

Because causal attention and ``L=D-A`` are lower triangular, these diagonal
values are the Laplacian eigenvalues.  We keep the largest k values separately
for every layer/head at every causal response prefix.  There is deliberately no
channel mean, ``AA^T`` symmetrization, label use, or supervised feature
selection here.

The formal cache censors off-diagonal attention <= ``attention_floor``.  The
result is therefore the spectrum of the cache-censored response graph, not a
claim that the unavailable full-precision attention matrix was recovered.
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
    reference_per_sample: int = 4
    neighbors: int = 10
    spectral_window: int = 8
    logdet_alpha: float = 1e-3
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
    """Choose deterministic, approximately uniform reference tokens."""
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
    """Return top-k causal Laplacian eigenvalues for every layer/head.

    Output shape is ``[len(positions), L*H*k]`` in layer-major, head-major,
    descending-eigenvalue order.  A prefix shorter than k nodes is zero-padded.

    We intentionally restrict the Laplacian to response nodes.  For a response
    node all queries that can attend to it from itself onward are response
    queries and are present in the cache.  Prompt-query rows are not present,
    so pretending to reconstruct a full prompt+response Laplacian would be
    unjustified.
    """
    config = SpectralConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    if response_count < 1:
        return np.empty((0, attention.num_channels * config.top_k), dtype=np.float32)

    if positions is None:
        requested = np.arange(response_count, dtype=np.int64)
    else:
        requested = np.asarray(list(positions), dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("positions must be one-dimensional")
        if requested.size == 0:
            return np.empty((0, attention.num_channels * config.top_k), dtype=np.float32)
        if np.any(requested < 0) or np.any(requested >= response_count):
            raise IndexError("requested response position is outside the response")
        requested = np.unique(requested)

    device = attention.response_values.device
    diagonal = (
        attention.attention_diagonal[:, :, attention.response_idx :]
        .float()
        .reshape(attention.num_channels, response_count)
    )
    # Each node starts with its exact stored self-attention.  As the response
    # prefix grows, retained future RR attention to that source is accumulated.
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


def spectral_volume(
    embeddings: np.ndarray,
    *,
    window: int = 8,
    alpha: float = 1e-3,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """EigenScore-inspired causal LogDet of recent graph spectral states.

    This is not the original EigenScore: EigenScore uses covariance across
    sentence embeddings from multiple sampled responses.  Here consecutive
    label-free graph spectral embeddings form the columns.  LogDet therefore
    measures the local volume occupied by the causal attention-graph trajectory.
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
        z = current.T
        z = z - z.mean(axis=0, keepdims=True)
        gram = (z.T @ z) / max(dimensions, 1)
        gram.flat[:: len(gram) + 1] += alpha
        sign, logdet = np.linalg.slogdet(gram)
        if sign <= 0 or not np.isfinite(logdet):
            eigenvalues = np.linalg.eigvalsh(gram)
            logdet = np.log(np.maximum(eigenvalues, epsilon)).sum()
        result[token] = logdet / float(len(current))
    return result.astype(np.float32, copy=False)


def spectral_state_dimension(num_layers: int, num_heads: int, top_k: int) -> int:
    return int(num_layers) * int(num_heads) * int(top_k)
