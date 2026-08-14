"""Label-blind spectral token representations from research attention views.

This module never opens raw attention files. It consumes only ResearchSample
objects supplied by ``research_dataset.open_research_dataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SpectralConfig:
    """Configuration for the first RP/RR spectral feasibility study."""

    heat_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    svd_bands: tuple[tuple[int, int], ...] = ((0, 1), (1, 4), (4, 8), (8, 16))
    block_rows: int = 4096
    epsilon: float = 1e-8

    def validate(self) -> None:
        if not self.heat_scales or any(
            (not math.isfinite(float(value)) or float(value) <= 0.0)
            for value in self.heat_scales
        ):
            raise ValueError("heat_scales must contain positive finite values")
        if not self.svd_bands:
            raise ValueError("svd_bands must not be empty")
        previous = 0
        for start, end in self.svd_bands:
            if start < 0 or end <= start or start < previous:
                raise ValueError("svd_bands must be ordered non-overlapping half-open ranges")
            previous = end
        if int(self.block_rows) < 1:
            raise ValueError("block_rows must be positive")
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive and finite")


def _normalized_laplacian_from_transport(
    transport: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Build a response-response normalized Laplacian from co-attention geometry.

    ``transport`` is ``[response_tokens, source_tokens]``. The Gram matrix
    captures similarity between response tokens' source-attention patterns.
    Its diagonal is removed so self similarity does not dominate the graph.
    """

    values = np.asarray(transport, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("transport must be a matrix")
    response_count = values.shape[0]
    if response_count == 0:
        return np.empty((0, 0), dtype=np.float64)
    if values.shape[1] == 0 or not np.any(values):
        return np.zeros((response_count, response_count), dtype=np.float64)

    gram = values @ values.T
    gram = 0.5 * (gram + gram.T)
    np.fill_diagonal(gram, 0.0)
    gram = np.maximum(gram, 0.0)
    degree = gram.sum(axis=1)
    positive = degree > epsilon
    inverse_sqrt = np.zeros_like(degree)
    inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = gram * inverse_sqrt[:, None] * inverse_sqrt[None, :]
    laplacian = -normalized
    diagonal = np.arange(response_count)
    laplacian[diagonal[positive], diagonal[positive]] = 1.0
    return 0.5 * (laplacian + laplacian.T)


def _relative_heat_kernel_signature(
    transport: np.ndarray,
    scales: tuple[float, ...],
    *,
    epsilon: float,
) -> np.ndarray:
    """Return sign-invariant node-local heat-kernel spectral signatures.

    Each scale is normalized by its within-graph mean and log transformed. This
    suppresses trivial response-length scale while retaining whether a token is
    locally spectrally more or less concentrated than the sample average.
    """

    laplacian = _normalized_laplacian_from_transport(
        transport,
        epsilon=epsilon,
    )
    if laplacian.shape[0] == 0:
        return np.empty((0, len(scales)), dtype=np.float32)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.clip(eigenvalues, 0.0, 2.0)
    squared_modes = np.square(eigenvectors)
    kernels = np.exp(
        -2.0
        * eigenvalues[:, None]
        * np.asarray(scales, dtype=np.float64)[None, :]
    )
    signature = squared_modes @ kernels
    mean = np.maximum(signature.mean(axis=0, keepdims=True), epsilon)
    signature = np.log(np.maximum(signature / mean, epsilon))
    return signature.astype(np.float32, copy=False)


def _svd_band_energy(
    transport: np.ndarray,
    bands: tuple[tuple[int, int], ...],
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return receiver and sender local energy over singular-value bands.

    Squared singular vectors make the representation invariant to individual
    singular-vector sign flips. Multiplication by node count expresses each
    token relative to the average leverage in that relation-specific operator.
    """

    values = np.asarray(transport, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("transport must be a matrix")
    receiver_count, sender_count = values.shape
    receiver = np.zeros((receiver_count, len(bands)), dtype=np.float64)
    sender = np.zeros((sender_count, len(bands)), dtype=np.float64)
    if receiver_count == 0 or sender_count == 0 or not np.any(values):
        return receiver.astype(np.float32), sender.astype(np.float32)

    left, singular_values, right_t = np.linalg.svd(values, full_matrices=False)
    spectral_energy = np.square(singular_values)
    total = max(float(spectral_energy.sum()), epsilon)
    right = right_t.T
    rank = len(singular_values)
    for column, (start, end) in enumerate(bands):
        start = min(int(start), rank)
        end = min(int(end), rank)
        if end <= start:
            continue
        weights = spectral_energy[start:end]
        receiver[:, column] = (
            np.square(left[:, start:end]) * weights[None, :]
        ).sum(axis=1) / total
        sender[:, column] = (
            np.square(right[:, start:end]) * weights[None, :]
        ).sum(axis=1) / total

    receiver *= max(receiver_count, 1)
    sender *= max(sender_count, 1)
    return receiver.astype(np.float32), sender.astype(np.float32)


def _scale_name(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def spectral_feature_names(config: SpectralConfig) -> tuple[str, ...]:
    names: list[str] = []
    for relation in ("rp", "rr"):
        names.extend(
            f"{relation}_log_hks_tau_{_scale_name(scale)}"
            for scale in config.heat_scales
        )
    for relation in ("rp_receiver", "rr_receiver", "rr_sender"):
        names.extend(
            f"{relation}_svd_energy_{start}_{end}"
            for start, end in config.svd_bands
        )
    return tuple(names)


def spectral_token_representation(sample, config: SpectralConfig | None = None):
    """Build one spectral vector per response token without labels.

    The data view is the channel-mean, cache-censored response attention exposed
    by ``ResearchSample.mean_response_attention``. Prompt and response-history
    columns are kept separate so their spectral geometries cannot cancel.

    Returns ``(features, feature_names)`` where features has shape ``[R,D]``.
    """

    config = SpectralConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    matrix = (
        sample.mean_response_attention(
            include_diagonal=False,
            block_rows=config.block_rows,
        )
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    prompt = matrix[:, : attention.response_idx]
    history = matrix[:, attention.response_idx :]

    rp_hks = _relative_heat_kernel_signature(
        prompt,
        config.heat_scales,
        epsilon=config.epsilon,
    )
    rr_hks = _relative_heat_kernel_signature(
        history,
        config.heat_scales,
        epsilon=config.epsilon,
    )
    rp_receiver, _rp_sender = _svd_band_energy(
        prompt,
        config.svd_bands,
        epsilon=config.epsilon,
    )
    rr_receiver, rr_sender = _svd_band_energy(
        history,
        config.svd_bands,
        epsilon=config.epsilon,
    )
    features = np.concatenate(
        (rp_hks, rr_hks, rp_receiver, rr_receiver, rr_sender),
        axis=1,
    ).astype(np.float32, copy=False)
    names = spectral_feature_names(config)
    if features.shape != (attention.num_response_tokens, len(names)):
        raise RuntimeError("spectral representation shape does not match feature names")
    if not np.isfinite(features).all():
        raise FloatingPointError("spectral representation contains non-finite values")
    return features, names
