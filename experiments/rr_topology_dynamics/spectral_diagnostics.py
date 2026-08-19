"""Optional spectral residual diagnostics for routing-attractor features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.spectral_feasibility.experiment import load_spectral_reference
from experiments.spectral_feasibility.representations import (
    SpectralConfig,
    prefix_causal_attention_modes,
    response_position_bin,
)


@dataclass(frozen=True)
class SpectralDiagnostics:
    residual_energy: np.ndarray
    layer_residual_energy: np.ndarray
    rank_residual_energy: np.ndarray
    embedding: np.ndarray


def load_rr_reference(path):
    """Load the frozen RR reference required only by spectral diagnostics."""

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


class SpectralDiagnosticsExtractor:
    """Explain distance from a frozen RR spectral subspace without scoring it."""

    def __init__(
        self,
        reference,
        *,
        top_k: int,
        block_rows: int,
    ) -> None:
        if int(top_k) != int(reference["top_k"]):
            raise ValueError("spectral_top_k differs from frozen RR reference")
        self.reference = reference
        self.config = SpectralConfig(top_k=int(top_k), block_rows=int(block_rows))

    def extract(self, sample) -> SpectralDiagnostics:
        attention = sample.attention()
        response_count = int(attention.num_response_tokens)
        layers = int(attention.num_layers)
        heads = int(attention.num_heads)
        if layers != int(self.reference["num_layers"]) or heads != int(
            self.reference["num_heads"]
        ):
            raise ValueError("sample attention geometry differs from RR spectral reference")

        modes = prefix_causal_attention_modes(sample, config=self.config)
        values = modes.values.reshape(response_count, -1)
        position_bins = int(self.reference["position_bins"])
        bins = np.asarray(
            [
                response_position_bin(token, response_count, position_bins)
                for token in range(response_count)
            ],
            dtype=np.int16,
        )
        standardized = (
            values - self.reference["rr_center"][bins]
        ) / self.reference["rr_scale"][bins]
        centered = standardized - self.reference["rr_pca_mean"]
        scores = centered @ self.reference["rr_pca_components"].T
        embedding = scores / self.reference["rr_pca_whiten_scale"]
        reconstructed = (
            scores @ self.reference["rr_pca_components"]
            + self.reference["rr_pca_mean"]
        )
        coordinate_energy = np.square(standardized - reconstructed).reshape(
            response_count, layers * heads, self.config.top_k
        )
        channel_energy = coordinate_energy.mean(axis=2)
        return SpectralDiagnostics(
            residual_energy=coordinate_energy.mean(axis=(1, 2)).astype(np.float32),
            layer_residual_energy=channel_energy.reshape(
                response_count, layers, heads
            ).mean(axis=2).astype(np.float32),
            rank_residual_energy=coordinate_energy.mean(axis=1).astype(np.float32),
            embedding=embedding.astype(np.float32),
        )
