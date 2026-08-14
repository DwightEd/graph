"""Causal channel-preserving spectral attention experiments."""

from .representations import (
    SpectralConfig,
    causal_spectral_state,
    prefix_laplacian_spectrum,
    prompt_transport_sketch,
    spectral_volume,
)

__all__ = [
    "SpectralConfig",
    "causal_spectral_state",
    "prefix_laplacian_spectrum",
    "prompt_transport_sketch",
    "spectral_volume",
]
