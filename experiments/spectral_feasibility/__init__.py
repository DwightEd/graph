"""RR causal spectral-subspace anomaly detection."""

from .representations import (
    SpectralConfig,
    prefix_laplacian_spectrum,
    prompt_transport_profile,
    rr_spectral_dimension,
)

__all__ = [
    "SpectralConfig",
    "prefix_laplacian_spectrum",
    "prompt_transport_profile",
    "rr_spectral_dimension",
]
