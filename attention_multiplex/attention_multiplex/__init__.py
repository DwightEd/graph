"""Attention-only dynamic multiplex graph representations."""

from .representation import (
    REPRESENTATION_SCHEMA,
    MultiplexConfig,
    MultiplexRepresentation,
    MultiplexUnfolding,
    SpectralRoles,
    build_multiplex_unfolding,
    joint_spectral_roles,
    represent_attention_multiplex,
)

__all__ = [
    "REPRESENTATION_SCHEMA",
    "MultiplexConfig",
    "MultiplexRepresentation",
    "MultiplexUnfolding",
    "SpectralRoles",
    "build_multiplex_unfolding",
    "joint_spectral_roles",
    "represent_attention_multiplex",
]
