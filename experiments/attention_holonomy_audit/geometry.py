"""Compositional geometry for per-event attention head profiles."""

from __future__ import annotations

import numpy as np
import torch


def event_profiles(
    head_value: torch.Tensor,
    head_observed: torch.Tensor,
    *,
    attention_floor: float,
    censored_fill_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive head profiles and centered-log-ratio coordinates.

    An event exists when at least one head retained the token pair. Missing heads
    are censored below the attention floor rather than observed zeros, so they
    receive one fixed sub-floor fill before normalization. The observation mask
    remains available to the nuisance model.
    """

    fill = float(attention_floor) * float(censored_fill_ratio)
    dense = torch.where(
        head_observed,
        head_value,
        torch.full_like(head_value, fill),
    ).clamp_min(torch.finfo(head_value.dtype).tiny)
    profile = dense / dense.sum(dim=-1, keepdim=True)
    log_profile = profile.log()
    clr = log_profile - log_profile.mean(dim=-1, keepdim=True)
    return profile, clr


def clr_to_profile(values: torch.Tensor | np.ndarray):
    """Map centered-log-ratio coordinates back to the probability simplex."""

    if isinstance(values, torch.Tensor):
        return torch.softmax(values, dim=-1)
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    probability = np.exp(shifted)
    return probability / probability.sum(axis=-1, keepdims=True)


def hellinger_squared(
    left: torch.Tensor | np.ndarray,
    right: torch.Tensor | np.ndarray,
):
    """Squared Hellinger distance on simplex-valued rows."""

    if isinstance(left, torch.Tensor):
        return 0.5 * ((left.clamp_min(0).sqrt() - right.clamp_min(0).sqrt()) ** 2).sum(
            dim=-1
        )
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return 0.5 * np.square(np.sqrt(np.clip(left, 0, None)) - np.sqrt(np.clip(right, 0, None))).sum(
        axis=-1
    )


def weighted_profile_mean(
    values: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    total = weight.sum().clamp_min(torch.finfo(values.dtype).tiny)
    result = (values * weight[:, None]).sum(dim=0) / total
    return result / result.sum().clamp_min(torch.finfo(values.dtype).tiny)
