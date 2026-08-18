"""Self-supervised objectives for causal source-set flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    route_element: torch.Tensor
    memory_element: torch.Tensor
    head: torch.Tensor
    layer: torch.Tensor
    temporal: torch.Tensor
    variance: torch.Tensor


def variance_floor_loss(
    values: torch.Tensor,
    *,
    target_std: float = 1.0,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("variance regularization expects [token, feature]")
    if len(values) < 2:
        return values.sum() * 0.0
    std = torch.sqrt(values.var(dim=0, unbiased=False) + float(epsilon))
    return torch.relu(float(target_std) - std).mean()
