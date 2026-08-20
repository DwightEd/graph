"""Majorization and Rényi--Hill statistics on exact-source routes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


HILL_ORDERS = (0.5, 1.0, 2.0, 4.0, math.inf)


@dataclass(frozen=True)
class MajorizationEvidence:
    """Signed concentration evidence and its two auditable components."""

    evidence: torch.Tensor
    area: torch.Tensor
    violation: torch.Tensor


def _probability(values: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    nonnegative = values.clamp_min(0.0)
    total = nonnegative.sum(dim=-1, keepdim=True)
    valid = total.squeeze(-1) > epsilon
    probability = nonnegative / total.clamp_min(epsilon)
    return probability, valid


def hill_diversity_spectrum(
    values: torch.Tensor,
    *,
    orders: tuple[float, ...] = HILL_ORDERS,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return Hill effective-source counts for several Rényi orders.

    Order 1 is the exponential Shannon entropy; order infinity is the inverse
    strongest-source share. Zero-mass rows return zeros.
    """

    probability, valid = _probability(values, epsilon)
    diversity = []
    for order in orders:
        if order == 1.0:
            entropy = -(
                probability * probability.clamp_min(epsilon).log()
            ).sum(dim=-1)
            value = entropy.exp()
        elif math.isinf(order):
            value = probability.amax(dim=-1).clamp_min(epsilon).reciprocal()
        else:
            power_sum = probability.pow(order).sum(dim=-1)
            value = power_sum.clamp_min(epsilon).pow(1.0 / (1.0 - order))
        diversity.append(torch.where(valid, value, torch.zeros_like(value)))
    return torch.stack(diversity, dim=-1)


def majorization_evidence(
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    tolerance: float = 1e-6,
    epsilon: float = 1e-12,
) -> MajorizationEvidence:
    """Compare top-cumulative curves of two exact-source distributions.

    Positive evidence means ``current`` majorizes ``reference``: its largest
    sources carry at least as much cumulative mass at every non-trivial rank.
    A crossing curve is reported as negative evidence instead of being called
    concentration.
    """

    if current.shape != reference.shape:
        raise ValueError("current and reference distributions must have the same shape")
    if current.shape[-1] < 1:
        raise ValueError("source dimension must be non-empty")

    current_probability, current_valid = _probability(current, epsilon)
    reference_probability, reference_valid = _probability(reference, epsilon)
    valid = current_valid & reference_valid

    current_curve = current_probability.sort(dim=-1, descending=True).values.cumsum(-1)
    reference_curve = reference_probability.sort(dim=-1, descending=True).values.cumsum(-1)
    difference = current_curve[..., :-1] - reference_curve[..., :-1]
    if difference.shape[-1] == 0:
        area = torch.zeros_like(current_valid, dtype=current.dtype)
        violation = torch.zeros_like(area)
    else:
        area = difference.mean(dim=-1)
        violation = (-difference).clamp_min(0.0).amax(dim=-1)

    zero = torch.zeros_like(area)
    area = torch.where(valid, area, zero)
    violation = torch.where(valid, violation, zero)
    evidence = torch.where(violation <= tolerance, area, -violation)
    evidence = torch.where(valid, evidence, zero)
    return MajorizationEvidence(evidence=evidence, area=area, violation=violation)
