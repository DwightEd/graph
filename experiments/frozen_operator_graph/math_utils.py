"""Numerically stable deterministic feature primitives."""

from __future__ import annotations

import torch


def safe_norm(value: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return value.float().pow(2).sum(dim=dim).clamp_min(eps).sqrt()


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    numerator = (a.float() * b.float()).sum(dim=-1)
    denominator = safe_norm(a, eps=eps) * safe_norm(b, eps=eps)
    return numerator / denominator.clamp_min(eps)


def entropy_from_mass(mass: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mass = mass.float().clamp_min(0)
    total = mass.sum(dim=-1, keepdim=True)
    probability = torch.where(total > eps, mass / total.clamp_min(eps), torch.zeros_like(mass))
    log_probability = torch.where(
        probability > 0,
        probability.clamp_min(eps).log(),
        torch.zeros_like(probability),
    )
    return -(probability * log_probability).sum(dim=-1)


def effective_number(mass: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mass = mass.float().clamp_min(0)
    total = mass.sum(dim=-1)
    effective = entropy_from_mass(mass, eps=eps).exp()
    return torch.where(total > eps, effective, torch.zeros_like(effective))


def top1_share(mass: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mass = mass.float().clamp_min(0)
    total = mass.sum(dim=-1)
    maximum = mass.max(dim=-1).values if mass.shape[-1] else total
    return torch.where(total > eps, maximum / total.clamp_min(eps), torch.zeros_like(total))


def weighted_mean_and_variance(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = weights.float().clamp_min(0)
    values = values.float()
    total = weights.sum(dim=-1)
    mean = (weights * values).sum(dim=-1) / total.clamp_min(eps)
    variance = (
        weights * (values - mean.unsqueeze(-1)).pow(2)
    ).sum(dim=-1) / total.clamp_min(eps)
    zero = total <= eps
    return torch.where(zero, torch.zeros_like(mean), mean), torch.where(
        zero, torch.zeros_like(variance), variance
    )


def stable_descending_indices(values: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Sort by descending value and then ascending source index."""

    values = values.float()
    source = source.long()
    by_source = torch.argsort(source, stable=True)
    by_value = torch.argsort(values[by_source], descending=True, stable=True)
    return by_source[by_value]


def minimum_prefix(
    score: torch.Tensor,
    source: torch.Tensor,
    retention: float,
    minimum: int,
) -> torch.Tensor:
    """Return deterministic source indices meeting a cumulative retention target."""

    if score.ndim != 1 or source.shape != score.shape:
        raise ValueError("score/source must be aligned vectors")
    if not len(score):
        return source.new_empty(0)
    order = stable_descending_indices(score, source)
    total = score.float().clamp_min(0).sum()
    if float(total.item()) <= 0.0:
        count = min(max(int(minimum), 0), len(order))
        return order[:count]
    threshold = total * float(retention)
    cumulative = score[order].float().clamp_min(0).cumsum(dim=0)
    count = int(torch.searchsorted(cumulative, threshold, right=False).item()) + 1
    count = min(max(count, int(minimum)), len(order))
    return order[:count]
