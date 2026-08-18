"""Robust label-free losses for Mechanism-Guided CASF."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    clean_energy: torch.Tensor
    corrupt_energy: torch.Tensor
    ranking: torch.Tensor
    type_token: torch.Tensor
    type_channel: torch.Tensor
    clean_recovery: torch.Tensor
    context_recovery: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor


def robust_clean_energy_loss(logits: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    """Trim the largest clean-target losses to tolerate unlabeled contamination."""

    values = F.binary_cross_entropy_with_logits(
        logits.float(), torch.zeros_like(logits, dtype=torch.float32), reduction="none"
    ).reshape(-1)
    if not len(values):
        return logits.sum() * 0.0
    keep = max(1, min(len(values), round(len(values) * float(keep_fraction))))
    return torch.topk(values, k=keep, largest=False, sorted=False).values.mean()


def corrupted_energy_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = logits[mask]
    if not len(selected):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        selected.float(), torch.ones_like(selected, dtype=torch.float32)
    )


def pairwise_ranking_loss(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    clean = clean_logits.detach()[mask]
    corrupted = corrupted_logits[mask]
    if not len(corrupted):
        return corrupted_logits.sum() * 0.0
    return F.relu(float(margin) - (corrupted.float() - clean.float())).mean()


def corruption_type_loss(
    logits: torch.Tensor,
    type_index: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = logits[mask]
    if not len(selected):
        return logits.sum() * 0.0
    target = torch.full(
        (len(selected),),
        int(type_index),
        dtype=torch.long,
        device=selected.device,
    )
    return F.cross_entropy(selected.float(), target)


def cosine_recovery_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("recovery prediction and target shapes differ")
    error = 1.0 - F.cosine_similarity(
        prediction.float(), target.detach().float(), dim=-1, eps=float(epsilon)
    )
    if mask is None:
        return error.mean() if error.numel() else prediction.sum() * 0.0
    if mask.shape != error.shape:
        raise ValueError("recovery mask does not match prediction geometry")
    selected = error[mask]
    return selected.mean() if len(selected) else prediction.sum() * 0.0


def variance_covariance_loss(
    values: torch.Tensor,
    *,
    target_std: float = 1.0,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style anti-collapse terms over token embeddings."""

    if values.ndim != 2:
        raise ValueError("variance/covariance regularization expects [row, feature]")
    if len(values) < 2:
        zero = values.sum() * 0.0
        return zero, zero
    values = values.float()
    centered = values - values.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + float(epsilon))
    variance = F.relu(float(target_std) - std).mean()
    covariance = centered.T @ centered / float(max(len(values) - 1, 1))
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().sum() / float(values.shape[1])
    return variance, covariance_loss


def combine_breakdown(
    *,
    clean_energy: torch.Tensor,
    corrupt_energy: torch.Tensor,
    ranking: torch.Tensor,
    type_token: torch.Tensor,
    type_channel: torch.Tensor,
    clean_recovery: torch.Tensor,
    context_recovery: torch.Tensor,
    variance: torch.Tensor,
    covariance: torch.Tensor,
    weights,
) -> LossBreakdown:
    total = (
        float(weights.clean_energy_weight) * clean_energy
        + float(weights.corrupt_energy_weight) * corrupt_energy
        + float(weights.ranking_weight) * ranking
        + float(weights.type_weight) * (type_token + type_channel)
        + float(weights.clean_recovery_weight) * clean_recovery
        + float(weights.context_recovery_weight) * context_recovery
        + float(weights.variance_weight) * variance
        + float(weights.covariance_weight) * covariance
    )
    return LossBreakdown(
        total=total,
        clean_energy=clean_energy,
        corrupt_energy=corrupt_energy,
        ranking=ranking,
        type_token=type_token,
        type_channel=type_channel,
        clean_recovery=clean_recovery,
        context_recovery=context_recovery,
        variance=variance,
        covariance=covariance,
    )