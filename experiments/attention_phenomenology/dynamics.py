"""Exact response-source concentration and temporal persistence."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PhenomenologyConfig


@dataclass(frozen=True)
class ResponseSourceDynamics:
    effective_number: torch.Tensor
    top1_share: torch.Tensor
    recent_mass_fraction: torch.Tensor
    mean_lag: torch.Tensor
    anchor_turnover: torch.Tensor
    distribution_velocity: torch.Tensor


def _source_probability(
    source_mass: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    total = source_mass.sum(dim=2)
    probability = source_mass / total[:, :, None].clamp_min(epsilon)
    probability = torch.where(
        total[:, :, None] > epsilon, probability, torch.zeros_like(probability)
    )
    return probability, total


def _anchor_turnover(
    source_mass: torch.Tensor, *, count: int, epsilon: float
) -> torch.Tensor:
    response_count, layers, _ = source_mass.shape
    keep = min(count, response_count)
    turnover = source_mass.new_zeros((response_count, layers))
    if response_count < 2 or keep == 0:
        return turnover

    values, indices = torch.topk(source_mass, k=keep, dim=2)
    valid = values > epsilon
    current_indices = indices[1:]
    previous_indices = indices[:-1]
    current_valid = valid[1:]
    previous_valid = valid[:-1]
    match = current_indices[:, :, :, None] == previous_indices[:, :, None, :]
    match &= current_valid[:, :, :, None] & previous_valid[:, :, None, :]
    intersection = match.any(dim=3).sum(dim=2).float()
    current_count = current_valid.sum(dim=2).float()
    previous_count = previous_valid.sum(dim=2).float()
    union = current_count + previous_count - intersection
    turnover[1:] = torch.where(
        union > 0, 1.0 - intersection / union, torch.zeros_like(union)
    )
    return turnover


def _distribution_velocity(probability: torch.Tensor) -> torch.Tensor:
    velocity = probability.new_zeros(probability.shape[:2])
    if probability.shape[0] < 2:
        return velocity
    affinity = (
        probability[1:].sqrt() * probability[:-1].sqrt()
    ).sum(dim=2).clamp(0.0, 1.0)
    velocity[1:] = (1.0 - affinity).sqrt()
    return velocity


def response_source_dynamics(
    source_mass: torch.Tensor,
    *,
    config: PhenomenologyConfig,
) -> ResponseSourceDynamics:
    """Summarize exact response anchors without discarding layer depth."""

    probability, total = _source_probability(source_mass, config.epsilon)
    entropy = -(
        probability * probability.clamp_min(config.epsilon).log()
    ).sum(dim=2)
    effective = torch.where(
        total > config.epsilon, entropy.exp(), torch.zeros_like(total)
    )
    top1 = torch.where(
        total > config.epsilon,
        probability.max(dim=2).values,
        torch.zeros_like(total),
    )

    response_count = source_mass.shape[0]
    query = torch.arange(response_count, device=source_mass.device)[:, None]
    source = torch.arange(response_count, device=source_mass.device)[None, :]
    lag = query - source
    recent = (
        probability
        * ((lag > 0) & (lag <= config.recent_lag_max))[:, None, :]
    ).sum(dim=2)
    mean_lag = (probability * lag.clamp_min(0)[:, None, :]).sum(dim=2)

    return ResponseSourceDynamics(
        effective_number=effective,
        top1_share=top1,
        recent_mass_fraction=recent,
        mean_lag=mean_lag,
        anchor_turnover=_anchor_turnover(
            source_mass, count=config.anchor_count, epsilon=config.epsilon
        ),
        distribution_velocity=_distribution_velocity(probability),
    )
