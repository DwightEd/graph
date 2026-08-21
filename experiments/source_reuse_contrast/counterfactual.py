"""Self-supervised losses and origin-specific counterfactual scores."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .grounding_config import GroundingGraphConfig
from .provenance import TokenGroundingTargets


@dataclass(frozen=True)
class ReconstructionLoss:
    total: torch.Tensor
    received: torch.Tensor
    grounding: torch.Tensor
    provenance: torch.Tensor


@dataclass(frozen=True)
class CounterfactualScores:
    reconstruction: torch.Tensor
    raw_reconstruction: torch.Tensor
    prompt_removed: torch.Tensor
    response_removed: torch.Tensor
    perturbed: torch.Tensor
    no_state: torch.Tensor
    shuffled_state: torch.Tensor
    endpoint_rewired: torch.Tensor
    prompt_gain: torch.Tensor
    response_gain: torch.Tensor
    closure: torch.Tensor
    fragility: torch.Tensor
    refinement_gain: torch.Tensor
    state_gain: torch.Tensor
    memory_specificity: torch.Tensor
    endpoint_specificity: torch.Tensor


def reconstruction_loss(
    prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target: TokenGroundingTargets,
    config: GroundingGraphConfig,
) -> ReconstructionLoss:
    received_prediction, grounding_prediction, provenance_prediction = prediction
    received = F.smooth_l1_loss(
        received_prediction,
        target.received_support,
    )
    grounding = F.smooth_l1_loss(
        grounding_prediction,
        target.grounding_field,
    )
    provenance = F.smooth_l1_loss(
        provenance_prediction,
        target.provenance,
    )
    total = (
        config.reuse_loss_weight * received
        + config.grounding_loss_weight * grounding
        + config.provenance_loss_weight * provenance
    )
    return ReconstructionLoss(total, received, grounding, provenance)


def build_scores(
    *,
    raw: ReconstructionLoss,
    full: ReconstructionLoss,
    no_prompt: ReconstructionLoss,
    no_response: ReconstructionLoss,
    perturbed: ReconstructionLoss,
    no_state: ReconstructionLoss,
    shuffled_state: ReconstructionLoss,
    endpoint_rewired: ReconstructionLoss,
) -> CounterfactualScores:
    prompt_gain = no_prompt.total - full.total
    response_gain = no_response.total - full.total
    return CounterfactualScores(
        reconstruction=full.total,
        raw_reconstruction=raw.total,
        prompt_removed=no_prompt.total,
        response_removed=no_response.total,
        perturbed=perturbed.total,
        no_state=no_state.total,
        shuffled_state=shuffled_state.total,
        endpoint_rewired=endpoint_rewired.total,
        prompt_gain=prompt_gain,
        response_gain=response_gain,
        closure=response_gain - prompt_gain,
        fragility=perturbed.total - full.total,
        refinement_gain=raw.total - full.total,
        state_gain=no_state.total - full.total,
        memory_specificity=shuffled_state.total - full.total,
        endpoint_specificity=endpoint_rewired.total - full.total,
    )
