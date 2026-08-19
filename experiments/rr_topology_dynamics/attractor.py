"""Core features for prompt-detached, self-reinforcing routing attractors."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .routing_state import RoutingState


PRIMARY_FEATURE_NAMES = (
    "prompt_attention_share",
    "prompt_source_effective_fraction",
    "prompt_source_top1_share",
    "response_source_effective_fraction",
    "response_source_top1_share",
    "recent_response_share",
    "source_stability",
    "prompt_groundedness",
)

CONTROL_FEATURE_NAMES = (
    "retained_attention_mass",
    "retained_edge_count",
)


@dataclass(frozen=True)
class AttractorFeatures:
    names: tuple[str, ...]
    values: torch.Tensor
    control_names: tuple[str, ...]
    controls: torch.Tensor


class AttractorFeatureExtractor:
    """Compute the predeclared concentration, stability, and grounding signals."""

    def __init__(self, *, recent_lag_max: int = 4, epsilon: float = 1e-8) -> None:
        if int(recent_lag_max) < 1:
            raise ValueError("recent_lag_max must be positive")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        self.recent_lag_max = int(recent_lag_max)
        self.epsilon = float(epsilon)

    def extract(self, state: RoutingState) -> AttractorFeatures:
        prompt_mass = state.prompt_source_mass.float()
        response_mass = state.response_source_mass.float()
        prompt_total = prompt_mass.sum(dim=1)
        response_total = response_mass.sum(dim=1)
        retained_total = prompt_total + response_total

        prompt_share = prompt_total / retained_total.clamp_min(self.epsilon)
        prompt_share = torch.where(
            retained_total > self.epsilon, prompt_share, torch.zeros_like(prompt_share)
        )
        prompt_effective, prompt_top1 = _source_concentration(
            prompt_mass,
            torch.full_like(prompt_total, state.prompt_count, dtype=torch.float32),
            self.epsilon,
        )
        response_effective, response_top1 = _source_concentration(
            response_mass,
            torch.arange(
                state.response_count,
                dtype=torch.float32,
                device=response_mass.device,
            ),
            self.epsilon,
        )
        recent_share = _recent_response_share(
            response_mass, self.recent_lag_max, self.epsilon
        )
        stability = _source_stability(prompt_mass, response_mass, self.epsilon)
        groundedness = _prompt_groundedness(
            prompt_share, response_mass, self.epsilon
        )

        values = torch.stack(
            (
                prompt_share,
                prompt_effective,
                prompt_top1,
                response_effective,
                response_top1,
                recent_share,
                stability,
                groundedness,
            ),
            dim=1,
        )
        controls = torch.stack(
            (retained_total, state.retained_edge_count.float()), dim=1
        )
        return AttractorFeatures(
            names=PRIMARY_FEATURE_NAMES,
            values=values,
            control_names=CONTROL_FEATURE_NAMES,
            controls=controls,
        )


def _source_concentration(
    mass: torch.Tensor, available_sources: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    total = mass.sum(dim=1)
    probability = mass / total[:, None].clamp_min(epsilon)
    positive = probability > 0
    entropy = -torch.where(
        positive,
        probability * probability.clamp_min(epsilon).log(),
        torch.zeros_like(probability),
    ).sum(dim=1)
    effective_fraction = entropy.exp() / available_sources.clamp_min(1.0)
    top1_share = probability.max(dim=1).values
    valid = total > epsilon
    return (
        torch.where(valid, effective_fraction, torch.zeros_like(effective_fraction)),
        torch.where(valid, top1_share, torch.zeros_like(top1_share)),
    )


def _recent_response_share(
    response_mass: torch.Tensor, recent_lag_max: int, epsilon: float
) -> torch.Tensor:
    response_count = int(response_mass.shape[0])
    query = torch.arange(response_count, device=response_mass.device)[:, None]
    source = torch.arange(response_count, device=response_mass.device)[None, :]
    recent = (query - source > 0) & (query - source <= recent_lag_max)
    total = response_mass.sum(dim=1)
    share = (response_mass * recent).sum(dim=1) / total.clamp_min(epsilon)
    return torch.where(total > epsilon, share, torch.zeros_like(share))


def _source_stability(
    prompt_mass: torch.Tensor, response_mass: torch.Tensor, epsilon: float
) -> torch.Tensor:
    mass = torch.cat((prompt_mass, response_mass), dim=1)
    total = mass.sum(dim=1)
    probability = mass / total[:, None].clamp_min(epsilon)
    stability = torch.zeros_like(total)
    if len(probability) < 2:
        return stability
    previous = probability[:-1]
    current = probability[1:]
    midpoint = 0.5 * (previous + current)
    previous_kl = torch.where(
        previous > 0,
        previous * (previous.clamp_min(epsilon) / midpoint.clamp_min(epsilon)).log(),
        torch.zeros_like(previous),
    ).sum(dim=1)
    current_kl = torch.where(
        current > 0,
        current * (current.clamp_min(epsilon) / midpoint.clamp_min(epsilon)).log(),
        torch.zeros_like(current),
    ).sum(dim=1)
    distance = torch.sqrt(
        (0.5 * (previous_kl + current_kl) / math.log(2.0)).clamp(0, 1)
    )
    valid = (total[:-1] > epsilon) & (total[1:] > epsilon)
    stability[1:] = torch.where(valid, 1.0 - distance, torch.zeros_like(distance))
    return stability


def _prompt_groundedness(
    prompt_share: torch.Tensor, response_mass: torch.Tensor, epsilon: float
) -> torch.Tensor:
    groundedness = torch.zeros_like(prompt_share)
    for token in range(len(prompt_share)):
        history = response_mass[token, :token]
        history_total = history.sum()
        relay = (history / history_total.clamp_min(epsilon)) @ groundedness[:token]
        groundedness[token] = prompt_share[token] + (
            1.0 - prompt_share[token]
        ) * relay
    return groundedness
