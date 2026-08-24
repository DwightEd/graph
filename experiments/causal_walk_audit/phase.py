"""Grammar rupture and response-closure phase scores."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .automaton import CLOSED_STATES, P0, P_PLUS
from .config import PhaseConfig

MAD_SCALE = 1.482602218505602


@dataclass(frozen=True)
class ChannelStats:
    median: torch.Tensor
    scale: torch.Tensor

    def to(self, device: str | torch.device) -> "ChannelStats":
        return ChannelStats(self.median.to(device), self.scale.to(device))


@dataclass(frozen=True)
class PhaseResult:
    standardized_surprisal: torch.Tensor
    cusum: torch.Tensor
    rupture: torch.Tensor
    closure_mass: torch.Tensor
    predicted_closure: torch.Tensor
    stability: torch.Tensor
    closure_ema: torch.Tensor
    closure_score: torch.Tensor
    rupture_closure: torch.Tensor
    prompt_lineage: torch.Tensor


@torch.no_grad()
def fit_channel_stats(
    values: torch.Tensor,
    *,
    scale_floor: float,
) -> ChannelStats:
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("phase reference needs at least two [row, channel] values")
    median = values.median(dim=0).values
    mad = (values - median).abs().median(dim=0).values
    scale = (MAD_SCALE * mad).clamp_min(scale_floor)
    return ChannelStats(median=median, scale=scale)


def _stability(q: torch.Tensor, epsilon: float) -> torch.Tensor:
    result = q.new_ones(q.shape[:2])
    if len(q) < 2:
        return result
    left, right = q[:-1], q[1:]
    middle = 0.5 * (left + right)

    def kl(part: torch.Tensor) -> torch.Tensor:
        return torch.where(
            part > 0,
            part
            * (
                part.clamp_min(epsilon).log()
                - middle.clamp_min(epsilon).log()
            ),
            torch.zeros_like(part),
        ).sum(dim=-1)

    jsd = 0.5 * (kl(left) + kl(right))
    log_two = torch.log(q.new_tensor(2.0))
    result[1:] = (1.0 - jsd / log_two).clamp(0.0, 1.0)
    return result


@torch.no_grad()
def score_phase(
    route: torch.Tensor,
    predicted: torch.Tensor,
    surprisal: torch.Tensor,
    *,
    stats: ChannelStats,
    config: PhaseConfig | None = None,
) -> PhaseResult:
    """Score a recent grammar rupture and a separate closure diagnostic."""

    config = PhaseConfig() if config is None else config
    route = route / route.sum(dim=-1, keepdim=True).clamp_min(config.epsilon)
    predicted = predicted / predicted.sum(dim=-1, keepdim=True).clamp_min(
        config.epsilon
    )
    stats = stats.to(route.device)
    z = (surprisal - stats.median[None]) / stats.scale[None]
    innovation = (z - config.cusum_slack).clamp_min(0.0)
    innovation[0] = 0.0

    cusum = torch.zeros_like(surprisal)
    rupture = torch.zeros_like(surprisal)
    for token in range(len(route)):
        previous_cusum = cusum[token - 1] if token else torch.zeros_like(cusum[0])
        previous_rupture = (
            rupture[token - 1] if token else torch.zeros_like(rupture[0])
        )
        cusum[token] = (previous_cusum + innovation[token]).clamp_min(0.0)
        rupture[token] = torch.maximum(
            cusum[token],
            config.rupture_decay * previous_rupture,
        )

    closure = route[..., list(CLOSED_STATES)].sum(dim=-1)
    predicted_closure = predicted[..., list(CLOSED_STATES)].sum(dim=-1)
    stability = _stability(route, config.epsilon)
    closure_ema = torch.zeros_like(closure)
    closure_ema[0] = closure[0]
    for token in range(1, len(route)):
        closure_ema[token] = (
            config.closure_decay * closure_ema[token - 1]
            + (1.0 - config.closure_decay) * closure[token]
        )

    prompt_lineage = route[..., P0] + route[..., P_PLUS]
    closure_score = closure * predicted_closure * closure_ema * stability
    closure_score[0] = 0.0
    return PhaseResult(
        standardized_surprisal=z,
        cusum=cusum,
        rupture=rupture,
        closure_mass=closure,
        predicted_closure=predicted_closure,
        stability=stability,
        closure_ema=closure_ema,
        closure_score=closure_score,
        rupture_closure=rupture * closure_score,
        prompt_lineage=prompt_lineage,
    )
