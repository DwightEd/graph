"""Robust rupture-times-lock-in scoring over arbitrary soft route states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .config import ChangeConfig


MAD_NORMAL_SCALE = 1.482602218505602


def _validate_signal(values: torch.Tensor, name: str) -> None:
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty [N,C] tensor")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class MedianMAD:
    median: torch.Tensor
    scale: torch.Tensor

    def validate(self) -> "MedianMAD":
        if self.median.ndim != 1 or self.scale.shape != self.median.shape:
            raise ValueError("median and scale must be aligned [C] tensors")
        if self.scale.device != self.median.device:
            raise ValueError("median and scale must share one device")
        if not bool(torch.isfinite(self.median).all()) or not bool(
            torch.isfinite(self.scale).all()
        ):
            raise ValueError("robust statistics must be finite")
        if bool((self.scale <= 0).any()):
            raise ValueError("robust scales must be positive")
        return self

    def to(self, device: str | torch.device) -> "MedianMAD":
        return MedianMAD(self.median.to(device), self.scale.to(device))


@dataclass(frozen=True)
class RobustChangeStats:
    surprisal: MedianMAD
    prompt_lineage_drop: MedianMAD

    @property
    def num_channels(self) -> int:
        return int(self.surprisal.median.numel())

    def validate(self) -> "RobustChangeStats":
        self.surprisal.validate()
        self.prompt_lineage_drop.validate()
        if self.prompt_lineage_drop.median.shape != self.surprisal.median.shape:
            raise ValueError("robust signal statistics must share channel geometry")
        if self.prompt_lineage_drop.median.device != self.surprisal.median.device:
            raise ValueError("robust signal statistics must share one device")
        return self

    def to(self, device: str | torch.device) -> "RobustChangeStats":
        return RobustChangeStats(
            surprisal=self.surprisal.to(device),
            prompt_lineage_drop=self.prompt_lineage_drop.to(device),
        )


@dataclass(frozen=True)
class ChangeLockinResult:
    prompt_lineage_drop: torch.Tensor
    standardized_surprisal: torch.Tensor
    standardized_prompt_lineage_drop: torch.Tensor
    stability: torch.Tensor
    predictive_jsd: torch.Tensor
    predicted_detached: torch.Tensor
    detached_mass: torch.Tensor
    feedback_ema: torch.Tensor
    cusum: torch.Tensor
    rupture_memory: torch.Tensor
    lockin: torch.Tensor
    raw_channel_score: torch.Tensor

    @property
    def score(self) -> torch.Tensor:
        return self.raw_channel_score

    def validate(self) -> "ChangeLockinResult":
        shape = self.raw_channel_score.shape
        if len(shape) != 2:
            raise ValueError("raw_channel_score must be [R,C]")
        for tensor in (
            self.prompt_lineage_drop,
            self.standardized_surprisal,
            self.standardized_prompt_lineage_drop,
            self.stability,
            self.predictive_jsd,
            self.predicted_detached,
            self.detached_mass,
            self.feedback_ema,
            self.cusum,
            self.rupture_memory,
            self.lockin,
            self.raw_channel_score,
        ):
            if tensor.shape != shape:
                raise ValueError("change-lock-in outputs must share [R,C] shape")
            if tensor.device != self.raw_channel_score.device:
                raise ValueError("change-lock-in outputs must share one device")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("change-lock-in outputs must be finite")
        for tensor in (
            self.prompt_lineage_drop,
            self.stability,
            self.predictive_jsd,
            self.predicted_detached,
            self.detached_mass,
            self.feedback_ema,
            self.cusum,
            self.rupture_memory,
            self.lockin,
            self.raw_channel_score,
        ):
            if bool((tensor < -2e-6).any()):
                raise ValueError("mass/change outputs must be non-negative")
        if not torch.allclose(
            self.raw_channel_score,
            self.rupture_memory * self.lockin,
            atol=2e-6,
            rtol=2e-6,
        ):
            raise ValueError("final score must equal rupture_memory * lockin")
        return self


@torch.no_grad()
def fit_median_mad(
    values: torch.Tensor,
    *,
    scale_floor: float = 1e-3,
) -> MedianMAD:
    """Fit a channel-wise normal-consistent median/MAD reference."""

    _validate_signal(values, "values")
    if not torch.isfinite(torch.tensor(scale_floor)) or float(scale_floor) <= 0.0:
        raise ValueError("scale_floor must be positive and finite")
    values = values.detach()
    median = values.median(dim=0).values
    mad = (values - median).abs().median(dim=0).values
    scale = (MAD_NORMAL_SCALE * mad).clamp_min(float(scale_floor))
    return MedianMAD(median=median, scale=scale).validate()


def prompt_lineage_drop(prompt_lineage: torch.Tensor) -> torch.Tensor:
    """Positive prefix-local loss of prompt lineage, with zero at token 0.

    The central data view has no evidence-span boundary. This signal must not
    be described as evidence attribution or evidence grounding.
    """

    _validate_signal(prompt_lineage, "prompt_lineage")
    if bool(((prompt_lineage < 0) | (prompt_lineage > 1 + 2e-5)).any()):
        raise ValueError("prompt_lineage must be a probability")
    drop = torch.zeros_like(prompt_lineage)
    drop[1:] = (prompt_lineage[:-1] - prompt_lineage[1:]).clamp_min(0.0)
    return drop


@torch.no_grad()
def fit_robust_change_stats(
    surprisal: torch.Tensor,
    prompt_drop: torch.Tensor,
    *,
    config: ChangeConfig | None = None,
) -> RobustChangeStats:
    """Fit label-free robust references from disjoint fit-stream signals."""

    config = ChangeConfig() if config is None else config
    config.validate()
    _validate_signal(surprisal, "surprisal")
    _validate_signal(prompt_drop, "prompt_lineage_drop")
    if surprisal.shape != prompt_drop.shape:
        raise ValueError(
            "surprisal and prompt_lineage_drop must have the same [N,C] shape"
        )
    if surprisal.device != prompt_drop.device:
        raise ValueError("fit signals must share one device")
    return RobustChangeStats(
        surprisal=fit_median_mad(surprisal, scale_floor=config.scale_floor),
        prompt_lineage_drop=fit_median_mad(
            prompt_drop, scale_floor=config.scale_floor
        ),
    ).validate()


def _normalized_route(q: torch.Tensor, eps: float) -> torch.Tensor:
    if q.ndim != 3 or min(q.shape) < 1:
        raise ValueError("route_distribution must be non-empty [R,C,M]")
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("route_distribution must be finite and non-negative")
    total = q.sum(dim=-1, keepdim=True)
    if bool((total <= eps).any()):
        raise ValueError("every route row/channel must carry positive mass")
    return q.detach() / total


def _jsd_stability(q: torch.Tensor, eps: float) -> torch.Tensor:
    stability = torch.ones(q.shape[:2], dtype=q.dtype, device=q.device)
    if q.shape[0] < 2:
        return stability
    left, right = q[:-1], q[1:]
    middle = 0.5 * (left + right)

    def kl(part: torch.Tensor) -> torch.Tensor:
        term = torch.where(
            part > 0,
            part * (torch.log(part.clamp_min(eps)) - torch.log(middle.clamp_min(eps))),
            torch.zeros_like(part),
        )
        return term.sum(dim=-1)

    jsd = 0.5 * (kl(left) + kl(right))
    stability[1:] = (1.0 - jsd / torch.log(torch.tensor(2.0, device=q.device, dtype=q.dtype))).clamp(0.0, 1.0)
    return stability


def _normalized_jsd(left: torch.Tensor, right: torch.Tensor, eps: float) -> torch.Tensor:
    """Jensen-Shannon divergence normalized to the closed interval [0,1]."""

    middle = 0.5 * (left + right)

    def kl(part: torch.Tensor) -> torch.Tensor:
        term = torch.where(
            part > 0,
            part
            * (
                torch.log(part.clamp_min(eps))
                - torch.log(middle.clamp_min(eps))
            ),
            torch.zeros_like(part),
        )
        return term.sum(dim=-1)

    jsd = 0.5 * (kl(left) + kl(right))
    log_two = torch.log(torch.tensor(2.0, device=left.device, dtype=left.dtype))
    return (jsd / log_two).clamp(0.0, 1.0)


def _indices(indices: int | Iterable[int], states: int, device: torch.device) -> torch.Tensor:
    if isinstance(indices, int) and not isinstance(indices, bool):
        values = [indices]
    else:
        values = list(indices)
    if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("detached_indices must contain one or more integer state indices")
    if len(set(values)) != len(values) or min(values) < 0 or max(values) >= states:
        raise ValueError("detached_indices must be unique valid route-state indices")
    return torch.tensor(values, dtype=torch.long, device=device)


@torch.no_grad()
def change_lockin_score(
    route_distribution: torch.Tensor,
    surprisal: torch.Tensor,
    prompt_lineage: torch.Tensor,
    *,
    stats: RobustChangeStats,
    detached_indices: int | Iterable[int],
    predicted_route_distribution: torch.Tensor | None = None,
    config: ChangeConfig | None = None,
) -> ChangeLockinResult:
    """Combine a routing rupture with stable, detached feedback persistence.

    ``detached_indices`` is intentionally explicit: the five-state layered
    automaton uses ``R_PLUS``, while the seven-state token-path ablation uses
    its closed-response states. No route vocabulary is hard-coded here.
    """

    config = ChangeConfig() if config is None else config
    config.validate()
    q = _normalized_route(route_distribution, config.eps)
    response_count, channels, states = q.shape
    expected = (response_count, channels)
    if surprisal.shape != expected or prompt_lineage.shape != expected:
        raise ValueError("surprisal and prompt_lineage must match route [R,C]")
    if surprisal.device != q.device or prompt_lineage.device != q.device:
        raise ValueError("score inputs must share one device")
    if not bool(torch.isfinite(surprisal).all()) or not bool(
        torch.isfinite(prompt_lineage).all()
    ):
        raise ValueError("score inputs must be finite")
    if bool(((prompt_lineage < 0) | (prompt_lineage > 1 + 2e-5)).any()):
        raise ValueError("prompt_lineage must be a probability")
    stats = stats.to(q.device).validate()
    if stats.num_channels != channels:
        raise ValueError("robust statistics do not match route channels")
    selected = _indices(detached_indices, states, q.device)
    if predicted_route_distribution is None:
        predicted_q = q
    else:
        predicted_q = _normalized_route(
            predicted_route_distribution.to(device=q.device, dtype=q.dtype),
            config.eps,
        )
        if predicted_q.shape != q.shape:
            raise ValueError("predicted route distribution must match [R,C,M]")

    prompt_drop = prompt_lineage_drop(prompt_lineage)
    z_surprisal = (
        surprisal - stats.surprisal.median.unsqueeze(0)
    ) / stats.surprisal.scale.unsqueeze(0)
    z_prompt_drop = (
        prompt_drop - stats.prompt_lineage_drop.median.unsqueeze(0)
    ) / stats.prompt_lineage_drop.scale.unsqueeze(0)
    innovation = (
        z_surprisal.clamp_min(0.0)
        + float(config.prompt_lineage_drop_weight)
        * z_prompt_drop.clamp_min(0.0)
        - float(config.cusum_slack)
    )

    cusum = torch.zeros_like(surprisal)
    rupture_memory = torch.zeros_like(surprisal)
    for token in range(response_count):
        previous_cusum = cusum[token - 1] if token else torch.zeros_like(cusum[0])
        cusum[token] = (previous_cusum + innovation[token]).clamp_min(0.0)
        previous_memory = (
            rupture_memory[token - 1]
            if token
            else torch.zeros_like(rupture_memory[0])
        )
        rupture_memory[token] = torch.maximum(
            cusum[token], float(config.rupture_decay) * previous_memory
        )

    stability = _jsd_stability(q, config.eps)
    predictive_jsd = _normalized_jsd(q, predicted_q, config.eps)
    detached_mass = q.index_select(-1, selected).sum(dim=-1)
    predicted_detached = predicted_q.index_select(-1, selected).sum(dim=-1)
    feedback_ema = torch.zeros_like(detached_mass)
    feedback_ema[0] = detached_mass[0]
    decay = float(config.feedback_ema_decay)
    for token in range(1, response_count):
        feedback_ema[token] = (
            decay * feedback_ema[token - 1]
            + (1.0 - decay) * detached_mass[token]
        )
    lockin = (
        (1.0 - prompt_lineage).clamp(0.0, 1.0)
        * detached_mass
        * feedback_ema
        * predicted_detached
        * stability
    )
    # One route state cannot establish temporal persistence by itself.
    lockin[0] = 0.0
    raw_channel_score = rupture_memory * lockin
    return ChangeLockinResult(
        prompt_lineage_drop=prompt_drop,
        standardized_surprisal=z_surprisal,
        standardized_prompt_lineage_drop=z_prompt_drop,
        stability=stability,
        predictive_jsd=predictive_jsd,
        predicted_detached=predicted_detached,
        detached_mass=detached_mass,
        feedback_ema=feedback_ema,
        cusum=cusum,
        rupture_memory=rupture_memory,
        lockin=lockin,
        raw_channel_score=raw_channel_score,
    ).validate()
