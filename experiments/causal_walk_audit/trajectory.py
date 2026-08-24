"""Direct/relay congruence, recoupling, audit escape, and lock-in metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .lineage import LineageTrace


@dataclass(frozen=True)
class TrajectoryReference:
    js_high: float
    js_low: float
    evidence_high: float
    evidence_low: float
    response_high: float

    def to_dict(self) -> dict[str, float]:
        return {
            "js_high": self.js_high,
            "js_low": self.js_low,
            "evidence_high": self.evidence_high,
            "evidence_low": self.evidence_low,
            "response_high": self.response_high,
        }

    @classmethod
    def from_dict(cls, value: dict[str, float]) -> "TrajectoryReference":
        return cls(**{name: float(current) for name, current in value.items()})


@dataclass(frozen=True)
class LayerTrajectory:
    anchor_js: np.ndarray
    direct_mass: np.ndarray
    relay_mass: np.ndarray
    response_base: np.ndarray
    known_anchor: np.ndarray


@dataclass(frozen=True)
class TokenTrajectory:
    anchor_js_mean: np.ndarray
    anchor_js_peak: np.ndarray
    recoupling_depth: np.ndarray
    recoupling_failure: np.ndarray
    response_persistence: np.ndarray
    evidence_escape: np.ndarray
    lock_in: np.ndarray


def _normalize(value: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    mass = value.sum(dim=-1)
    return value / mass.clamp_min(epsilon)[..., None], mass


def _js_divergence(first: torch.Tensor, second: torch.Tensor, epsilon: float) -> torch.Tensor:
    middle = 0.5 * (first + second)
    kl_first = (
        first
        * (first.clamp_min(epsilon).log() - middle.clamp_min(epsilon).log())
    ).sum(-1)
    kl_second = (
        second
        * (second.clamp_min(epsilon).log() - middle.clamp_min(epsilon).log())
    ).sum(-1)
    return 0.5 * (kl_first + kl_second)


def layer_trajectory(
    trace: LineageTrace,
    *,
    minimum_anchor_mass: float,
    epsilon: float = 1e-8,
) -> LayerTrajectory:
    direct, direct_mass = _normalize(trace.direct_anchor(), epsilon)
    relay, relay_mass = _normalize(trace.relay_anchor(), epsilon)
    valid = (direct_mass >= minimum_anchor_mass) & (
        relay_mass >= minimum_anchor_mass
    )
    js = _js_divergence(direct, relay, epsilon)
    js = torch.where(valid, js, torch.full_like(js, float("nan")))

    def head_mean(value: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(value)
        total = torch.where(finite, value, torch.zeros_like(value)).sum(dim=-1)
        return total / finite.sum(dim=-1).clamp_min(1)

    return LayerTrajectory(
        anchor_js=head_mean(js).cpu().numpy().astype(np.float32),
        direct_mass=direct_mass.mean(dim=-1).cpu().numpy().astype(np.float32),
        relay_mass=relay_mass.mean(dim=-1).cpu().numpy().astype(np.float32),
        response_base=trace.response_base()
        .mean(dim=-1)
        .cpu()
        .numpy()
        .astype(np.float32),
        known_anchor=trace.known_anchor_mass()
        .mean(dim=-1)
        .cpu()
        .numpy()
        .astype(np.float32),
    )


def fit_trajectory_reference(layers: list[LayerTrajectory]) -> TrajectoryReference:
    js = np.concatenate(
        [item.anchor_js[np.isfinite(item.anchor_js)] for item in layers]
    )
    evidence = np.concatenate([item.known_anchor.ravel() for item in layers])
    response = np.concatenate([item.response_base.ravel() for item in layers])
    return TrajectoryReference(
        js_high=float(np.quantile(js, 0.9)) if len(js) else 0.0,
        js_low=float(np.quantile(js, 0.5)) if len(js) else 0.0,
        evidence_high=float(np.quantile(evidence, 0.75)),
        evidence_low=float(np.quantile(evidence, 0.25)),
        response_high=float(np.quantile(response, 0.75)),
    )


def summarize_trajectory(
    layer: LayerTrajectory,
    reference: TrajectoryReference,
    *,
    horizon: int,
) -> TokenTrajectory:
    tokens, layers = layer.known_anchor.shape
    finite = np.isfinite(layer.anchor_js)
    count = finite.sum(axis=1)
    js_mean = np.where(
        count > 0,
        np.where(finite, layer.anchor_js, 0.0).sum(axis=1)
        / np.maximum(count, 1),
        0.0,
    )
    js_peak = np.where(
        count > 0,
        np.where(finite, layer.anchor_js, -np.inf).max(axis=1),
        0.0,
    )
    recoupling_depth = np.zeros(tokens, dtype=np.float32)
    recoupling_failure = np.zeros(tokens, dtype=np.float32)

    for token in range(tokens):
        values = layer.anchor_js[token]
        finite_layers = np.flatnonzero(np.isfinite(values))
        if not len(finite_layers):
            continue
        peak = int(finite_layers[np.nanargmax(values[finite_layers])])
        if values[peak] < reference.js_high:
            continue
        later = np.flatnonzero(
            np.isfinite(values[peak + 1 :])
            & (values[peak + 1 :] <= reference.js_low)
        )
        if len(later):
            recoupling_depth[token] = float(later[0] + 1)
        else:
            recoupling_depth[token] = float(layers - peak - 1)
            recoupling_failure[token] = 1.0

    evidence_state = (
        layer.known_anchor.mean(axis=1) >= reference.evidence_high
    )
    response_state = (
        layer.response_base.mean(axis=1) >= reference.response_high
    ) & (layer.known_anchor.mean(axis=1) <= reference.evidence_low)
    persistence = np.zeros(tokens, dtype=np.float32)
    escape = np.zeros(tokens, dtype=np.float32)
    for token in range(tokens):
        future = slice(token + 1, min(tokens, token + 1 + horizon))
        if future.start == future.stop:
            continue
        persistence[token] = float(response_state[future].mean())
        escape[token] = float(evidence_state[future].any())

    js_scale = max(reference.js_high - reference.js_low, 1e-6)
    normalized_js = np.maximum(
        (js_peak - reference.js_low) / js_scale,
        0.0,
    )
    lock_in = normalized_js * persistence * (1.0 - escape)
    return TokenTrajectory(
        anchor_js_mean=np.nan_to_num(js_mean).astype(np.float32),
        anchor_js_peak=js_peak.astype(np.float32),
        recoupling_depth=recoupling_depth,
        recoupling_failure=recoupling_failure,
        response_persistence=persistence,
        evidence_escape=escape,
        lock_in=lock_in.astype(np.float32),
    )
