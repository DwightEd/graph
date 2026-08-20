"""Causal concentration traces and explicit three-state dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .majorization import hill_diversity_spectrum, majorization_evidence
from .routing import RoutingEdges


@dataclass(frozen=True)
class ExactPromptRoutes:
    probability: torch.Tensor
    excess_mass: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class CausalRouteTrace:
    majorization_evidence: torch.Tensor
    concentration_level: torch.Tensor
    hill_shape: torch.Tensor
    source_affinity: torch.Tensor
    valid_channel_fraction: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class CausalStateTrace:
    state_probability: torch.Tensor
    entry_probability: torch.Tensor
    basin_probability: torch.Tensor
    current_probability: torch.Tensor
    forecast_probability: torch.Tensor


def exact_prompt_routes(
    edges: RoutingEdges,
    *,
    token: int,
    epsilon: float = 1e-12,
) -> ExactPromptRoutes:
    """Build one token's exact prompt routes from observable excess mass.

    Each layer/head remains an independent channel. The stored censoring floor
    is subtracted from retained off-diagonal weights, so an absent edge has the
    same zero excess as any unobserved below-floor edge.
    """

    if not 0 <= token < edges.num_response_tokens:
        raise ValueError("token is outside the response")
    shape = (edges.num_layers, edges.num_heads, edges.response_idx)
    excess = edges.weight.new_zeros(shape)
    selected = (edges.source < edges.response_idx) & (edges.query == token)
    if selected.any():
        weight = (edges.weight[selected] - edges.attention_floor).clamp_min(0.0)
        excess.index_put_(
            (
                edges.layer[selected],
                edges.head[selected],
                edges.source[selected],
            ),
            weight,
            accumulate=True,
        )
    mass = excess.sum(dim=-1)
    valid = mass > epsilon
    probability = excess / mass.unsqueeze(-1).clamp_min(epsilon)
    return ExactPromptRoutes(
        probability=probability,
        excess_mass=mass,
        valid=valid,
    )


def _valid_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum()
    if int(count) == 0:
        return values.new_zeros(())
    return values[valid].mean()


def _causal_trace_from_rows(
    rows,
    *,
    token_count: int,
    channels: int,
    sources: int,
    prototype: torch.Tensor,
    history_decay: float,
    majorization_tolerance: float,
    epsilon: float,
) -> CausalRouteTrace:
    history = prototype.new_zeros((channels, sources))
    history_valid = torch.zeros(
        channels, dtype=torch.bool, device=prototype.device
    )
    previous = prototype.new_zeros((channels, sources))
    previous_valid = torch.zeros_like(history_valid)

    majorization_rows = []
    concentration_rows = []
    hill_shape_rows = []
    affinity_rows = []
    valid_fraction_rows = []
    token_valid_rows = []

    for current_values in rows:
        current_values = current_values.clamp_min(0.0).reshape(channels, sources)
        total = current_values.sum(dim=-1, keepdim=True)
        current_valid = total.squeeze(-1) > epsilon
        current = current_values / total.clamp_min(epsilon)
        comparison_valid = current_valid & history_valid
        comparison = majorization_evidence(
            current,
            history,
            tolerance=majorization_tolerance,
            epsilon=epsilon,
        )
        current_hill = hill_diversity_spectrum(current, epsilon=epsilon)
        history_hill = hill_diversity_spectrum(history, epsilon=epsilon)
        log_diversity_ratio = (
            history_hill.clamp_min(epsilon).log()
            - current_hill.clamp_min(epsilon).log()
        )
        concentration = log_diversity_ratio.mean(dim=-1)
        hill_shape = log_diversity_ratio.std(dim=-1, unbiased=False)

        affinity_valid = current_valid & previous_valid
        affinity = (current.sqrt() * previous.sqrt()).sum(dim=-1).clamp(0.0, 1.0)

        majorization_rows.append(_valid_mean(comparison.evidence, comparison_valid))
        concentration_rows.append(_valid_mean(concentration, comparison_valid))
        hill_shape_rows.append(_valid_mean(hill_shape, comparison_valid))
        affinity_rows.append(_valid_mean(affinity, affinity_valid))
        valid_fraction_rows.append(comparison_valid.float().mean())
        token_valid_rows.append(comparison_valid.any() & affinity_valid.any())

        initialize = current_valid & ~history_valid
        update = current_valid & history_valid
        history[initialize] = current[initialize]
        history[update] = (
            history_decay * history[update]
            + (1.0 - history_decay) * current[update]
        )
        history_valid |= current_valid
        previous[current_valid] = current[current_valid]
        previous_valid = current_valid

    if len(majorization_rows) != token_count:
        raise ValueError("route row iterator did not cover every response token")
    return CausalRouteTrace(
        majorization_evidence=torch.stack(majorization_rows),
        concentration_level=torch.stack(concentration_rows),
        hill_shape=torch.stack(hill_shape_rows),
        source_affinity=torch.stack(affinity_rows),
        valid_channel_fraction=torch.stack(valid_fraction_rows),
        valid=torch.stack(token_valid_rows),
    )


def causal_route_trace(
    prompt_source_probability: torch.Tensor,
    *,
    history_decay: float = 0.9,
    majorization_tolerance: float = 1e-6,
    epsilon: float = 1e-12,
) -> CausalRouteTrace:
    """Summarize ``[token, layer, head, exact prompt source]`` causally.

    The historical reference seen at token ``t`` contains only rows before
    ``t``. Layer/head channels are compared independently and are aggregated
    only after each channel's exact-source statistics have been computed.
    """

    if prompt_source_probability.ndim != 4:
        raise ValueError(
            "prompt_source_probability must have shape [token, layer, head, source]"
        )
    if not 0.0 <= history_decay < 1.0:
        raise ValueError("history_decay must be in [0, 1)")

    token_count, layers, heads, sources = prompt_source_probability.shape
    channels = layers * heads
    values = prompt_source_probability.reshape(token_count, channels, sources)
    return _causal_trace_from_rows(
        (values[token] for token in range(token_count)),
        token_count=token_count,
        channels=channels,
        sources=sources,
        prototype=values,
        history_decay=history_decay,
        majorization_tolerance=majorization_tolerance,
        epsilon=epsilon,
    )


def causal_route_trace_from_edges(
    edges: RoutingEdges,
    *,
    history_decay: float = 0.9,
    majorization_tolerance: float = 1e-6,
    epsilon: float = 1e-12,
) -> CausalRouteTrace:
    """Stream exact prompt routes without allocating ``[token, L, H, P]``."""

    if not 0.0 <= history_decay < 1.0:
        raise ValueError("history_decay must be in [0, 1)")
    channels = edges.num_layers * edges.num_heads
    sources = edges.response_idx
    selected = edges.source < sources
    query = edges.query[selected]
    channel = edges.layer[selected] * edges.num_heads + edges.head[selected]
    source = edges.source[selected]
    weight = (edges.weight[selected] - edges.attention_floor).clamp_min(0.0)

    order = query.argsort()
    query = query[order]
    channel = channel[order]
    source = source[order]
    weight = weight[order]
    counts = torch.bincount(query, minlength=edges.num_response_tokens)
    stops = counts.cumsum(dim=0)
    starts = torch.cat((stops.new_zeros(1), stops[:-1]))

    def rows():
        for token in range(edges.num_response_tokens):
            current = weight.new_zeros((channels, sources))
            start = int(starts[token])
            stop = int(stops[token])
            if stop > start:
                flat_index = channel[start:stop] * sources + source[start:stop]
                current.reshape(-1).index_add_(0, flat_index, weight[start:stop])
            yield current

    return _causal_trace_from_rows(
        rows(),
        token_count=edges.num_response_tokens,
        channels=channels,
        sources=sources,
        prototype=edges.weight,
        history_decay=history_decay,
        majorization_tolerance=majorization_tolerance,
        epsilon=epsilon,
    )


class CausalStateFilter:
    """Filter distributed, entry, and concentrated-resident route states.

    Observations are ``[majorization z, Hill-concentration z, affinity]``.
    The filter is descriptive: its states concern routing geometry and do not
    themselves assert that a token is hallucinated.
    """

    def __init__(self, transition: torch.Tensor | None = None):
        if transition is None:
            transition = torch.tensor(
                [
                    [0.96, 0.04, 0.00],
                    [0.15, 0.20, 0.65],
                    [0.05, 0.05, 0.90],
                ],
                dtype=torch.float32,
            )
        if transition.shape != (3, 3):
            raise ValueError("transition must have shape [3, 3]")
        if not bool(torch.isfinite(transition).all()):
            raise ValueError("transition must be finite")
        if bool((transition < 0).any()):
            raise ValueError("transition probabilities must be non-negative")
        expected_sum = torch.ones(
            3, dtype=transition.dtype, device=transition.device
        )
        if not torch.allclose(transition.sum(dim=-1), expected_sum):
            raise ValueError("transition rows must sum to one")
        self.transition = transition

    def run(
        self,
        observations: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> CausalStateTrace:
        if observations.ndim != 2 or observations.shape[1] != 3:
            raise ValueError("observations must have shape [token, 3]")
        if valid is None:
            valid = torch.ones(
                len(observations), dtype=torch.bool, device=observations.device
            )
        if valid.shape != (len(observations),):
            raise ValueError("valid must have shape [token]")

        transition = self.transition.to(
            device=observations.device,
            dtype=observations.dtype,
        )
        posterior = observations.new_tensor([1.0, 0.0, 0.0])
        rows = []
        forecasts = []
        initial = observations.new_tensor([1.0, 0.0, 0.0])
        nan_state = observations.new_full((3,), float("nan"))
        nan_score = observations.new_tensor(float("nan"))
        for observation, is_valid in zip(observations, valid, strict=True):
            if not bool(is_valid):
                rows.append(nan_state)
                forecasts.append(nan_score)
                posterior = initial
                continue
            majorization_z, concentration_z, affinity = observation
            concentration = 0.5 * (majorization_z + concentration_z)
            affinity = affinity.clamp(0.0, 1.0)
            emission_logits = torch.stack(
                (
                    -1.5 * concentration,
                    1.5 * concentration + 3.0 * (0.5 - affinity),
                    1.5 * concentration + 3.0 * (affinity - 0.5),
                )
            )
            emission = emission_logits.softmax(dim=0)
            posterior = posterior * emission
            posterior = posterior / posterior.sum().clamp_min(1e-12)
            rows.append(posterior)
            forecast = posterior @ transition
            forecasts.append(forecast[1:].sum())
            posterior = forecast

        state_probability = torch.stack(rows)
        return CausalStateTrace(
            state_probability=state_probability,
            entry_probability=state_probability[:, 1],
            basin_probability=state_probability[:, 2],
            current_probability=state_probability[:, 1:].sum(dim=-1),
            forecast_probability=torch.stack(forecasts),
        )
