"""Self-supervised objectives and local token scoring for HoloRoute."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from experiments.attention_holonomy_audit.graph import AttentionEventGraph

from .masking import mask_graph_inputs, structurally_supported_events
from .model import HoloRouteEncoder, HoloRouteOutput

SCORE_FEATURES = (
    "event_reconstruction",
    "path_prediction",
    "depth_prediction",
    "query_prediction",
    "depth_relay_disagreement",
    "diamond_holonomy",
)


def vector_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    huber = F.smooth_l1_loss(
        torch.log1p(prediction),
        torch.log1p(target),
        reduction="none",
    ).mean(dim=-1)
    return cosine.clamp_min(0.0) + huber


def variance_covariance_loss(state: torch.Tensor) -> torch.Tensor:
    if len(state) < 2:
        return state.new_tensor(0.0)
    centered = state - state.mean(dim=0, keepdim=True)
    std = centered.var(dim=0, unbiased=False).add(1e-4).sqrt()
    variance = F.relu(1.0 - std).mean()
    covariance = centered.T @ centered / max(len(state) - 1, 1)
    covariance = covariance - torch.diag(torch.diag(covariance))
    return variance + covariance.square().mean()


@dataclass(frozen=True)
class LossResult:
    total: torch.Tensor
    event: torch.Tensor
    path: torch.Tensor
    depth: torch.Tensor
    query: torch.Tensor
    holonomy: torch.Tensor
    variance: torch.Tensor
    masked_events: int


def _mean_selected(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if bool(mask.any()) else values.new_tensor(0.0)


def self_supervised_loss(
    model: HoloRouteEncoder,
    graph: AttentionEventGraph,
    config,
    *,
    generator: torch.Generator,
) -> LossResult:
    masked = mask_graph_inputs(
        graph,
        event_fraction=config.masking.event_fraction,
        relay_fraction=config.masking.relay_fraction,
        minimum_events=config.masking.minimum_events,
        generator=generator,
    )
    output = model(
        graph,
        event_head_value=masked.head_value,
        event_head_observed=masked.head_observed,
        relay_keep=masked.relay_keep,
        query_event_keep=~masked.event_mask,
    )
    target = graph.event_head_value
    event_error = vector_error(output.event_prediction, target)
    path_error = vector_error(output.path_prediction, target)
    depth_error = vector_error(output.depth_prediction, target)
    query_error = vector_error(output.query_prediction, target)

    event_loss = _mean_selected(event_error, masked.event_mask)
    path_mask = output.relay_coverage & masked.event_mask
    depth_mask = output.depth_coverage & masked.event_mask
    query_mask = output.query_coverage & masked.event_mask
    path_loss = _mean_selected(path_error, path_mask)
    depth_loss = _mean_selected(depth_error, depth_mask)
    query_loss = _mean_selected(query_error, query_mask)
    holonomy_loss = (
        output.holonomy_error.mean()
        if output.holonomy_error.numel()
        else target.new_tensor(0.0)
    )
    variance_loss = variance_covariance_loss(output.state)

    weights = config.loss
    total = (
        weights.event_weight * event_loss
        + weights.path_weight * path_loss
        + weights.depth_weight * depth_loss
        + weights.query_weight * query_loss
        + weights.holonomy_weight * holonomy_loss
        + weights.variance_weight * variance_loss
    )
    return LossResult(
        total=total,
        event=event_loss,
        path=path_loss,
        depth=depth_loss,
        query=query_loss,
        holonomy=holonomy_loss,
        variance=variance_loss,
        masked_events=int(masked.event_mask.sum().item()),
    )


def _aggregate_event_values(
    graph: AttentionEventGraph,
    values: torch.Tensor,
    available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = graph.num_response_tokens
    total = values.new_zeros(tokens)
    count = values.new_zeros(tokens)
    selected = available & torch.isfinite(values)
    if bool(selected.any()):
        token = graph.event_query[selected]
        total.index_add_(0, token, values[selected])
        count.index_add_(0, token, torch.ones_like(values[selected]))
    output = torch.full((tokens,), torch.nan, dtype=values.dtype, device=values.device)
    valid = count > 0
    output[valid] = total[valid] / count[valid]
    return output, count


def _aggregate_holonomy(
    graph: AttentionEventGraph,
    output: HoloRouteOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = graph.num_response_tokens
    value = output.holonomy_error.new_full((tokens,), torch.nan)
    count = output.holonomy_error.new_zeros(tokens)
    if output.holonomy_error.numel():
        total = output.holonomy_error.new_zeros(tokens)
        total.index_add_(0, output.holonomy_token, output.holonomy_error)
        count.index_add_(0, output.holonomy_token, torch.ones_like(output.holonomy_error))
        valid = count > 0
        value[valid] = total[valid] / count[valid]
    return value, count


@torch.no_grad()
def score_graph(
    model: HoloRouteEncoder,
    graph: AttentionEventGraph,
    config,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leave-one-event-out token mechanism errors and coverage counts."""

    model.eval()
    events = graph.num_events
    generator = torch.Generator(device=graph.device)
    generator.manual_seed(int(seed))
    eligible = torch.nonzero(
        structurally_supported_events(graph),
        as_tuple=False,
    ).flatten()
    if len(eligible) == 0:
        eligible = torch.arange(events, device=graph.device)
    order = eligible[
        torch.randperm(len(eligible), generator=generator, device=graph.device)
    ]
    batch_size = max(
        config.masking.minimum_events,
        int(round(len(order) * config.masking.event_fraction)),
    )
    rounds = max(config.masking.score_rounds, int(np.ceil(len(order) / batch_size)))

    sums = graph.event_head_value.new_zeros((events, 5))
    counts = graph.event_head_value.new_zeros((events, 5))
    for round_index in range(rounds):
        begin = (round_index * batch_size) % len(order)
        selected = order[begin : begin + batch_size]
        if len(selected) < batch_size:
            selected = torch.cat((selected, order[: batch_size - len(selected)]))
        event_mask = torch.zeros(events, dtype=torch.bool, device=graph.device)
        event_mask[selected] = True
        value = graph.event_head_value.clone()
        observed = graph.event_head_observed.clone()
        value[event_mask] = 0.0
        observed[event_mask] = False
        output = model(
            graph,
            event_head_value=value,
            event_head_observed=observed,
            query_event_keep=~event_mask,
        )
        target = graph.event_head_value
        errors = (
            vector_error(output.event_prediction, target),
            vector_error(output.path_prediction, target),
            vector_error(output.depth_prediction, target),
            vector_error(output.query_prediction, target),
            output.context_disagreement,
        )
        available = (
            event_mask,
            event_mask & output.relay_coverage,
            event_mask & output.depth_coverage,
            event_mask & output.query_coverage,
            event_mask & output.relay_coverage & output.depth_coverage,
        )
        for index, (error, mask) in enumerate(zip(errors, available)):
            sums[mask, index] += error[mask]
            counts[mask, index] += 1.0

    event_feature = sums / counts.clamp_min(1.0)
    output = model(graph)
    holonomy_token, holonomy_count = _aggregate_holonomy(graph, output)
    token_features: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    for index in range(5):
        token_value, token_count = _aggregate_event_values(
            graph,
            event_feature[:, index],
            counts[:, index] > 0,
        )
        token_features.append(token_value)
        token_counts.append(token_count)
    token_features.append(holonomy_token)
    token_counts.append(holonomy_count)
    feature = torch.stack(token_features, dim=-1)
    coverage = torch.stack(token_counts, dim=-1)
    return feature.cpu().numpy().astype(np.float32), coverage.cpu().numpy().astype(np.float32)
