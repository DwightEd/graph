"""Self-supervised graph views, losses and model training."""

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import HoloRouteConfig
from .graph import EventGraph, build_graph
from .model import DEPTH, EVENT, QUERY, RELAY, HoloRoute


@dataclass(frozen=True)
class GraphView:
    values: torch.Tensor
    observed: torch.Tensor
    event_mask: torch.Tensor
    relay_keep: torch.Tensor
    relay_targets: torch.Tensor
    query_keep: torch.Tensor


@dataclass(frozen=True)
class TrainingResult:
    state_dict: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    best_validation_loss: float


def sample_seed(seed: int, sample_id: str, stream: int = 0) -> int:
    payload = f"{seed}\0{stream}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def autocast_context(model: torch.nn.Module, enabled: bool):
    device = next(model.parameters()).device
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def supported_events(graph: EventGraph) -> torch.Tensor:
    supported = torch.zeros(graph.event_count, dtype=torch.bool, device=graph.device)
    if graph.depth_edges.shape[1]:
        supported[graph.depth_edges[1]] = True
    if graph.relay_edges.shape[1]:
        supported[graph.relay_edges[1]] = True
    for group in range(graph.queries.count):
        members = graph.queries.members(group)
        if len(members) > 1:
            supported[members] = True
    return supported


def choose_events(
    eligible: torch.Tensor,
    fraction: float,
    minimum: int,
    generator: torch.Generator,
) -> torch.Tensor:
    selected = torch.zeros_like(eligible)
    index = torch.nonzero(eligible, as_tuple=False).flatten()
    if not len(index):
        return selected
    count = min(len(index), max(int(round(len(index) * fraction)), minimum))
    order = torch.randperm(len(index), generator=generator, device=index.device)
    selected[index[order[:count]]] = True
    return selected


def choose_relay_edges(
    graph: EventGraph,
    fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_count = graph.relay_edges.shape[1]
    keep = torch.ones(edge_count, dtype=torch.bool, device=graph.device)
    targets = torch.zeros(graph.event_count, dtype=torch.bool, device=graph.device)
    if not edge_count:
        return keep, targets

    edge_target = graph.relay_edges[1]
    for target in torch.unique(edge_target).tolist():
        edges = torch.nonzero(edge_target == target, as_tuple=False).flatten()
        if len(edges) < 2:
            continue
        if torch.rand((), generator=generator, device=graph.device) >= fraction:
            continue
        dropped = edges[
            torch.randint(len(edges), (1,), generator=generator, device=graph.device)
        ]
        keep[dropped] = False
        targets[target] = True
    return keep, targets


def create_training_view(
    graph: EventGraph,
    config: HoloRouteConfig,
    generator: torch.Generator,
) -> GraphView:
    eligible = supported_events(graph)
    if not bool(eligible.any()):
        eligible = torch.ones(graph.event_count, dtype=torch.bool, device=graph.device)
    event_mask = choose_events(
        eligible,
        config.train.mask_fraction,
        config.train.minimum_masked_events,
        generator,
    )
    values = graph.events.value.clone()
    observed = graph.events.observed.clone()
    values[event_mask] = 0.0
    observed[event_mask] = False
    relay_keep, relay_targets = choose_relay_edges(
        graph,
        config.train.relay_drop_fraction,
        generator,
    )
    return GraphView(
        values=values,
        observed=observed,
        event_mask=event_mask,
        relay_keep=relay_keep,
        relay_targets=relay_targets,
        query_keep=~event_mask,
    )


def reconstruction_error(
    predicted_value: torch.Tensor,
    predicted_support: torch.Tensor,
    graph: EventGraph,
    config: HoloRouteConfig,
) -> torch.Tensor:
    target = graph.events.value
    observed = graph.events.observed
    observed_float = observed.float()
    count = observed_float.sum(dim=-1).clamp_min(1.0)

    value = F.smooth_l1_loss(
        torch.log1p(predicted_value),
        torch.log1p(target),
        reduction="none",
    )
    value = (value * observed_float).sum(dim=-1) / count

    cosine_prediction = predicted_value * observed_float
    cosine_target = target * observed_float
    cosine = 1.0 - F.cosine_similarity(
        cosine_prediction,
        cosine_target,
        dim=-1,
        eps=1e-8,
    )
    cosine = torch.where(
        observed.any(dim=-1),
        cosine.clamp_min(0.0),
        torch.zeros_like(cosine),
    )

    support = F.binary_cross_entropy_with_logits(
        predicted_support,
        observed_float,
        reduction="none",
    ).mean(dim=-1)

    censored = (~observed).float()
    censored_count = censored.sum(dim=-1).clamp_min(1.0)
    bound = F.relu(predicted_value - graph.attention_floor).square()
    bound = (bound * censored).sum(dim=-1) / censored_count

    return cosine + value + config.loss.support * support + config.loss.censored * bound


def variance_regularizer(state: torch.Tensor) -> torch.Tensor:
    if len(state) < 2:
        return state.new_tensor(0.0)
    centered = state - state.mean(dim=0, keepdim=True)
    standard_deviation = centered.var(dim=0, unbiased=False).add(1e-4).sqrt()
    variance = F.relu(1.0 - standard_deviation).mean()
    covariance = centered.T @ centered / max(len(state) - 1, 1)
    covariance = covariance - torch.diag(torch.diag(covariance))
    return variance + covariance.square().mean()


def selected_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    return values[selected].mean() if bool(selected.any()) else values.new_tensor(0.0)


def self_supervised_loss(
    model: HoloRoute,
    graph: EventGraph,
    config: HoloRouteConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    view = create_training_view(graph, config, generator)
    output = model(
        graph,
        values=view.values,
        observed=view.observed,
        relay_keep=view.relay_keep,
        query_keep=view.query_keep,
    )

    error = tuple(
        reconstruction_error(
            output.predictions.value[:, index],
            output.predictions.support[:, index],
            graph,
            config,
        )
        for index in (EVENT, DEPTH, RELAY, QUERY)
    )

    event_loss = selected_mean(error[EVENT], view.event_mask)
    depth_loss = selected_mean(error[DEPTH], view.event_mask & output.coverage[:, 0])
    relay_loss = selected_mean(error[RELAY], view.relay_targets & output.coverage[:, 1])
    query_loss = selected_mean(error[QUERY], view.event_mask & output.coverage[:, 2])
    holonomy_loss = (
        output.holonomy.mean()
        if output.holonomy.numel()
        else event_loss.new_tensor(0.0)
    )
    variance_loss = variance_regularizer(output.state)

    weight = config.loss
    return (
        weight.event * event_loss
        + weight.depth * depth_loss
        + weight.query * query_loss
        + weight.relay * relay_loss
        + weight.holonomy * holonomy_loss
        + weight.variance * variance_loss
    )


def validation_loss(
    model: HoloRoute,
    dataset,
    sample_ids: tuple[str, ...],
    config: HoloRouteConfig,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    values: list[float] = []
    with torch.no_grad():
        for sample_id in sample_ids:
            sample = dataset[sample_id]
            try:
                graph = build_graph(sample, config.graph).to(device)
                if not graph.event_count:
                    continue
                masks = []
                for mask_index in range(config.train.validation_masks):
                    generator = torch.Generator(device=graph.device)
                    generator.manual_seed(
                        sample_seed(config.train.seed, sample_id, 10_000 + mask_index)
                    )
                    with autocast_context(model, config.train.mixed_precision):
                        loss = self_supervised_loss(model, graph, config, generator)
                    masks.append(float(loss.float().item()))
                values.append(float(np.mean(masks)))
            finally:
                sample.release_attention()
    return float(np.mean(values)) if values else float("inf")


def train_model(
    model: HoloRoute,
    dataset,
    fit_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    config: HoloRouteConfig,
) -> TrainingResult:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    device = next(model.parameters()).device
    use_amp = device.type == "cuda" and config.train.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(config.train.epochs):
        model.train()
        order = list(fit_ids)
        random.Random(config.train.seed + epoch).shuffle(order)
        losses: list[float] = []

        for sample_id in tqdm(order, desc=f"HoloRoute epoch {epoch + 1}", unit="sample"):
            sample = dataset[sample_id]
            try:
                graph = build_graph(sample, config.graph).to(device)
                if not graph.event_count:
                    continue
                generator = torch.Generator(device=graph.device)
                generator.manual_seed(sample_seed(config.train.seed, sample_id, epoch))
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(model, use_amp):
                    loss = self_supervised_loss(model, graph, config, generator)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.train.gradient_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().float().item()))
                del graph, loss
            finally:
                sample.release_attention()

        current_validation = validation_loss(model, dataset, validation_ids, config)
        current_training = float(np.mean(losses)) if losses else float("inf")
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": current_training,
                "validation_loss": current_validation,
            }
        )
        if current_validation < best_validation:
            best_validation = current_validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training produced no event graphs")
    return TrainingResult(best_state, history, best_validation)
