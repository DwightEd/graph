"""Label-free endpoint prediction for the copied DBGNN encoder."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from experiments.grounded_route.artifacts import EncodedTokenGraph

from .config import DBGNNConfig
from .graph import DBGNNGraph, build_dbgnn_graph


@dataclass(frozen=True)
class EndpointPairs:
    """Held-out edges and role/lag-matched unretained endpoints."""

    endpoint_index: torch.Tensor
    negative_source: torch.Tensor
    eligible_count: int

    @property
    def count(self) -> int:
        return int(self.endpoint_index.numel())


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    route_loss: torch.Tensor
    variance_loss: torch.Tensor
    positive_count: int
    eligible_count: int


def _lag_bucket(lag: int) -> int:
    return int(lag).bit_length() - 1


def _cpu_generator(generator: torch.Generator) -> torch.Generator:
    if generator.device.type == "cpu":
        return generator
    return torch.Generator().manual_seed(generator.initial_seed())


def _sample_pairs(
    graph: DBGNNGraph,
    config: DBGNNConfig,
    generator: torch.Generator,
) -> EndpointPairs:
    endpoints = graph.edge_index_fo.T.detach().cpu().tolist()
    observed = {tuple(pair) for pair in endpoints}
    terminal = {target for _, target in endpoints}
    indegree: dict[int, int] = {}
    for _, target in endpoints:
        indegree[target] = indegree.get(target, 0) + 1
    candidates: list[tuple[int, list[int]]] = []

    for index, (source, target) in enumerate(endpoints):
        if (
            source < graph.response_start
            or source not in terminal
            or indegree[target] < 2
        ):
            continue
        bucket = _lag_bucket(target - source)
        negative = [
            candidate
            for candidate in range(graph.response_start, target)
            if _lag_bucket(target - candidate) == bucket
            and (candidate, target) not in observed
            and candidate in terminal
        ]
        if negative:
            candidates.append((index, negative))

    desired = min(
        config.positives_per_graph,
        int(len(endpoints) * config.edge_drop_fraction + 0.999999),
        len(candidates),
    )
    if desired < 1:
        empty = torch.empty(0, dtype=torch.long)
        return EndpointPairs(empty, empty, len(candidates))

    sampling_generator = _cpu_generator(generator)
    order = torch.randperm(len(candidates), generator=sampling_generator)[:desired]
    endpoint_index: list[int] = []
    negative_source: list[int] = []
    for selected in order.tolist():
        index, choices = candidates[selected]
        choice = torch.randint(
            len(choices),
            (),
            generator=sampling_generator,
        ).item()
        endpoint_index.append(index)
        negative_source.append(choices[choice])

    return EndpointPairs(
        endpoint_index=torch.tensor(endpoint_index, dtype=torch.long),
        negative_source=torch.tensor(negative_source, dtype=torch.long),
        eligible_count=len(candidates),
    )


def _without_endpoint_pairs(
    graph: EncodedTokenGraph,
    endpoint: torch.Tensor,
) -> EncodedTokenGraph:
    node_count = int(graph.token_ids.numel())
    held_out = endpoint[0] * node_count + endpoint[1]
    typed_endpoint = graph.edge_index[0] * node_count + graph.edge_index[1]
    keep = ~torch.isin(typed_endpoint, held_out.to(typed_endpoint.device))
    return replace(
        graph,
        edge_index=graph.edge_index[:, keep],
        edge_layer=graph.edge_layer[keep],
        edge_head=graph.edge_head[keep],
        edge_weight=graph.edge_weight[keep],
    )


def _model_device(model) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def _variance_loss(embedding: torch.Tensor) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    standard_deviation = embedding.var(dim=0, unbiased=False).add(1e-4).sqrt()
    return F.relu(1.0 - standard_deviation).mean()


def self_supervised_loss(
    model,
    encoded_graph: EncodedTokenGraph,
    config: DBGNNConfig,
    generator: torch.Generator,
) -> LossOutput:
    """Predict held-out causal endpoints without reading hallucination labels."""

    full_graph = build_dbgnn_graph(
        encoded_graph,
        delta_layers=config.delta_layers,
        higher_order_mode="no_transition",
    )
    pairs = _sample_pairs(full_graph, config, generator)
    positive_endpoint = full_graph.edge_index_fo[:, pairs.endpoint_index]
    masked = _without_endpoint_pairs(encoded_graph, positive_endpoint)
    model_graph = build_dbgnn_graph(
        masked,
        delta_layers=config.delta_layers,
        higher_order_mode=config.higher_order_mode,
    ).to(_model_device(model))
    embedding = model.encode(model_graph)

    if pairs.count:
        source, target = positive_endpoint.to(embedding.device)
        negative_source = pairs.negative_source.to(embedding.device)
        positive_score = model.edge_score(embedding, source, target)
        negative_score = model.edge_score(embedding, negative_source, target)
        pair_loss = F.softplus(negative_score - positive_score)
        weight = full_graph.edge_weight_fo[pairs.endpoint_index].to(embedding.device)
        route_loss = (pair_loss * weight).sum() / weight.sum().clamp_min(1e-12)
    else:
        route_loss = embedding.sum() * 0.0

    variance_loss = _variance_loss(embedding[model_graph.response_start :])
    loss = route_loss + config.variance_weight * variance_loss
    return LossOutput(
        loss=loss,
        route_loss=route_loss,
        variance_loss=variance_loss,
        positive_count=pairs.count,
        eligible_count=pairs.eligible_count,
    )
