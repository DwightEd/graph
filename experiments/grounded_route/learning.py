"""Label-free exact-endpoint prediction for grounded-route embeddings."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LearningConfig
from .graph import TokenGraph
from .model import GroundedRouteEncoder, lag_bucket


@dataclass(frozen=True)
class EndpointPairs:
    edge: torch.Tensor
    negative_source: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.edge.numel())


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    route: torch.Tensor
    variance: torch.Tensor
    pair_count: int


def observed_endpoint_keys(graph: TokenGraph) -> torch.Tensor:
    row = (
        (graph.edges.layer * graph.head_count + graph.edges.head)
        * graph.response_count
        + graph.edge_response_target
    )
    return row * graph.token_count + graph.edges.source


def sample_positive_edges(
    edge_count: int,
    limit: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Uniformly sample edge indices without allocating an edge-sized permutation."""

    count = min(int(edge_count), int(limit))
    if count < 1:
        return torch.empty(0, dtype=torch.long, device=device)
    if count == edge_count:
        return torch.arange(edge_count, dtype=torch.long, device=device)

    start = edge_count - count
    upper = torch.arange(start + 1, edge_count + 1, device=device)
    draw = torch.floor(
        torch.rand(
            count,
            dtype=torch.float64,
            generator=generator,
            device=device,
        )
        * upper
    ).long()
    chosen: set[int] = set()
    sample: list[int] = []
    for current, candidate in zip(
        range(start, edge_count),
        draw.detach().cpu().tolist(),
        strict=True,
    ):
        selected = current if candidate in chosen else candidate
        chosen.add(selected)
        sample.append(selected)
    return torch.tensor(sample, dtype=torch.long, device=device)


def matched_negative_edges(
    graph: TokenGraph,
    count: int,
    generator: torch.Generator,
    attempt_factor: int = 4,
    positive_edges_per_graph: int = 16_384,
) -> EndpointPairs:
    """Sample causal non-edges with the same role and logarithmic lag bin."""

    if not graph.edge_count or count < 1:
        empty = torch.empty(0, dtype=torch.long)
        return EndpointPairs(empty, empty)

    if generator.device.type == "cpu":
        sampling_generator = generator
    else:
        sampling_generator = torch.Generator().manual_seed(generator.initial_seed())

    anchor = sample_positive_edges(
        graph.edge_count,
        positive_edges_per_graph,
        sampling_generator,
        torch.device("cpu"),
    )
    source = graph.edges.source[anchor]
    target = graph.edges.target[anchor]
    bucket = lag_bucket(target - source, 63)
    minimum_lag = torch.bitwise_left_shift(torch.ones_like(bucket), bucket)
    maximum_lag = minimum_lag * 2 - 1
    prompt = source < graph.response_start

    minimum_source = target - maximum_lag
    maximum_source = target - minimum_lag
    minimum_source = torch.where(
        prompt,
        minimum_source.clamp_min(0),
        minimum_source.clamp_min(graph.response_start),
    )
    maximum_source = torch.where(
        prompt,
        maximum_source.clamp_max(graph.response_start - 1),
        maximum_source.clamp_max(target - 1),
    )
    width = (maximum_source - minimum_source + 1).clamp_min(0)

    attempts = int(count) * int(attempt_factor)
    random = torch.rand(
        (len(anchor), attempts),
        dtype=torch.float64,
        generator=sampling_generator,
    )
    candidate = minimum_source[:, None] + torch.floor(
        random * width[:, None].clamp_min(1)
    ).long()
    candidate_row = (
        (graph.edges.layer[anchor] * graph.head_count + graph.edges.head[anchor])
        * graph.response_count
        + target
        - graph.response_start
    )
    candidate_key = candidate_row[:, None] * graph.token_count + candidate
    observed = observed_endpoint_keys(graph)
    location = torch.searchsorted(observed, candidate_key)
    location = location.clamp_max(len(observed) - 1)
    valid = width[:, None] > 0
    valid = valid & (observed[location] != candidate_key)
    valid = valid & (valid.cumsum(dim=1) <= count)

    edge = anchor[:, None].expand_as(candidate)[valid]
    return EndpointPairs(edge=edge, negative_source=candidate[valid])


def variance_regularizer(embedding: torch.Tensor) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.new_tensor(0.0)
    standard_deviation = embedding.var(dim=0, unbiased=False).add(1e-4).sqrt()
    return F.relu(1.0 - standard_deviation).mean()


def self_supervised_loss(
    model: GroundedRouteEncoder,
    graph: TokenGraph,
    config: LearningConfig | None = None,
    generator: torch.Generator | None = None,
) -> LossOutput:
    config = LearningConfig() if config is None else config
    if generator is None:
        generator = torch.Generator().manual_seed(0)

    graph = graph.canonicalize()
    output = model(graph)
    pairs = matched_negative_edges(
        graph,
        config.negative_count,
        generator,
        config.negative_attempt_factor,
        config.positive_edges_per_graph,
    )
    if pairs.count:
        edge = pairs.edge
        positive = model.endpoint_score(
            output,
            graph,
            graph.edges.source[edge],
            graph.edges.target[edge],
            graph.edges.layer[edge],
            graph.edges.head[edge],
        )
        negative = model.endpoint_score(
            output,
            graph,
            pairs.negative_source,
            graph.edges.target[edge],
            graph.edges.layer[edge],
            graph.edges.head[edge],
        )
        pair_loss = F.softplus(negative - positive)
        importance = graph.edges.weight[edge].to(device=pair_loss.device)
        route = (pair_loss * importance).sum() / importance.sum().clamp_min(1e-12)
    else:
        route = output.response_embedding.sum() * 0.0

    variance = variance_regularizer(output.response_embedding)
    loss = route + config.variance_weight * variance
    return LossOutput(loss=loss, route=route, variance=variance, pair_count=pairs.count)
