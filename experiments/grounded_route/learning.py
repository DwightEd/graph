"""Label-free route prediction for GroundedRoute node embeddings."""

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
class SelectedRows:
    edge: torch.Tensor
    row: torch.Tensor
    count: int


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    route: torch.Tensor
    variance: torch.Tensor
    pair_count: int
    row_count: int = 0


def edge_rows(graph: TokenGraph) -> torch.Tensor:
    return (
        (graph.edges.layer * graph.head_count + graph.edges.head)
        * graph.response_count
        + graph.edge_response_target
    )


def observed_endpoint_keys(graph: TokenGraph) -> torch.Tensor:
    return edge_rows(graph) * graph.token_count + graph.edges.source


def cpu_generator(generator: torch.Generator) -> torch.Generator:
    if generator.device.type == "cpu":
        return generator
    return torch.Generator().manual_seed(generator.initial_seed())


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


def sample_route_rows(
    graph: TokenGraph,
    limit: int,
    generator: torch.Generator,
) -> SelectedRows:
    """Sample complete attention rows and keep every retained source in each row."""

    if not graph.edge_count or limit < 1:
        empty = torch.empty(0, dtype=torch.long)
        return SelectedRows(empty, empty, 0)

    rows = edge_rows(graph)
    unique, counts = torch.unique_consecutive(rows, return_counts=True)
    row_count = min(int(limit), len(unique))
    random = cpu_generator(generator)
    selected = torch.randperm(len(unique), generator=random)[:row_count]
    selected = selected.sort().values

    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            counts.cumsum(0),
        )
    )
    edge_blocks = [
        torch.arange(
            int(offsets[index].item()),
            int(offsets[index + 1].item()),
            dtype=torch.long,
        )
        for index in selected.tolist()
    ]
    edge = torch.cat(edge_blocks)
    local_row = torch.repeat_interleave(
        torch.arange(row_count, dtype=torch.long),
        counts[selected],
    )
    return SelectedRows(edge=edge, row=local_row, count=row_count)


def matched_negative_sources(
    graph: TokenGraph,
    positive_edge: torch.Tensor,
    count: int,
    generator: torch.Generator,
    attempt_factor: int = 4,
) -> EndpointPairs:
    """Sample causal non-edges with the same source role and logarithmic lag bin."""

    if not len(positive_edge) or count < 1:
        empty = torch.empty(0, dtype=torch.long)
        return EndpointPairs(empty, empty)

    positive_edge = positive_edge.to("cpu")
    random = cpu_generator(generator)
    source = graph.edges.source[positive_edge]
    target = graph.edges.target[positive_edge]
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
    draw = torch.rand(
        (len(positive_edge), attempts),
        dtype=torch.float64,
        generator=random,
    )
    candidate = minimum_source[:, None] + torch.floor(
        draw * width[:, None].clamp_min(1)
    ).long()

    candidate_row = edge_rows(graph)[positive_edge]
    candidate_key = candidate_row[:, None] * graph.token_count + candidate
    observed = observed_endpoint_keys(graph)
    location = torch.searchsorted(observed, candidate_key)
    location = location.clamp_max(len(observed) - 1)

    valid = width[:, None] > 0
    valid &= observed[location] != candidate_key
    valid &= valid.cumsum(dim=1) <= count

    edge = positive_edge[:, None].expand_as(candidate)[valid]
    return EndpointPairs(edge=edge, negative_source=candidate[valid])


def matched_negative_edges(
    graph: TokenGraph,
    count: int,
    generator: torch.Generator,
    attempt_factor: int = 4,
    positive_edges_per_graph: int = 16_384,
) -> EndpointPairs:
    """Sample matched negatives for a uniform subset of retained edges."""

    random = cpu_generator(generator)
    positive = sample_positive_edges(
        graph.edge_count,
        positive_edges_per_graph,
        random,
        torch.device("cpu"),
    )
    return matched_negative_sources(
        graph,
        positive,
        count,
        random,
        attempt_factor,
    )


def segment_log_softmax(
    score: torch.Tensor,
    group: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    maximum = score.new_full((group_count,), -torch.inf)
    maximum.scatter_reduce_(
        0,
        group,
        score,
        reduce="amax",
        include_self=True,
    )
    shifted = score - maximum[group]
    normalizer = score.new_zeros(group_count)
    normalizer.index_add_(0, group, shifted.exp())
    return shifted - normalizer[group].clamp_min(1e-12).log()


def variance_regularizer(embedding: torch.Tensor) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.new_tensor(0.0)
    standard_deviation = embedding.var(dim=0, unbiased=False).add(1e-4).sqrt()
    return F.relu(1.0 - standard_deviation).mean()


def pairwise_endpoint_loss(
    model: GroundedRouteEncoder,
    output,
    graph: TokenGraph,
    config: LearningConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    pairs = matched_negative_edges(
        graph,
        config.negative_count,
        generator,
        config.negative_attempt_factor,
        config.positive_edges_per_graph,
    )
    if not pairs.count:
        return output.response_embedding.sum() * 0.0, 0

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
    importance = graph.edges.weight[edge].to(pair_loss.device)
    route = (pair_loss * importance).sum() / importance.sum().clamp_min(1e-12)
    return route, pairs.count


def row_distribution_loss(
    model: GroundedRouteEncoder,
    output,
    graph: TokenGraph,
    config: LearningConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int, int]:
    """Predict the full retained source distribution of sampled layer-head rows."""

    selected = sample_route_rows(
        graph,
        config.route_rows_per_graph,
        generator,
    )
    if not selected.count:
        zero = output.response_embedding.sum() * 0.0
        return zero, 0, 0

    device = output.node_embedding.device
    positive_edge = selected.edge
    positive_row = selected.row.to(device)
    positive_score = model.endpoint_score(
        output,
        graph,
        graph.edges.source[positive_edge],
        graph.edges.target[positive_edge],
        graph.edges.layer[positive_edge],
        graph.edges.head[positive_edge],
    )

    negatives = matched_negative_sources(
        graph,
        positive_edge,
        config.negative_count,
        generator,
        config.negative_attempt_factor,
    )
    if negatives.count:
        row_lookup = torch.full(
            (graph.edge_count,),
            -1,
            dtype=torch.long,
        )
        row_lookup[positive_edge] = selected.row
        negative_row = row_lookup[negatives.edge].to(device)
        negative_score = model.endpoint_score(
            output,
            graph,
            negatives.negative_source,
            graph.edges.target[negatives.edge],
            graph.edges.layer[negatives.edge],
            graph.edges.head[negatives.edge],
        )
        candidate_score = torch.cat((positive_score, negative_score))
        candidate_row = torch.cat((positive_row, negative_row))
    else:
        candidate_score = positive_score
        candidate_row = positive_row

    log_probability = segment_log_softmax(
        candidate_score,
        candidate_row,
        selected.count,
    )
    positive_weight = graph.edges.weight[positive_edge].to(device)
    row_mass = positive_weight.new_zeros(selected.count)
    row_mass.index_add_(0, positive_row, positive_weight)
    target_probability = positive_weight / row_mass[positive_row].clamp_min(1e-12)

    row_loss = positive_weight.new_zeros(selected.count)
    row_loss.index_add_(
        0,
        positive_row,
        -target_probability * log_probability[: len(positive_edge)],
    )
    return (
        row_loss.mean(),
        len(positive_edge) + negatives.count,
        selected.count,
    )


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
    if config.objective == "row_distribution":
        route, pair_count, row_count = row_distribution_loss(
            model,
            output,
            graph,
            config,
            generator,
        )
    else:
        route, pair_count = pairwise_endpoint_loss(
            model,
            output,
            graph,
            config,
            generator,
        )
        row_count = 0

    variance = variance_regularizer(output.response_embedding)
    loss = route + config.variance_weight * variance
    return LossOutput(
        loss=loss,
        route=route,
        variance=variance,
        pair_count=pair_count,
        row_count=row_count,
    )
