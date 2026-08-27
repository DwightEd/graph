"""Censored attention-row mass modeling without fabricated non-edge labels."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.grounded_route.graph import TokenGraph

from .config import LearningConfig
from .model import DirectedRouteHypergraphEncoder, EncoderOutput


@dataclass(frozen=True)
class SelectedRows:
    """Canonical ``(layer, head, response target)`` row identifiers."""

    row: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.row.numel())


@dataclass(frozen=True)
class RowCandidates:
    """All retained endpoints plus SELF and UNRESOLVED for selected rows."""

    source: torch.Tensor
    endpoint_row: torch.Tensor
    target: torch.Tensor
    layer: torch.Tensor
    head: torch.Tensor
    group: torch.Tensor
    weight: torch.Tensor

    @property
    def endpoint_count(self) -> int:
        return int(self.source.numel())

    @property
    def count(self) -> int:
        return int(self.weight.numel())

    @property
    def row_count(self) -> int:
        return int(self.target.numel())


@dataclass(frozen=True)
class RowLoss:
    loss: torch.Tensor
    candidate_count: int
    row_count: int


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    row: torch.Tensor
    variance: torch.Tensor
    candidate_count: int
    row_count: int


def cpu_generator(generator: torch.Generator) -> torch.Generator:
    if generator.device.type == "cpu":
        return generator
    return torch.Generator().manual_seed(generator.initial_seed())


def sample_rows(
    graph: TokenGraph,
    limit: int,
    generator: torch.Generator,
) -> SelectedRows:
    """Uniformly sample from all ``R * L * H`` rows, including empty rows."""

    total = graph.response_count * graph.layer_count * graph.head_count
    count = min(max(int(limit), 0), total)
    if count == total:
        row = torch.arange(total, dtype=torch.long)
    else:
        row = torch.randperm(total, generator=cpu_generator(generator))[:count]
        row = row.sort().values
    return SelectedRows(row=row)


def edge_rows(graph: TokenGraph) -> torch.Tensor:
    """Return canonical global row IDs for retained incidences."""

    return (
        (graph.edges.layer * graph.head_count + graph.edges.head)
        * graph.response_count
        + graph.edge_response_target
    )


def row_candidates(graph: TokenGraph, selected: SelectedRows) -> RowCandidates:
    """Materialize the exact censored categorical target for selected rows."""

    device = graph.device
    # TokenGraph keeps sparse endpoints on CPU and dense row mass on graph.device.
    row = selected.row
    channel = torch.div(row, graph.response_count, rounding_mode="floor")
    response = row.remainder(graph.response_count)
    layer = torch.div(channel, graph.head_count, rounding_mode="floor")
    head = channel.remainder(graph.head_count)
    target = graph.response_start + response

    retained_row = edge_rows(graph)
    location = torch.searchsorted(row, retained_row)
    lookup = location.clamp_max(max(selected.count - 1, 0))
    keep = location < selected.count
    if selected.count:
        keep &= row[lookup] == retained_row

    endpoint_row = location[keep].to(device)
    source = graph.edges.source[keep].to(device)
    endpoint_weight = graph.edges.weight[keep].to(device)
    local_row = torch.arange(selected.count, device=device)
    response_device = response.to(device)
    layer_device = layer.to(device)
    head_device = head.to(device)
    self_weight = graph.diagonal[
        response_device,
        layer_device,
        head_device,
    ]
    unresolved_weight = graph.unresolved[
        response_device,
        layer_device,
        head_device,
    ]

    return RowCandidates(
        source=source,
        endpoint_row=endpoint_row,
        target=target.to(device),
        layer=layer_device,
        head=head_device,
        group=torch.cat((endpoint_row, local_row, local_row)),
        weight=torch.cat((endpoint_weight, self_weight, unresolved_weight)),
    )


def segment_log_softmax(
    score: torch.Tensor,
    group: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    """Compute a log softmax independently inside every selected row."""

    maximum = score.new_full((group_count,), -torch.inf)
    maximum.scatter_reduce_(0, group, score, reduce="amax", include_self=True)
    shifted = score - maximum[group]
    normalizer = score.new_zeros(group_count)
    normalizer.index_add_(0, group, shifted.exp())
    return shifted - normalizer[group].clamp_min(1e-12).log()


def row_distribution_loss(
    model: DirectedRouteHypergraphEncoder,
    output: EncoderOutput,
    graph: TokenGraph,
    selected: SelectedRows,
) -> RowLoss:
    """Rank mass on known retained support and two buckets from pre-consume state."""

    candidates = row_candidates(graph, selected)
    if not selected.count:
        zero = output.response_embedding.sum() * 0.0
        return RowLoss(zero, 0, 0)

    endpoint_target = candidates.target[candidates.endpoint_row]
    endpoint_layer = candidates.layer[candidates.endpoint_row]
    endpoint_head = candidates.head[candidates.endpoint_row]
    endpoint_score = model.endpoint_score(
        output,
        graph,
        candidates.source,
        endpoint_target,
        endpoint_layer,
        endpoint_head,
    )
    self_score = model.bucket_score(
        output,
        graph,
        candidates.target,
        candidates.layer,
        candidates.head,
        "self",
    )
    unresolved_score = model.bucket_score(
        output,
        graph,
        candidates.target,
        candidates.layer,
        candidates.head,
        "unresolved",
    )
    score = torch.cat((endpoint_score, self_score, unresolved_score))
    log_probability = segment_log_softmax(
        score,
        candidates.group,
        selected.count,
    )
    loss = -(candidates.weight * log_probability).sum() / selected.count
    return RowLoss(loss, candidates.count, selected.count)


def variance_regularizer(embedding: torch.Tensor) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    standard_deviation = embedding.var(dim=0, unbiased=False).add(1e-4).sqrt()
    return F.relu(1.0 - standard_deviation).mean()


def self_supervised_loss(
    model: DirectedRouteHypergraphEncoder,
    graph: TokenGraph,
    config: LearningConfig | None = None,
    generator: torch.Generator | None = None,
) -> LossOutput:
    """Fit conditional row mass and regularize the final node geometry."""

    config = LearningConfig() if config is None else config
    if generator is None:
        generator = torch.Generator().manual_seed(0)

    output = model(graph, return_layer_input=True)
    selected = sample_rows(graph, config.rows_per_graph, generator)
    row = row_distribution_loss(model, output, graph, selected)
    variance = variance_regularizer(output.response_embedding)
    loss = row.loss + config.variance_weight * variance
    return LossOutput(
        loss=loss,
        row=row.loss,
        variance=variance,
        candidate_count=row.candidate_count,
        row_count=row.row_count,
    )
