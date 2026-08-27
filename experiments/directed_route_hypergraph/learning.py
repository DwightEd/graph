"""Censored attention-row mass modeling without fabricated non-edge labels."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.grounded_route.graph import TokenGraph

from .config import LearningConfig
from .corruption import corrupt_graph
from .flow import ordered_flow
from .layout import endpoint_layout_plan, ordered_endpoint_layout
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
class LayoutLoss:
    """Balanced complete endpoint-layout reconstruction objective."""

    loss: torch.Tensor
    sink: torch.Tensor
    self_mass: torch.Tensor
    external_endpoint: torch.Tensor
    candidate_count: int
    row_count: int
    self_row_count: int
    external_row_count: int


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    row: torch.Tensor
    flow: torch.Tensor
    layout: torch.Tensor
    layout_sink: torch.Tensor
    layout_self: torch.Tensor
    layout_external: torch.Tensor
    variance: torch.Tensor
    candidate_count: int
    row_count: int
    layout_candidate_count: int
    layout_row_count: int
    layout_self_row_count: int
    layout_external_row_count: int


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


def flow_consistency_loss(
    output: EncoderOutput,
    clean_graph: TokenGraph,
    residual_weight: float,
) -> torch.Tensor:
    """Recover clean ordered P/R/U path flow from the student graph."""

    target = ordered_flow(
        clean_graph,
        residual_weight=residual_weight,
    ).token_trace.to(
        device=output.flow_logits.device,
        dtype=output.flow_logits.dtype,
    )
    return -(
        target.detach() * F.log_softmax(output.flow_logits, dim=-1)
    ).sum(dim=-1).mean()


def _layout_plan_fits(
    graph: TokenGraph,
    response_index: torch.Tensor,
    *,
    max_elements: int,
    max_work_elements: int,
    layer_order: tuple[int, ...] | None,
) -> bool:
    if max_elements < 1:
        raise ValueError("layout max_elements must be positive")
    if max_work_elements < 1:
        raise ValueError("layout max_work_elements must be positive")
    plan = endpoint_layout_plan(
        graph,
        response_index,
        layer_order=layer_order,
    )
    target_elements = len(plan.response_index) * (graph.token_count + 1)
    element_count = max(target_elements, plan.peak_state_elements)
    return (
        element_count <= max_elements
        and plan.work_element_count <= max_work_elements
    )


def sample_layout_rows(
    graph: TokenGraph,
    limit: int,
    generator: torch.Generator,
    *,
    max_elements: int,
    max_work_elements: int,
    layer_order: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Choose an exact endpoint-layout subset that fits explicit budgets.

    Rows are sampled uniformly without replacement.  If their exact backward
    dependency closure is too large, the same random priority order is halved
    until it fits.  A first-response fallback keeps the objective active on
    exceptionally long graphs; if even that exact row violates an explicitly
    tiny budget, the layout objective is skipped for that graph while local-row
    and P/R/U losses remain active.
    """

    count = min(max(int(limit), 0), graph.response_count)
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    order = torch.randperm(
        graph.response_count,
        generator=cpu_generator(generator),
    )
    current = count
    while current:
        selected = order[:current].sort().values
        if _layout_plan_fits(
            graph,
            selected,
            max_elements=max_elements,
            max_work_elements=max_work_elements,
            layer_order=layer_order,
        ):
            return selected
        if current == 1:
            break
        current = max(1, current // 2)

    fallback = torch.zeros(1, dtype=torch.long)
    if _layout_plan_fits(
        graph,
        fallback,
        max_elements=max_elements,
        max_work_elements=max_work_elements,
        layer_order=layer_order,
    ):
        return fallback
    return torch.empty(0, dtype=torch.long)


def _empty_layout_loss(output: EncoderOutput) -> LayoutLoss:
    zero = output.response_embedding.sum() * 0.0
    return LayoutLoss(
        loss=zero,
        sink=zero,
        self_mass=zero,
        external_endpoint=zero,
        candidate_count=0,
        row_count=0,
        self_row_count=0,
        external_row_count=0,
    )


def endpoint_layout_loss(
    model: DirectedRouteHypergraphEncoder,
    output: EncoderOutput,
    clean_graph: TokenGraph,
    *,
    response_index: torch.Tensor | None = None,
    rows_per_batch: int = 64,
    min_mass: float = 1e-4,
    max_elements: int = 8_000_000,
    max_work_elements: int = 250_000_000,
    layer_order: tuple[int, ...] | None = None,
) -> LayoutLoss:
    """Recover clean full-path endpoint layouts for exact selected rows.

    The target is the layer-ordered attention-flow proxy over exact endpoints,
    not a hallucination label and not an OV-aware contribution layout.  An
    arbitrary response subset is computed through its exact layer-wise
    dependency closure, so selected rows equal the corresponding rows of the
    full layout without materializing unrelated response states.
    """

    if rows_per_batch < 1:
        raise ValueError("layout rows_per_batch must be positive")
    if not 0.0 < min_mass < 1.0:
        raise ValueError("layout min_mass must be between zero and one")
    if response_index is None:
        response_index = torch.arange(clean_graph.response_count, dtype=torch.long)
    else:
        response_index = torch.unique(
            torch.as_tensor(response_index, dtype=torch.long).detach().cpu(),
            sorted=True,
        )
    if len(response_index) and bool(
        ((response_index < 0) | (response_index >= clean_graph.response_count)).any()
    ):
        raise ValueError("layout response_index is outside the graph")
    if not len(response_index):
        return _empty_layout_loss(output)

    plan = endpoint_layout_plan(
        clean_graph,
        response_index,
        layer_order=layer_order,
    )
    target_elements = len(response_index) * (clean_graph.token_count + 1)
    element_count = max(target_elements, plan.peak_state_elements)
    if max_elements < 1:
        raise ValueError("layout max_elements must be positive")
    if element_count > max_elements:
        raise ValueError(
            f"selected endpoint layout requires {element_count} dense elements, "
            f"exceeding layout_max_elements={max_elements}; reduce "
            "layout_rows_per_graph or set layout_weight=0"
        )
    if max_work_elements < 1:
        raise ValueError("layout max_work_elements must be positive")
    if plan.work_element_count > max_work_elements:
        raise ValueError(
            f"selected endpoint relay work estimate {plan.work_element_count} "
            f"exceeds layout_max_work_elements={max_work_elements}; reduce "
            "layout_rows_per_graph or set layout_weight=0"
        )

    target = ordered_endpoint_layout(
        clean_graph,
        residual_weight=model.config.residual_weight,
        layer_order=layer_order,
        response_index=response_index,
    ).distribution.to(
        device=output.response_embedding.device,
        dtype=output.response_embedding.dtype,
    )
    expected_shape = (len(response_index), clean_graph.token_count + 1)
    if target.shape != expected_shape:
        raise ValueError("endpoint layout has invalid dimensions")
    if not torch.isfinite(target).all() or not torch.allclose(
        target.sum(dim=1),
        torch.ones(len(response_index), device=target.device, dtype=target.dtype),
        atol=2e-5,
        rtol=2e-5,
    ):
        raise ValueError("endpoint layout must contain finite probability rows")

    response_index = response_index.to(target.device)
    zero = output.response_embedding.sum() * 0.0
    sink_total = zero
    self_total = zero
    external_total = zero
    self_rows = 0
    external_rows = 0
    layout_key = model.layout_key(output.node_embedding)
    for start in range(0, len(response_index), rows_per_batch):
        stop = min(start + rows_per_batch, len(response_index))
        current_index = response_index[start:stop]
        logits = model.endpoint_layout_logits(
            output,
            clean_graph,
            current_index,
            key=layout_key,
        )
        token_logits = logits[:, :-1]
        target_block = target[start:stop]
        target_token = clean_graph.response_start + current_index
        block_row = torch.arange(stop - start, device=target.device)

        all_normalizer = torch.logsumexp(logits, dim=1)
        token_normalizer = torch.logsumexp(token_logits, dim=1)
        sink_log_probability = logits[:, -1] - all_normalizer
        known_log_probability = token_normalizer - all_normalizer
        sink_mass = target_block[:, -1]
        sink_total = sink_total - (
            sink_mass * sink_log_probability
            + (1.0 - sink_mass) * known_log_probability
        ).sum()

        known_mass = 1.0 - sink_mass
        self_mass = target_block[block_row, target_token]
        known = known_mass > min_mass
        if bool(known.any()):
            token_log_probability = token_logits - token_normalizer[:, None]
            self_log_probability = token_log_probability[block_row, target_token]
            external_logits = token_logits.clone()
            external_logits[block_row, target_token] = torch.finfo(
                token_logits.dtype
            ).min
            external_log_probability = (
                torch.logsumexp(external_logits, dim=1) - token_normalizer
            )
            conditional_self = (self_mass / known_mass.clamp_min(min_mass)).clamp(0, 1)
            self_total = self_total - (
                conditional_self[known] * self_log_probability[known]
                + (1.0 - conditional_self[known])
                * external_log_probability[known]
            ).sum()
            self_rows += int(known.sum().item())

            external_mass = known_mass - self_mass
            external = external_mass > min_mass
            if bool(external.any()):
                external_target = target_block[:, :-1].clone()
                external_target[block_row, target_token] = 0.0
                external_target = external_target / external_mass.clamp_min(
                    min_mass
                )[:, None]
                conditional_external_log_probability = F.log_softmax(
                    external_logits,
                    dim=1,
                )
                external_energy = -(
                    external_target[external]
                    * conditional_external_log_probability[external]
                ).sum(dim=1)
                external_candidates = target_token[external].float().clamp_min(2.0)
                external_total = external_total + (
                    external_energy / external_candidates.log()
                ).sum()
                external_rows += int(external.sum().item())

    row_count = len(response_index)
    sink_loss = sink_total / max(row_count, 1)
    self_loss = self_total / max(self_rows, 1)
    external_loss = external_total / max(external_rows, 1)
    candidate_count = sum(
        clean_graph.response_start + int(response) + 2
        for response in response_index.detach().cpu().tolist()
    )
    return LayoutLoss(
        loss=sink_loss + self_loss + external_loss,
        sink=sink_loss,
        self_mass=self_loss,
        external_endpoint=external_loss,
        candidate_count=candidate_count,
        row_count=row_count,
        self_row_count=self_rows,
        external_row_count=external_rows,
    )


def self_supervised_loss(
    model: DirectedRouteHypergraphEncoder,
    graph: TokenGraph,
    config: LearningConfig | None = None,
    generator: torch.Generator | None = None,
) -> LossOutput:
    """Denoise clean row mass and ordered path flow without labels."""

    config = LearningConfig() if config is None else config
    if generator is None:
        generator = torch.Generator().manual_seed(0)

    # Keep the supervised row subset matched when corruption rates are ablated.
    selected = sample_rows(graph, config.rows_per_graph, generator)
    student_graph = corrupt_graph(
        graph,
        incidence_dropout=config.incidence_dropout,
        head_dropout=config.head_dropout,
        generator=generator,
    )
    output = model(student_graph, return_layer_input=True)
    row = row_distribution_loss(model, output, graph, selected)
    if config.flow_weight < 0 or config.layout_weight < 0:
        raise ValueError("flow_weight and layout_weight must be non-negative")
    zero = output.response_embedding.sum() * 0.0
    flow = (
        flow_consistency_loss(output, graph, model.config.residual_weight)
        if config.flow_weight > 0
        else zero
    )
    if config.layout_order == "ordered":
        layout_order = None
    elif config.layout_order == "reverse":
        layout_order = tuple(reversed(range(graph.layer_count)))
    else:
        raise ValueError("layout_order must be 'ordered' or 'reverse'")
    if config.layout_weight > 0:
        layout_rows = sample_layout_rows(
            graph,
            config.layout_rows_per_graph,
            generator,
            max_elements=config.layout_max_elements,
            max_work_elements=config.layout_max_work_elements,
            layer_order=layout_order,
        )
        layout = endpoint_layout_loss(
            model,
            output,
            graph,
            response_index=layout_rows,
            rows_per_batch=config.layout_rows_per_batch,
            min_mass=config.layout_min_mass,
            max_elements=config.layout_max_elements,
            max_work_elements=config.layout_max_work_elements,
            layer_order=layout_order,
        )
    else:
        layout = _empty_layout_loss(output)
    variance = variance_regularizer(output.response_embedding)
    loss = (
        row.loss
        + config.flow_weight * flow
        + config.layout_weight * layout.loss
        + config.variance_weight * variance
    )
    return LossOutput(
        loss=loss,
        row=row.loss,
        flow=flow,
        layout=layout.loss,
        layout_sink=layout.sink,
        layout_self=layout.self_mass,
        layout_external=layout.external_endpoint,
        variance=variance,
        candidate_count=row.candidate_count,
        row_count=row.row_count,
        layout_candidate_count=layout.candidate_count,
        layout_row_count=layout.row_count,
        layout_self_row_count=layout.self_row_count,
        layout_external_row_count=layout.external_row_count,
    )