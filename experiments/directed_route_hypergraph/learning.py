"""Label-free recovery of genuinely withheld typed route endpoints."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from experiments.grounded_route.graph import TokenGraph
from experiments.grounded_route.learning import EndpointPairs, matched_negative_edges

from .config import LearningConfig
from .corruption import corrupt_graph
from .flow import ordered_flow
from .layout import endpoint_layout_plan, ordered_endpoint_layout
from .model import DirectedRouteHypergraphEncoder, EncoderOutput


@dataclass(frozen=True)
class EndpointLoss:
    loss: torch.Tensor
    pair_count: int
    heldout_edge_count: int


@dataclass(frozen=True)
class KLLoss:
    loss: torch.Tensor
    raw: torch.Tensor


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
    endpoint: torch.Tensor
    flow: torch.Tensor
    layout: torch.Tensor
    layout_sink: torch.Tensor
    layout_self: torch.Tensor
    layout_external: torch.Tensor
    variance: torch.Tensor
    kl: torch.Tensor
    raw_kl: torch.Tensor
    pair_count: int
    heldout_edge_count: int
    masked_edge_count: int
    masked_mass_total: float
    masked_row_fraction: float
    native_unresolved_mass_mean: float
    masked_mass_mean: float
    effective_kl_weight: float
    active_latent_dimensions: int
    posterior_logvar_mean: float
    layout_candidate_count: int
    layout_row_count: int
    layout_self_row_count: int
    layout_external_row_count: int


def cpu_generator(generator: torch.Generator) -> torch.Generator:
    if generator.device.type == "cpu":
        return generator
    return torch.Generator().manual_seed(generator.initial_seed())


def sample_held_out_endpoints(
    graph: TokenGraph,
    config: LearningConfig,
    generator: torch.Generator,
) -> EndpointPairs:
    """Reuse the role/lag-matched causal non-edge sampler from GroundedRoute."""

    if config.positive_edges_per_graph < 0:
        raise ValueError("positive_edges_per_graph must be non-negative")
    if not 0.0 <= config.holdout_fraction <= 1.0:
        raise ValueError("holdout_fraction must be in [0, 1]")
    if config.negative_count < 1:
        raise ValueError("negative_count must be positive")
    if config.negative_attempt_factor < 1:
        raise ValueError("negative_attempt_factor must be positive")
    fraction_budget = math.ceil(graph.edge_count * config.holdout_fraction)
    budget = min(config.positive_edges_per_graph, fraction_budget)
    return matched_negative_edges(
        graph,
        config.negative_count,
        generator,
        attempt_factor=config.negative_attempt_factor,
        positive_edges_per_graph=budget,
    )


def held_out_endpoint_loss(
    model: DirectedRouteHypergraphEncoder,
    output: EncoderOutput,
    clean_graph: TokenGraph,
    pairs: EndpointPairs,
) -> EndpointLoss:
    """Rank each withheld endpoint above its matched causal non-edge."""

    heldout_edge_count = int(torch.unique(pairs.edge).numel())
    if not pairs.count:
        zero = output.decoder_response_embedding.sum() * 0.0
        return EndpointLoss(zero, 0, 0)

    edge = pairs.edge
    target = clean_graph.edges.target[edge]
    layer = clean_graph.edges.layer[edge]
    head = clean_graph.edges.head[edge]
    positive_score = model.endpoint_score(
        output,
        clean_graph,
        clean_graph.edges.source[edge],
        target,
        layer,
        head,
    )
    negative_score = model.endpoint_score(
        output,
        clean_graph,
        pairs.negative_source,
        target,
        layer,
        head,
    )
    weight = clean_graph.edges.weight[edge].to(
        device=positive_score.device,
        dtype=positive_score.dtype,
    )
    loss = (weight * F.softplus(negative_score - positive_score)).sum()
    loss = loss / weight.sum().clamp_min(1e-12)
    return EndpointLoss(loss, pairs.count, heldout_edge_count)


def variance_regularizer(embedding: torch.Tensor) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    standard_deviation = embedding.var(dim=0, unbiased=False).add(1e-4).sqrt()
    return F.relu(1.0 - standard_deviation).mean()


def variational_kl(
    output: EncoderOutput,
    response_start: int,
    free_bits: float,
) -> KLLoss:
    """Return response-only Gaussian KL with per-dimension free bits."""

    if free_bits < 0:
        raise ValueError("kl_free_bits must be non-negative")
    mean = output.posterior_mean[response_start:]
    log_variance = output.posterior_log_variance[response_start:]
    if not len(mean):
        zero = output.decoder_embedding.sum() * 0.0
        return KLLoss(zero, zero)
    element = 0.5 * (
        mean.square() + log_variance.exp() - 1.0 - log_variance
    )
    raw = element.mean()
    loss = element.mean(dim=0).clamp_min(float(free_bits)).mean()
    return KLLoss(loss, raw)


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
    layout_key = model.layout_key(output.decoder_embedding)
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
    kl_weight: float | None = None,
) -> LossOutput:
    """Recover masked endpoints and optional clean-route auxiliaries."""

    config = LearningConfig() if config is None else config
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    if any(
        weight < 0
        for weight in (
            config.flow_weight,
            config.layout_weight,
            config.variance_weight,
            config.kl_weight,
        )
    ):
        raise ValueError("loss weights must be non-negative")
    if config.kl_free_bits < 0:
        raise ValueError("kl_free_bits must be non-negative")
    effective_kl_weight = config.kl_weight if kl_weight is None else kl_weight
    if effective_kl_weight < 0:
        raise ValueError("effective KL weight must be non-negative")

    clean_graph = graph.canonicalize()
    pairs = sample_held_out_endpoints(clean_graph, config, generator)
    forced_edge = torch.unique(pairs.edge)
    corruption = corrupt_graph(
        clean_graph,
        incidence_dropout=config.incidence_dropout,
        head_dropout=config.head_dropout,
        generator=generator,
        forced_edge=forced_edge,
    )
    output = model(corruption.graph, masked_mass=corruption.masked_mass)
    endpoint = held_out_endpoint_loss(model, output, clean_graph, pairs)
    zero = output.decoder_response_embedding.sum() * 0.0
    flow = (
        flow_consistency_loss(output, clean_graph, model.config.residual_weight)
        if config.flow_weight > 0
        else zero
    )
    if config.layout_order == "ordered":
        layout_order = None
    elif config.layout_order == "reverse":
        layout_order = tuple(reversed(range(clean_graph.layer_count)))
    else:
        raise ValueError("layout_order must be 'ordered' or 'reverse'")
    if config.layout_weight > 0:
        layout_rows = sample_layout_rows(
            clean_graph,
            config.layout_rows_per_graph,
            generator,
            max_elements=config.layout_max_elements,
            max_work_elements=config.layout_max_work_elements,
            layer_order=layout_order,
        )
        layout = endpoint_layout_loss(
            model,
            output,
            clean_graph,
            response_index=layout_rows,
            rows_per_batch=config.layout_rows_per_batch,
            min_mass=config.layout_min_mass,
            max_elements=config.layout_max_elements,
            max_work_elements=config.layout_max_work_elements,
            layer_order=layout_order,
        )
    else:
        layout = _empty_layout_loss(output)
    variance = variance_regularizer(
        output.posterior_mean[clean_graph.response_start :]
    )
    if model.config.latent_mode == "vae":
        kl = variational_kl(
            output,
            clean_graph.response_start,
            config.kl_free_bits,
        )
    else:
        kl = KLLoss(zero, zero)
        effective_kl_weight = 0.0
    loss = (
        endpoint.loss
        + config.flow_weight * flow
        + config.layout_weight * layout.loss
        + config.variance_weight * variance
        + effective_kl_weight * kl.loss
    )
    masked_rows = int(torch.count_nonzero(corruption.masked_mass).item())
    response_mean = output.posterior_mean[clean_graph.response_start :]
    active_latent_dimensions = (
        int((response_mean.var(dim=0, unbiased=False) > 1e-2).sum().item())
        if len(response_mean) > 1
        else 0
    )
    response_log_variance = output.posterior_log_variance[
        clean_graph.response_start :
    ]
    return LossOutput(
        loss=loss,
        endpoint=endpoint.loss,
        flow=flow,
        layout=layout.loss,
        layout_sink=layout.sink,
        layout_self=layout.self_mass,
        layout_external=layout.external_endpoint,
        variance=variance,
        kl=kl.loss,
        raw_kl=kl.raw,
        pair_count=endpoint.pair_count,
        heldout_edge_count=endpoint.heldout_edge_count,
        masked_edge_count=int(corruption.masked_edge.numel()),
        masked_mass_total=float(corruption.masked_mass.sum().item()),
        masked_row_fraction=(
            masked_rows / max(corruption.masked_mass.numel(), 1)
        ),
        native_unresolved_mass_mean=float(clean_graph.unresolved.mean().item()),
        masked_mass_mean=float(corruption.masked_mass.mean().item()),
        effective_kl_weight=float(effective_kl_weight),
        active_latent_dimensions=active_latent_dimensions,
        posterior_logvar_mean=float(response_log_variance.mean().item()),
        layout_candidate_count=layout.candidate_count,
        layout_row_count=layout.row_count,
        layout_self_row_count=layout.self_row_count,
        layout_external_row_count=layout.external_row_count,
    )
