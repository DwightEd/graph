"""Prompt-provenance cuts on a multiplex attention graph.

The method is deterministic and label-free. It first propagates lower and upper
bounds on prompt provenance through the response graph. Response edges are then
split into prompt-rooted, response-closed and unresolved parts. Two matched
counterfactual graph views remove one part at a time while conserving each
attention row's retained mass. The token score compares the resulting routing
state changes.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .config import PCutConfig
from .graph import AttentionGraph


@dataclass(frozen=True)
class Provenance:
    lower_before: torch.Tensor
    upper_before: torch.Tensor
    lower_after: torch.Tensor
    upper_after: torch.Tensor


@dataclass(frozen=True)
class EdgeParts:
    prompt_rooted: torch.Tensor
    response_closed: torch.Tensor
    uncertain: torch.Tensor


@dataclass(frozen=True)
class ViewWeights:
    full: torch.Tensor
    no_prompt: torch.Tensor
    no_closed: torch.Tensor
    no_prompt_unknown: torch.Tensor
    no_closed_unknown: torch.Tensor
    no_prompt_supported: torch.Tensor
    no_closed_supported: torch.Tensor


@dataclass(frozen=True)
class PCutResult:
    prompt_necessity: torch.Tensor
    response_closed_necessity: torch.Tensor
    closure: torch.Tensor
    coverage: torch.Tensor
    token_layer_embedding: torch.Tensor
    token_embedding: torch.Tensor
    no_prompt_embedding: torch.Tensor
    no_closed_embedding: torch.Tensor
    prompt_origin_lower: torch.Tensor
    prompt_origin_upper: torch.Tensor
    edge_parts: EdgeParts
    uncertainty_width: torch.Tensor
    cut_fallback_fraction: torch.Tensor


def row_index(graph: AttentionGraph) -> torch.Tensor:
    response_target = graph.edges.target - graph.response_start
    return (
        (response_target * graph.layer_count + graph.edges.layer)
        * graph.head_count
        + graph.edges.head
    )


def prompt_provenance(graph: AttentionGraph) -> Provenance:
    response = graph.response_count
    heads = graph.head_count
    layers = graph.layer_count
    prompt = graph.response_start
    device = graph.device

    lower_previous = torch.zeros((response, heads), dtype=torch.float32, device=device)
    upper_previous = torch.zeros_like(lower_previous)
    lower_before = torch.empty((layers, response, heads), dtype=torch.float32, device=device)
    upper_before = torch.empty_like(lower_before)
    lower_after = torch.empty((response, layers, heads), dtype=torch.float32, device=device)
    upper_after = torch.empty_like(lower_after)

    for layer in range(layers):
        lower_before[layer] = lower_previous
        upper_before[layer] = upper_previous
        current_lower = graph.diagonal[:, layer] * lower_previous
        current_upper = graph.diagonal[:, layer] * upper_previous + graph.unresolved[:, layer]

        current = graph.edges.layer_slice(layer)
        source = graph.edges.source[current]
        target = graph.edges.target[current] - prompt
        head = graph.edges.head[current]
        weight = graph.edges.weight[current]
        if weight.numel():
            source_is_prompt = source < prompt
            source_response = (source - prompt).clamp_min(0)
            lower_source = torch.where(
                source_is_prompt,
                torch.ones_like(weight),
                lower_previous[source_response, head],
            )
            upper_source = torch.where(
                source_is_prompt,
                torch.ones_like(weight),
                upper_previous[source_response, head],
            )
            flat_target = target * heads + head
            current_lower.view(-1).index_add_(0, flat_target, weight * lower_source)
            current_upper.view(-1).index_add_(0, flat_target, weight * upper_source)

        lower_previous = current_lower.clamp(0.0, 1.0)
        upper_previous = torch.maximum(current_upper.clamp(0.0, 1.0), lower_previous)
        lower_after[:, layer] = lower_previous
        upper_after[:, layer] = upper_previous

    return Provenance(lower_before, upper_before, lower_after, upper_after)


def split_edges(graph: AttentionGraph, provenance: Provenance) -> EdgeParts:
    weight = graph.edges.weight
    source = graph.edges.source
    layer = graph.edges.layer
    head = graph.edges.head
    prompt = graph.response_start
    source_is_prompt = source < prompt
    source_response = (source - prompt).clamp_min(0)

    lower = torch.where(
        source_is_prompt,
        torch.ones_like(weight),
        provenance.lower_before[layer, source_response, head],
    )
    upper = torch.where(
        source_is_prompt,
        torch.ones_like(weight),
        provenance.upper_before[layer, source_response, head],
    )
    rooted = weight * lower
    closed = weight * (1.0 - upper)
    uncertain = weight * (upper - lower)
    return EdgeParts(rooted, closed, uncertain)


def conserve_row_mass(
    graph: AttentionGraph,
    kept: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = graph.response_count * graph.layer_count * graph.head_count
    index = row_index(graph)
    original = graph.edges.weight.new_zeros(rows)
    remaining = graph.edges.weight.new_zeros(rows)
    if graph.edge_count:
        original.index_add_(0, index, graph.edges.weight)
        remaining.index_add_(0, index, kept)

    supported = remaining > epsilon
    scale = torch.zeros_like(remaining)
    scale[supported] = original[supported] / remaining[supported]
    scaled = kept * scale[index]
    fallback = torch.where(supported, torch.zeros_like(original), original)
    shape = (graph.response_count, graph.layer_count, graph.head_count)
    return scaled, fallback.reshape(shape), supported.reshape(shape)


def build_views(
    graph: AttentionGraph,
    parts: EdgeParts,
    config: PCutConfig,
) -> ViewWeights:
    prompt_source = graph.edges.source < graph.response_start
    no_prompt_kept = torch.where(
        prompt_source,
        torch.zeros_like(graph.edges.weight),
        parts.response_closed + parts.uncertain,
    )
    no_closed_kept = parts.prompt_rooted + parts.uncertain
    no_prompt, no_prompt_unknown, no_prompt_supported = conserve_row_mass(
        graph,
        no_prompt_kept,
        config.epsilon,
    )
    no_closed, no_closed_unknown, no_closed_supported = conserve_row_mass(
        graph,
        no_closed_kept,
        config.epsilon,
    )
    return ViewWeights(
        full=graph.edges.weight,
        no_prompt=no_prompt,
        no_closed=no_closed,
        no_prompt_unknown=no_prompt_unknown,
        no_closed_unknown=no_closed_unknown,
        no_prompt_supported=no_prompt_supported,
        no_closed_supported=no_closed_supported,
    )


def sinusoidal_identity(count: int, dimension: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(count, dtype=torch.float32, device=device)[:, None]
    half = max(dimension // 2, 1)
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, dtype=torch.float32, device=device)
        / max(half - 1, 1)
    )[None]
    angle = position * frequency
    embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)
    if embedding.shape[1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[1]))
    return F.normalize(embedding[:, :dimension], dim=-1)


def head_projection(heads: int, components: int, device: torch.device) -> torch.Tensor:
    head = torch.arange(heads, dtype=torch.float32, device=device)[:, None]
    component = torch.arange(components, dtype=torch.float32, device=device)[None]
    projection = torch.cos(math.pi * (head + 0.5) * component / max(heads, 1))
    projection[:, 0] = 1.0
    return F.normalize(projection, dim=0)


def compress_heads(state: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    compressed = torch.einsum("rhd,hk->rkd", state, projection)
    return compressed.flatten(1)


def rollout(
    graph: AttentionGraph,
    edge_weight: torch.Tensor,
    extra_unknown: torch.Tensor,
    config: PCutConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    identity = sinusoidal_identity(
        graph.token_count + 1,
        config.identity_dim,
        graph.device,
    )
    unknown = identity[-1]
    state = identity[: graph.token_count, None].expand(-1, graph.head_count, -1).clone()
    projection = head_projection(
        graph.head_count,
        config.head_projection_dim,
        graph.device,
    )
    layer_embedding = state.new_empty(
        (
            graph.response_count,
            graph.layer_count,
            config.identity_dim * config.head_projection_dim,
        )
    )

    for layer in range(graph.layer_count):
        previous = state
        response_state = (
            graph.diagonal[:, layer, :, None] * previous[graph.response_start :]
            + (graph.unresolved[:, layer] + extra_unknown[:, layer])[:, :, None]
            * unknown[None, None]
        )

        current = graph.edges.layer_slice(layer)
        source = graph.edges.source[current]
        target = graph.edges.target[current] - graph.response_start
        head = graph.edges.head[current]
        weight = edge_weight[current]
        if weight.numel():
            message = previous[source, head] * weight[:, None]
            flat_target = target * graph.head_count + head
            response_state.view(-1, config.identity_dim).index_add_(
                0,
                flat_target,
                message,
            )

        state = previous.clone()
        state[graph.response_start :] = response_state
        layer_embedding[:, layer] = compress_heads(response_state, projection)

    tail = min(config.tail_layers, graph.layer_count)
    token_embedding = layer_embedding[:, -tail:].mean(dim=1)
    return layer_embedding, token_embedding


def cosine_change(full: torch.Tensor, cut: torch.Tensor) -> torch.Tensor:
    full = F.normalize(full.float(), dim=-1)
    cut = F.normalize(cut.float(), dim=-1)
    return (1.0 - (full * cut).sum(dim=-1)).clamp_min(0.0)


@torch.no_grad()
def compute_pcut(
    graph: AttentionGraph,
    config: PCutConfig | None = None,
) -> PCutResult:
    config = PCutConfig() if config is None else config
    provenance = prompt_provenance(graph)
    parts = split_edges(graph, provenance)
    views = build_views(graph, parts, config)

    zero_unknown = torch.zeros_like(graph.unresolved)
    full_layer, full_token = rollout(graph, views.full, zero_unknown, config)
    no_prompt_layer, no_prompt_token = rollout(
        graph,
        views.no_prompt,
        views.no_prompt_unknown,
        config,
    )
    no_closed_layer, no_closed_token = rollout(
        graph,
        views.no_closed,
        views.no_closed_unknown,
        config,
    )

    tail = min(config.tail_layers, graph.layer_count)
    prompt_necessity = cosine_change(
        full_layer[:, -tail:],
        no_prompt_layer[:, -tail:],
    ).mean(dim=1)
    closed_necessity = cosine_change(
        full_layer[:, -tail:],
        no_closed_layer[:, -tail:],
    ).mean(dim=1)
    closure = closed_necessity - prompt_necessity

    uncertainty_width = (
        provenance.upper_after[:, -tail:] - provenance.lower_after[:, -tail:]
    ).mean(dim=(1, 2))
    supported = views.no_prompt_supported[:, -tail:] & views.no_closed_supported[:, -tail:]
    fallback_fraction = 1.0 - supported.float().mean(dim=(1, 2))
    coverage = (fallback_fraction < 1.0).float()

    return PCutResult(
        prompt_necessity=prompt_necessity,
        response_closed_necessity=closed_necessity,
        closure=closure,
        coverage=coverage,
        token_layer_embedding=full_layer,
        token_embedding=full_token,
        no_prompt_embedding=no_prompt_token,
        no_closed_embedding=no_closed_token,
        prompt_origin_lower=provenance.lower_after,
        prompt_origin_upper=provenance.upper_after,
        edge_parts=parts,
        uncertainty_width=uncertainty_width,
        cut_fallback_fraction=fallback_fraction,
    )
