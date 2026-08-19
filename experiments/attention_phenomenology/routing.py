"""Build role-aligned and endpoint-preserving attention routing tensors.

Raw attention access stays behind ``ResearchSample.iter_sparse_attention_blocks``.
The module does not fit references, read labels, or construct null models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PhenomenologyConfig


@dataclass(frozen=True)
class RoutingEdges:
    """Retained off-diagonal attention edges plus exact self diagonals."""

    num_layers: int
    num_heads: int
    num_response_tokens: int
    num_tokens: int
    response_idx: int
    attention_floor: float
    layer: torch.Tensor
    head: torch.Tensor
    query: torch.Tensor
    source: torch.Tensor
    weight: torch.Tensor
    diagonal: torch.Tensor

    @property
    def device(self):
        return self.diagonal.device


@dataclass(frozen=True)
class RoutingTensor:
    """Fixed routing roles and exact response-source coordinates."""

    edges: RoutingEdges
    role_probability: torch.Tensor
    known_role_probability: torch.Tensor
    prompt_mass: torch.Tensor
    rr_mass: torch.Tensor
    self_mass: torch.Tensor
    unresolved_mass: torch.Tensor
    known_mass: torch.Tensor
    source_mass: torch.Tensor
    rr_query: torch.Tensor
    rr_layer: torch.Tensor
    rr_head: torch.Tensor
    rr_source: torch.Tensor
    rr_weight: torch.Tensor


def collect_routing_edges(
    sample,
    *,
    config: PhenomenologyConfig | None = None,
) -> RoutingEdges:
    """Decode retained causal rows through the shared research interface."""

    config = PhenomenologyConfig() if config is None else config
    attention = sample.attention()
    response_idx = int(attention.response_idx)

    layer_parts = []
    head_parts = []
    query_parts = []
    source_parts = []
    weight_parts = []

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        target = response_idx + block.query
        off_diagonal = block.source < target
        if off_diagonal.any():
            layer_parts.append(block.layer[off_diagonal].long())
            head_parts.append(block.head[off_diagonal].long())
            query_parts.append(block.query[off_diagonal].long())
            source_parts.append(block.source[off_diagonal].long())
            weight_parts.append(block.weight[off_diagonal].float().clamp_min(0.0))

    device = attention.response_values.device
    if layer_parts:
        layer = torch.cat(layer_parts)
        head = torch.cat(head_parts)
        query = torch.cat(query_parts)
        source = torch.cat(source_parts)
        weight = torch.cat(weight_parts)
    else:
        layer = torch.empty(0, dtype=torch.long, device=device)
        head = torch.empty_like(layer)
        query = torch.empty_like(layer)
        source = torch.empty_like(layer)
        weight = torch.empty(0, dtype=torch.float32, device=device)

    diagonal = (
        attention.attention_diagonal[:, :, response_idx:]
        .float()
        .permute(2, 0, 1)
        .contiguous()
    )
    return RoutingEdges(
        num_layers=int(attention.num_layers),
        num_heads=int(attention.num_heads),
        num_response_tokens=int(attention.num_response_tokens),
        num_tokens=int(attention.num_tokens),
        response_idx=response_idx,
        attention_floor=float(attention.attention_floor),
        layer=layer,
        head=head,
        query=query,
        source=source,
        weight=weight,
        diagonal=diagonal,
    )


def prompt_bin(source: torch.Tensor, prompt_count: int, bins: int) -> torch.Tensor:
    return torch.div(source * bins, max(prompt_count, 1), rounding_mode="floor").clamp_max(
        bins - 1
    )


def lag_bin(lag: torch.Tensor, bins: int) -> torch.Tensor:
    return torch.floor(torch.log2(lag.float())).long().clamp_max(bins - 1)


def build_routing_tensor(
    edges: RoutingEdges,
    *,
    config: PhenomenologyConfig | None = None,
) -> RoutingTensor:
    """Create a fixed role simplex and retain exact RR endpoints."""

    config = PhenomenologyConfig() if config is None else config
    response_count = edges.num_response_tokens
    shape = (response_count, edges.num_layers, edges.num_heads)
    role = torch.zeros((*shape, config.role_count), device=edges.device)
    role[..., config.self_role] = edges.diagonal

    if edges.weight.numel():
        is_prompt = edges.source < edges.response_idx
        role_index = torch.empty_like(edges.source)
        role_index[is_prompt] = prompt_bin(
            edges.source[is_prompt], edges.response_idx, config.prompt_bins
        )
        history = ~is_prompt
        response_source = edges.source[history] - edges.response_idx
        role_index[history] = config.prompt_bins + lag_bin(
            edges.query[history] - response_source, config.rr_lag_bins
        )
        role.index_put_(
            (edges.query, edges.layer, edges.head, role_index),
            edges.weight,
            accumulate=True,
        )

    known_mass = role[..., : config.unresolved_role].sum(dim=-1)
    row_scale = torch.where(
        known_mass > 1.0, known_mass.reciprocal(), torch.ones_like(known_mass)
    )
    role[..., : config.unresolved_role] *= row_scale[..., None]
    known_mass = known_mass * row_scale
    unresolved = (1.0 - known_mass).clamp_min(0.0)
    role[..., config.unresolved_role] = unresolved

    known_roles = role[..., : config.unresolved_role]
    known_probability = known_roles / known_mass[..., None].clamp_min(config.epsilon)
    known_probability = torch.where(
        known_mass[..., None] > config.epsilon,
        known_probability,
        torch.zeros_like(known_probability),
    )

    prompt_mass = role[..., : config.prompt_bins].sum(dim=-1)
    rr_slice = slice(config.prompt_bins, config.prompt_bins + config.rr_lag_bins)
    rr_mass = role[..., rr_slice].sum(dim=-1)
    self_mass = role[..., config.self_role]

    if edges.weight.numel():
        scaled_weight = edges.weight * row_scale[
            edges.query, edges.layer, edges.head
        ]
        is_rr = edges.source >= edges.response_idx
        rr_query = edges.query[is_rr]
        rr_layer = edges.layer[is_rr]
        rr_head = edges.head[is_rr]
        rr_source = edges.source[is_rr] - edges.response_idx
        rr_weight = scaled_weight[is_rr]
    else:
        rr_query = edges.query
        rr_layer = edges.layer
        rr_head = edges.head
        rr_source = edges.source
        rr_weight = edges.weight

    source_mass = torch.zeros(
        (response_count, edges.num_layers, response_count),
        dtype=torch.float32,
        device=edges.device,
    )
    if rr_weight.numel():
        source_mass.index_put_(
            (rr_query, rr_layer, rr_source), rr_weight, accumulate=True
        )

    return RoutingTensor(
        edges=edges,
        role_probability=role,
        known_role_probability=known_probability,
        prompt_mass=prompt_mass,
        rr_mass=rr_mass,
        self_mass=self_mass,
        unresolved_mass=unresolved,
        known_mass=known_mass,
        source_mass=source_mass,
        rr_query=rr_query,
        rr_layer=rr_layer,
        rr_head=rr_head,
        rr_source=rr_source,
        rr_weight=rr_weight,
    )
