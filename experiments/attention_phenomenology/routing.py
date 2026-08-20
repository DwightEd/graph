"""Decode sparse attention into prompt/response routing state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PhenomenologyConfig


PROMPT = 0
RESPONSE_HISTORY = 1
SELF = 2
UNRESOLVED = 3
ROLE_NAMES = ("prompt", "response_history", "self", "unresolved")


@dataclass(frozen=True)
class RoutingEdges:
    """Retained causal edges and exact self-attention diagonals."""

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
class RoutingState:
    """Per-head role masses plus normalized weights for every retained edge."""

    edges: RoutingEdges
    role_probability: torch.Tensor
    edge_weight: torch.Tensor
    prompt_mass: torch.Tensor
    response_mass: torch.Tensor
    self_mass: torch.Tensor
    unresolved_mass: torch.Tensor
    known_mass: torch.Tensor
    role_names: tuple[str, ...] = ROLE_NAMES


def collect_routing_edges(
    sample,
    *,
    config: PhenomenologyConfig | None = None,
) -> RoutingEdges:
    """Read retained causal attention through the research-dataset seam."""

    config = PhenomenologyConfig() if config is None else config
    attention = sample.attention()
    response_idx = int(attention.response_idx)
    parts = {name: [] for name in ("layer", "head", "query", "source", "weight")}

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        target = response_idx + block.query
        causal_off_diagonal = block.source < target
        if not causal_off_diagonal.any():
            continue
        parts["layer"].append(block.layer[causal_off_diagonal].long())
        parts["head"].append(block.head[causal_off_diagonal].long())
        parts["query"].append(block.query[causal_off_diagonal].long())
        parts["source"].append(block.source[causal_off_diagonal].long())
        parts["weight"].append(block.weight[causal_off_diagonal].float().clamp_min(0.0))

    device = attention.response_values.device
    if parts["weight"]:
        layer, head, query, source, weight = (
            torch.cat(parts[name]) for name in ("layer", "head", "query", "source", "weight")
        )
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


def build_routing_state(edges: RoutingEdges) -> RoutingState:
    """Aggregate edges into four explicit roles without discarding exact endpoints."""

    shape = (edges.num_response_tokens, edges.num_layers, edges.num_heads)
    role_mass = torch.zeros((*shape, len(ROLE_NAMES)), device=edges.device)
    role_mass[..., SELF] = edges.diagonal

    if edges.weight.numel():
        edge_role = torch.where(
            edges.source < edges.response_idx,
            torch.full_like(edges.source, PROMPT),
            torch.full_like(edges.source, RESPONSE_HISTORY),
        )
        # Every sparse edge contributes to one [token, layer, head, role] cell.
        role_mass.index_put_(
            (edges.query, edges.layer, edges.head, edge_role),
            edges.weight,
            accumulate=True,
        )

    observed_mass = role_mass[..., :UNRESOLVED].sum(dim=-1)
    row_scale = torch.ones_like(observed_mass)
    oversized = observed_mass > 1.0
    row_scale[oversized] = observed_mass[oversized].reciprocal()
    role_mass[..., :UNRESOLVED] *= row_scale.unsqueeze(-1)
    known_mass = observed_mass * row_scale
    role_mass[..., UNRESOLVED] = (1.0 - known_mass).clamp_min(0.0)

    edge_weight = edges.weight
    if edge_weight.numel():
        edge_weight = edge_weight * row_scale[edges.query, edges.layer, edges.head]

    return RoutingState(
        edges=edges,
        role_probability=role_mass,
        edge_weight=edge_weight,
        prompt_mass=role_mass[..., PROMPT],
        response_mass=role_mass[..., RESPONSE_HISTORY],
        self_mass=role_mass[..., SELF],
        unresolved_mass=role_mass[..., UNRESOLVED],
        known_mass=known_mass,
    )
