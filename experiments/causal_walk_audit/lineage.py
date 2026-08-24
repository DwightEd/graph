"""Anchor-resolved lineage on a layer-unrolled causal attention graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .anchors import AnchorMap

DIRECT = 0
ONE_HOP = 1
MULTI_HOP = 2
DEPTH_COUNT = 3


@dataclass(frozen=True)
class LineageTrace:
    state: torch.Tensor       # [token, layer, head, anchor+response_base, depth]
    unresolved: torch.Tensor  # [token, layer, head]
    anchor_count: int
    anchor_mode: str

    @property
    def response_base_index(self) -> int:
        return self.anchor_count

    def direct_anchor(self) -> torch.Tensor:
        return self.state[..., : self.anchor_count, DIRECT]

    def relay_anchor(self) -> torch.Tensor:
        return self.state[..., : self.anchor_count, ONE_HOP:].sum(dim=-1)

    def multihop_anchor(self) -> torch.Tensor:
        return self.state[..., : self.anchor_count, MULTI_HOP]

    def response_base(self) -> torch.Tensor:
        return self.state[..., self.response_base_index, :].sum(dim=-1)

    def response_base_one_hop(self) -> torch.Tensor:
        return self.state[..., self.response_base_index, ONE_HOP]

    def response_base_multihop(self) -> torch.Tensor:
        return self.state[..., self.response_base_index, MULTI_HOP]

    def known_anchor_mass(self) -> torch.Tensor:
        return self.state[..., : self.anchor_count, :].sum(dim=(-1, -2))


def _shift_depth(value: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(value)
    shifted[..., ONE_HOP] = value[..., DIRECT]
    shifted[..., MULTI_HOP] = value[..., ONE_HOP] + value[..., MULTI_HOP]
    return shifted


def _conserve(state: torch.Tensor, unresolved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    total = state.sum(dim=(-1, -2)) + unresolved
    scale = torch.where(total > 1.0, total.reciprocal(), torch.ones_like(total))
    state = state * scale[..., None, None]
    unresolved = unresolved * scale
    unresolved = unresolved + (1.0 - state.sum(dim=(-1, -2)) - unresolved).clamp_min(0.0)
    return state, unresolved


def propagate_anchor_lineage(routing, anchors: AnchorMap) -> LineageTrace:
    """Propagate a conserved attention-derived anchor lineage through layers."""

    edges = routing.edges
    tokens = edges.num_response_tokens
    anchors_count = anchors.count
    kinds = anchors_count + 1
    response_base = anchors_count

    previous = routing.prompt_mass.new_zeros((tokens, kinds, DEPTH_COUNT))
    previous[:, response_base, DIRECT] = 1.0
    states, unresolved_layers = [], []

    layer_order = torch.argsort(edges.layer, stable=True)
    counts = torch.bincount(edges.layer, minlength=edges.num_layers)
    pointer = torch.cat((counts.new_zeros(1), counts.cumsum(0)))

    for layer in range(edges.num_layers):
        head_state = previous.new_zeros((tokens, edges.num_heads, kinds, DEPTH_COUNT))
        head_unresolved = routing.unresolved_mass[:, layer].clone()
        head_state += routing.self_mass[:, layer, :, None, None] * previous[:, None]

        indices = layer_order[int(pointer[layer]) : int(pointer[layer + 1])]
        if indices.numel():
            query = edges.query[indices]
            head = edges.head[indices]
            source = edges.source[indices]
            weight = routing.edge_weight[indices]

            prompt = source < edges.response_idx
            if bool(prompt.any()):
                selected = torch.nonzero(prompt, as_tuple=False).flatten()
                anchor = anchors.token_anchor[source[selected]]
                head_state.index_put_(
                    (query[selected], head[selected], anchor),
                    torch.nn.functional.one_hot(
                        torch.full_like(anchor, DIRECT), DEPTH_COUNT
                    ).to(head_state.dtype)
                    * weight[selected, None],
                    accumulate=True,
                )

            response = ~prompt
            if bool(response.any()):
                selected = torch.nonzero(response, as_tuple=False).flatten()
                response_source = source[selected] - edges.response_idx
                message = _shift_depth(previous[response_source])
                message = message * weight[selected, None, None]
                head_state.index_put_(
                    (query[selected], head[selected]),
                    message,
                    accumulate=True,
                )

        head_state, head_unresolved = _conserve(head_state, head_unresolved)
        states.append(head_state)
        unresolved_layers.append(head_unresolved)
        previous = head_state.mean(dim=1)

    return LineageTrace(
        state=torch.stack(states, dim=1),
        unresolved=torch.stack(unresolved_layers, dim=1),
        anchor_count=anchors_count,
        anchor_mode=anchors.mode,
    )
