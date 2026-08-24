"""Layer-unfolded attention-lineage automaton with five exhaustive states.

The transport between layers is pre-registered and parameter-free.  Because
the cache does not expose ``W_V``, ``W_O``, residual, or MLP contributions, the
result is explicitly an attention-lineage proxy, not a claim about physical
hidden-state contribution.  Head transport is the permutation-invariant equal
mean of the previous layer's heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph_builder import CausalRoutingGraph, RP


P0 = 0
P_PLUS = 1
R0 = 2
R_PLUS = 3
U = 4
STATE_NAMES = ("P0", "P_PLUS", "R0", "R_PLUS", "U")
MAX_CONSERVATION_ERROR = 1e-4


@dataclass(frozen=True)
class LayeredAutomatonResult:
    route_distribution: torch.Tensor
    prompt_lineage: torch.Tensor
    detached: torch.Tensor
    conservation_error: torch.Tensor
    state_names: tuple[str, ...] = STATE_NAMES

    @property
    def flat_route_distribution(self) -> torch.Tensor:
        r, layers, heads, states = self.route_distribution.shape
        return self.route_distribution.reshape(r, layers * heads, states)

    @property
    def flat_prompt_lineage(self) -> torch.Tensor:
        r, layers, heads = self.prompt_lineage.shape
        return self.prompt_lineage.reshape(r, layers * heads)

    @property
    def flat_detached(self) -> torch.Tensor:
        r, layers, heads = self.detached.shape
        return self.detached.reshape(r, layers * heads)

    def validate(self) -> "LayeredAutomatonResult":
        if self.route_distribution.ndim != 4 or self.route_distribution.shape[-1] != 5:
            raise ValueError("layered route_distribution must be [R,L,H,5]")
        shape = self.route_distribution.shape[:-1]
        if self.prompt_lineage.shape != shape or self.detached.shape != shape:
            raise ValueError("prompt_lineage and detached must be [R,L,H]")
        if self.conservation_error.shape != shape:
            raise ValueError("conservation_error must be [R,L,H]")
        for tensor in (
            self.route_distribution,
            self.prompt_lineage,
            self.detached,
            self.conservation_error,
        ):
            if tensor.device != self.route_distribution.device:
                raise ValueError("layered automaton tensors must share one device")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("layered automaton tensors must be finite")
        if bool((self.route_distribution < -2e-6).any()):
            raise ValueError("route states must be non-negative")
        total = self.route_distribution.sum(dim=-1)
        if not torch.allclose(total, torch.ones_like(total), atol=2e-6, rtol=2e-6):
            raise ValueError("five-state route distribution must sum to one")
        if bool((self.conservation_error < 0).any()) or float(
            self.conservation_error.max().item()
        ) > MAX_CONSERVATION_ERROR:
            raise ValueError(
                "layered automaton pre-normalization conservation error exceeds "
                "the numerical tolerance"
            )
        return self


def _response_transport(state: torch.Tensor) -> torch.Tensor:
    """Apply T_R to ``[...,5]`` lineage state."""

    result = torch.zeros_like(state)
    result[..., P_PLUS] = state[..., P0] + state[..., P_PLUS]
    result[..., R_PLUS] = state[..., R0] + state[..., R_PLUS]
    result[..., U] = state[..., U]
    return result


@torch.no_grad()
def layered_attention_automaton(graph: CausalRoutingGraph) -> LayeredAutomatonResult:
    """Propagate exact RR endpoints through ordered attention layers.

    Before layer zero each response token is initialized in ``R0``. At later
    layers, the preceding heads are transported by an explicit equal mean. No
    edge ever reads a response token at or after its target.
    """

    graph.validate()
    r, layers, heads = (
        graph.num_response_tokens,
        graph.num_layers,
        graph.num_heads,
    )
    device = graph.device
    dtype = graph.weight.dtype
    route = torch.zeros((r, layers, heads, 5), dtype=dtype, device=device)
    error = torch.zeros((r, layers, heads), dtype=dtype, device=device)
    previous = torch.zeros((r, 5), dtype=dtype, device=device)
    previous[:, R0] = 1.0

    rr = graph.relation != RP
    rr_layer = graph.layer[rr]
    rr_head = graph.head[rr]
    rr_target = graph.query[rr]
    rr_source = graph.source[rr] - graph.response_idx
    rr_weight = graph.weight[rr]

    for layer in range(layers):
        current = torch.zeros((r, heads, 5), dtype=dtype, device=device)
        current[..., P0] = graph.prompt_mass[:, layer]
        # T_S is identity: exact diagonal mass keeps the current token lineage.
        current += graph.self_mass[:, layer].unsqueeze(-1) * previous.unsqueeze(1)
        current[..., U] += graph.unresolved_mass[:, layer]

        selected = rr_layer == layer
        if bool(selected.any()):
            source_state = _response_transport(previous[rr_source[selected]])
            message = source_state * rr_weight[selected].unsqueeze(-1)
            flat_target = rr_target[selected] * heads + rr_head[selected]
            rr_sum = torch.zeros((r * heads, 5), dtype=dtype, device=device)
            rr_sum.index_add_(0, flat_target, message)
            current += rr_sum.reshape(r, heads, 5)

        raw_total = current.sum(dim=-1)
        error[:, layer] = (raw_total - 1.0).abs()
        if bool((raw_total <= 0).any()):
            raise ValueError("layered automaton encountered an empty routing row")
        maximum_error = float(error[:, layer].max().item())
        if maximum_error > MAX_CONSERVATION_ERROR:
            raise ValueError(
                "layered automaton recurrence violated mass conservation: "
                f"maximum error={maximum_error:.8g}"
            )
        # Correct only floating-point accumulation drift. Graph-level masses
        # have already undergone the declared overshoot correction.
        current = current / raw_total.unsqueeze(-1)
        route[:, layer] = current
        # Pre-registered, permutation-invariant head transport in the absence
        # of W_O. It must not be described as a physical contribution map.
        previous = current.mean(dim=1)

    prompt_lineage = route[..., P0] + route[..., P_PLUS]
    detached = route[..., R_PLUS]
    return LayeredAutomatonResult(
        route_distribution=route,
        prompt_lineage=prompt_lineage,
        detached=detached,
        conservation_error=error,
    ).validate()
