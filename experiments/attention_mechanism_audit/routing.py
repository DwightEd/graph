"""Fine-role attention routing and response-carrier ancestry.

This module deliberately keeps attention routing separate from functional
gradient attribution.  It refines the old prompt/response partition into
evidence, question, constraint, other prompt, earlier response, diagonal and
unresolved mass, while reusing the exact censor-aware endpoint bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import torch

from experiments.directed_route_hypergraph.routing_dispersion import (
    attention_routing_dispersion,
)
from experiments.grounded_route.graph import TokenGraph

from .functional_flow import PROMPT_ROLE_NAMES, prompt_role_array


HISTORY = len(PROMPT_ROLE_NAMES)
DIAGONAL = HISTORY + 1
UNRESOLVED = HISTORY + 2
ROUTING_ROLE_NAMES = (*PROMPT_ROLE_NAMES, "history", "diagonal", "unresolved")
EPSILON = 1e-12


@dataclass(frozen=True)
class ResponseCarrier:
    """Sparse, head-averaged response-to-response attention carrier.

    Repeating a dense ``[query, layer, head, source]`` carrier would require
    quadratic response memory per head.  Since every head consumes the same
    previous-layer ancestry, averaging its linear transition is exactly
    equivalent to applying this carrier whose edge weights are divided by the
    number of heads.
    """

    query: torch.Tensor
    source: torch.Tensor
    layer: torch.Tensor
    weight: torch.Tensor
    response_count: int
    layer_count: int

    @cached_property
    def layer_offsets(self) -> tuple[int, ...]:
        count = torch.bincount(
            self.layer.detach().cpu(), minlength=self.layer_count
        )
        return (0, *count.cumsum(0).tolist())

    def layer_edges(
        self, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start, stop = self.layer_offsets[layer : layer + 2]
        return self.query[start:stop], self.source[start:stop], self.weight[start:stop]

    def query_mass(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mass = torch.zeros(
            (self.response_count, self.layer_count), dtype=dtype, device=device
        )
        if self.weight.numel():
            mass.index_put_(
                (self.query.to(device), self.layer.to(device)),
                self.weight.to(device=device, dtype=dtype),
                accumulate=True,
            )
        return mass


def _nan_first(value: torch.Tensor, response_count: int) -> torch.Tensor:
    """Insert the unavailable first-token row without inventing zero signal."""

    result = torch.full(
        (response_count, *value.shape[1:]),
        torch.nan,
        dtype=value.dtype,
        device=value.device,
    )
    if response_count > 1:
        result[1:] = value
    return result


def _generalized_head_js(role_mass: torch.Tensor) -> torch.Tensor:
    """Normalized generalized JSD over heads for arbitrary role count."""

    # role_mass: [R, L, H, K], including unresolved, and rows conserve mass.
    totals = role_mass.sum(dim=-1)
    distributions = role_mass / totals.clamp_min(EPSILON)[..., None]
    valid = totals > 0
    rows, layers, heads, roles = distributions.shape
    output = role_mass.new_full((rows, layers), torch.nan)
    for row in range(rows):
        for layer in range(layers):
            selected = distributions[row, layer, valid[row, layer]]
            count = selected.shape[0]
            if count == 0:
                continue
            if count == 1:
                output[row, layer] = 0.0
                continue
            mixture = selected.mean(dim=0)
            mixture_entropy = -torch.xlogy(mixture, mixture).sum()
            head_entropy = -torch.xlogy(selected, selected).sum(dim=-1).mean()
            maximum = role_mass.new_tensor(float(min(count, roles))).log()
            output[row, layer] = (
                (mixture_entropy - head_entropy) / maximum
            ).clamp(0.0, 1.0)
    return output


@torch.no_grad()
def direct_role_bases_and_carrier(
    graph: TokenGraph,
    prompt_roles: Any,
) -> tuple[torch.Tensor, ResponseCarrier, torch.Tensor, torch.Tensor]:
    """Build direct prompt-role bases and a strict response carrier.

    Returns query-aligned head-averaged ``B [Q,L,K]``, a sparse strict-lower
    ``ResponseCarrier``, exact diagonal ``D [Q,L]`` and unresolved mass
    ``U [Q,L]``.  Keeping the diagonal separate is essential: it propagates
    the previous-layer state of the same query, not an earlier same-layer
    token state.
    """

    graph = graph.canonicalize().check()
    role_ids = prompt_role_array(
        prompt_roles,
        graph.response_start,
        graph.token_ids.detach().cpu().numpy(),
    )
    device = graph.diagonal.device
    dtype = graph.diagonal.dtype
    response_count = graph.response_count
    layers = graph.layer_count
    heads = graph.head_count
    prompt_roles_count = len(PROMPT_ROLE_NAMES)

    direct = torch.zeros(
        (response_count, layers, prompt_roles_count),
        dtype=dtype,
        device=device,
    )
    diagonal = graph.diagonal.to(device=device, dtype=dtype).mean(dim=2)
    unresolved = graph.unresolved.to(device=device, dtype=dtype).mean(dim=2)
    role_tensor = torch.as_tensor(role_ids, dtype=torch.long, device=device)

    edges = graph.edges.to(device)
    if edges.count:
        query = edges.target - graph.response_start
        source = edges.source
        layer = edges.layer
        weight = edges.weight.to(dtype=dtype)
        prompt = source < graph.response_start
        if bool(prompt.any()):
            direct.index_put_(
                (
                    query[prompt],
                    layer[prompt],
                    role_tensor[source[prompt]],
                ),
                weight[prompt] / heads,
                accumulate=True,
            )
        history = ~prompt
        if bool(history.any()):
            response_source = source[history] - graph.response_start
            if bool((response_source >= query[history]).any()):
                raise ValueError("response carrier must be strictly lower triangular")
            carrier = ResponseCarrier(
                query=query[history],
                source=response_source,
                layer=layer[history],
                weight=weight[history] / heads,
                response_count=response_count,
                layer_count=layers,
            )
        else:
            carrier = ResponseCarrier(
                query=query.new_empty(0),
                source=query.new_empty(0),
                layer=query.new_empty(0),
                weight=weight.new_empty(0),
                response_count=response_count,
                layer_count=layers,
            )
    else:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        carrier = ResponseCarrier(
            query=empty_long,
            source=empty_long.clone(),
            layer=empty_long.clone(),
            weight=torch.empty(0, dtype=dtype, device=device),
            response_count=response_count,
            layer_count=layers,
        )

    if response_count:
        carrier_mass = carrier.query_mass(dtype=dtype, device=device)
        row = direct.sum(dim=-1) + carrier_mass + diagonal + unresolved
        if row.numel() and not torch.allclose(
            row,
            torch.ones_like(row),
            atol=5e-4,
            rtol=0.0,
        ):
            raise ValueError("fine-role routing rows do not conserve attention mass")
    return direct, carrier, diagonal, unresolved


@torch.no_grad()
def response_carrier_ancestry(
    direct: torch.Tensor,
    carrier: ResponseCarrier,
    diagonal: torch.Tensor,
    unresolved: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compose ancestry across Transformer layers, never within one layer.

    Transformer positions in a layer are computed in parallel.  Therefore a
    response carrier ``q <- u`` at layer ``l`` reads ``u``'s state from layer
    ``l-1``.  The exact diagonal likewise reads ``q``'s previous-layer state.
    The recurrence is

        G_q^l = B_q^l + sum_{u<q} P^l[q,u] G_u^(l-1)
                + D_q^l G_q^(l-1).

    The sparse carrier stores head weights divided by ``H``.  Because all
    heads consume the same previous-layer state and the transition is linear,
    this is algebraically identical to completing each head transition and
    averaging afterward, as in ``routing_lineage.py``.
    """

    if direct.ndim != 3:
        raise ValueError("expected direct prompt roles with shape [Q,L,K]")
    response_count, layers, roles = direct.shape
    if (
        carrier.response_count != response_count
        or carrier.layer_count != layers
        or carrier.query.shape != carrier.source.shape
        or carrier.query.shape != carrier.layer.shape
        or carrier.query.shape != carrier.weight.shape
    ):
        raise ValueError("carrier shape does not match direct bases")
    if diagonal.shape != (response_count, layers) or unresolved.shape != (
        response_count, layers
    ):
        raise ValueError("diagonal/unresolved shape does not match direct bases")
    if carrier.query.numel() and bool((carrier.source >= carrier.query).any()):
        raise ValueError("response carrier is not strictly lower triangular")

    previous_grounded = direct.new_zeros((response_count, roles))
    previous_ungrounded = direct.new_ones(response_count)
    previous_censored = direct.new_zeros(response_count)
    direct_trace = direct.new_zeros((response_count, layers, roles))
    relayed_trace = torch.zeros_like(direct_trace)
    grounded_trace = torch.zeros_like(direct_trace)
    ungrounded_trace = direct.new_zeros((response_count, layers))
    censored_trace = torch.zeros_like(ungrounded_trace)

    for layer in range(layers):
        query, source, weight = carrier.layer_edges(layer)
        inherited_grounded = diagonal[:, layer, None] * previous_grounded
        inherited_ungrounded = diagonal[:, layer] * previous_ungrounded
        inherited_censored = diagonal[:, layer] * previous_censored
        if weight.numel():
            inherited_grounded.index_add_(
                0, query, previous_grounded[source] * weight[:, None]
            )
            inherited_ungrounded.index_add_(
                0, query, previous_ungrounded[source] * weight
            )
            inherited_censored.index_add_(
                0, query, previous_censored[source] * weight
            )

        layer_direct = direct[:, layer]
        layer_relayed = inherited_grounded
        previous_grounded = layer_direct + layer_relayed
        previous_ungrounded = inherited_ungrounded
        previous_censored = unresolved[:, layer] + inherited_censored
        conserved = (
            previous_grounded.sum(dim=-1)
            + previous_ungrounded
            + previous_censored
        )
        if not torch.allclose(
            conserved,
            torch.ones_like(conserved),
            atol=1e-3,
            rtol=0.0,
        ):
            raise ValueError("layer-wise response ancestry does not conserve mass")
        direct_trace[:, layer] = layer_direct
        relayed_trace[:, layer] = layer_relayed
        grounded_trace[:, layer] = previous_grounded
        ungrounded_trace[:, layer] = previous_ungrounded
        censored_trace[:, layer] = previous_censored

    return {
        "routing_direct_role_ancestry": direct_trace,
        "routing_relayed_role_ancestry": relayed_trace,
        "routing_grounded_role_ancestry": grounded_trace,
        "routing_ungrounded_history_ancestry": ungrounded_trace,
        "routing_unresolved_ancestry": censored_trace,
    }


@torch.no_grad()
def routing_flow(graph: TokenGraph, prompt_roles: Any) -> dict[str, np.ndarray]:
    """Compute fine-role, dispersion-bound and recursive routing trajectories."""

    graph = graph.canonicalize().check()
    role_ids = prompt_role_array(
        prompt_roles,
        graph.response_start,
        graph.token_ids.detach().cpu().numpy(),
    )
    device = graph.diagonal.device
    dtype = graph.diagonal.dtype
    response_count = graph.response_count
    layers = graph.layer_count
    heads = graph.head_count
    role_count = len(ROUTING_ROLE_NAMES)

    raw_query_role_mass = torch.zeros(
        (response_count, layers, heads, role_count), dtype=dtype, device=device
    )
    role_tensor = torch.as_tensor(role_ids, dtype=torch.long, device=device)
    edges = graph.edges.to(device)
    if edges.count:
        query = edges.target - graph.response_start
        source = edges.source
        layer = edges.layer
        head = edges.head
        weight = edges.weight.to(dtype=dtype)
        source_role = torch.full_like(source, HISTORY)
        prompt = source < graph.response_start
        source_role[prompt] = role_tensor[source[prompt]]
        raw_query_role_mass.index_put_(
            (query, layer, head, source_role), weight, accumulate=True
        )
    raw_query_role_mass[..., DIAGONAL] = graph.diagonal.to(device)
    raw_query_role_mass[..., UNRESOLVED] = graph.unresolved.to(device)

    direct, carrier, diagonal, unresolved = direct_role_bases_and_carrier(
        graph, role_ids
    )
    query_ancestry = response_carrier_ancestry(
        direct, carrier, diagonal, unresolved
    )
    dispersion = attention_routing_dispersion(graph)
    entropy_bounds = _nan_first(dispersion.token_entropy_bounds, response_count)
    hhi_bounds = _nan_first(dispersion.token_hhi_bounds, response_count)
    concentration_bounds = _nan_first(
        dispersion.token_concentration_bounds, response_count
    )
    query_role_mass = _nan_first(raw_query_role_mass[:-1], response_count)
    role_js = _nan_first(
        _generalized_head_js(raw_query_role_mass)[:-1], response_count
    )
    mean_role_mass = query_role_mass.mean(dim=2)
    known_coverage = 1.0 - query_role_mass[..., UNRESOLVED]
    aligned_direct = _nan_first(direct[:-1], response_count)
    query_carrier_mass = carrier.query_mass(dtype=dtype, device=device)
    aligned_carrier_mass = _nan_first(query_carrier_mass[:-1], response_count)
    aligned_diagonal = _nan_first(diagonal[:-1], response_count)
    aligned_unresolved = _nan_first(unresolved[:-1], response_count)
    ancestry = {
        name: _nan_first(value[:-1], response_count)
        for name, value in query_ancestry.items()
    }
    available = torch.ones(response_count, dtype=torch.bool, device=device)
    if response_count:
        available[0] = False

    output: dict[str, np.ndarray] = {
        "routing_available": available.cpu().numpy(),
        "routing_role_names": np.asarray(ROUTING_ROLE_NAMES),
        "routing_fine_role_mass": query_role_mass.cpu().numpy(),
        "routing_mean_fine_role_mass": mean_role_mass.cpu().numpy(),
        "routing_head_role_js": role_js.cpu().numpy(),
        "routing_entropy_bounds": entropy_bounds.cpu().numpy(),
        "routing_hhi_bounds": hhi_bounds.cpu().numpy(),
        "routing_concentration_bounds": concentration_bounds.cpu().numpy(),
        "routing_known_attention_coverage": known_coverage.cpu().numpy(),
        "routing_direct_role_base": aligned_direct.cpu().numpy(),
        "routing_response_carrier_mass": aligned_carrier_mass.cpu().numpy(),
        "routing_exact_diagonal": aligned_diagonal.cpu().numpy(),
        "routing_direct_unresolved": aligned_unresolved.cpu().numpy(),
        "routing_token_index": np.arange(response_count, dtype=np.int64),
    }
    output.update({name: value.cpu().numpy() for name, value in ancestry.items()})
    return output


__all__ = [
    "ROUTING_ROLE_NAMES",
    "direct_role_bases_and_carrier",
    "response_carrier_ancestry",
    "routing_flow",
]
