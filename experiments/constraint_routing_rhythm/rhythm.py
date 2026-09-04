"""Describe evidence uptake and later delivery in functional route maps.

The rhythm is a proposal and visualization object.  None of its quantities is
used by the primary evidence-cut score; selected endpoints only determine
which U/D interventions are run as diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .routes import FunctionalRoutes


@dataclass(frozen=True)
class Rhythm:
    query_position: Tensor
    prediction_position: Tensor
    functional_reach: Tensor
    future_influence: Tensor
    future_delivery: Tensor
    evidence_uptake: Tensor
    evidence_binding: Tensor
    relay_capacity: Tensor
    relay_mass: Tensor
    carrier_mask: Tensor
    upstream_edges: Tensor
    downstream_edges: Tensor


def build_rhythm(
    routes: FunctionalRoutes,
    response_start: int,
    evidence_mask: Tensor,
    window: int = 10,
    horizon_low: int = 10,
    horizon_high: int = 100,
    carrier_quantile: float = 0.4,
    mass_floor: float = 1e-6,
    max_carriers: int = 8,
    split_layer: int = 0,
    build_endpoints: bool = True,
) -> Rhythm:
    """Build an explanatory uptake--anchor--delivery rhythm.

    Route matrices are indexed ``[response query row, absolute source]``.
    Event values are aligned to the token predicted by that query, while an
    upstream endpoint targets the generated token's own later carrier state.
    Uptake comes only from layers below ``split_layer`` and delivery only from
    later layers, so the proposed two-hop route matches the audited U/D bands.
    """

    if routes.split_layer != split_layer:
        raise ValueError("route maps and relay gate must use the same layer split")
    if mass_floor < 0:
        raise ValueError("mass_floor must be nonnegative")
    rows, sources = routes.all_map.shape
    query = torch.arange(routes.row_start, routes.row_start + rows)
    prediction = query + 1
    source_position = torch.arange(sources)

    distance = query[:, None] - source_position[None]
    distance = distance.clamp(min=0, max=window) / float(window)
    functional_reach = (routes.local_map * distance).sum(dim=1)

    future_influence = torch.full((rows,), torch.nan, dtype=routes.late_map.dtype)
    future_delivery = torch.full(
        (rows,), torch.nan, dtype=routes.late_absolute_map.dtype
    )
    evidence_uptake = torch.zeros(rows, dtype=routes.early_absolute_map.dtype)
    evidence_binding = torch.zeros(rows, dtype=routes.early_map.dtype)
    valid_carrier = prediction < sources
    has_future = torch.zeros(rows, dtype=torch.bool)

    evidence = torch.zeros(sources, dtype=torch.bool)
    supplied_evidence = torch.as_tensor(evidence_mask, dtype=torch.bool).flatten()
    evidence[: min(sources, len(supplied_evidence))] = supplied_evidence[:sources]

    for event, carrier in enumerate(prediction.tolist()):
        if carrier >= sources:
            continue
        carrier_row = carrier - routes.row_start
        if 0 <= carrier_row < rows:
            evidence_uptake[event] = routes.early_absolute_map[
                carrier_row, evidence
            ].sum()
            evidence_binding[event] = routes.early_map[carrier_row, evidence].sum()

        future = (query - carrier >= horizon_low) & (query - carrier <= horizon_high)
        if future.any():
            future_influence[event] = routes.late_map[future, carrier].mean()
            future_delivery[event] = routes.late_absolute_map[future, carrier].mean()
            has_future[event] = True

    relay_capacity = torch.minimum(evidence_binding, future_influence)
    relay_mass = torch.minimum(evidence_uptake, future_delivery)
    eligible = (
        valid_carrier
        & has_future
        & torch.isfinite(relay_capacity)
        & torch.isfinite(relay_mass)
        & (relay_capacity > 0)
        & (relay_mass > 0)
        & (relay_mass >= mass_floor)
    )
    top_relay = top_quantile(relay_capacity, eligible, carrier_quantile)
    # Reach remains an independent rhythm trace. Selecting endpoints only
    # from the relay bottleneck avoids manufacturing agreement between the
    # two public descriptors before the U/D audit.
    carrier_mask = top_relay
    selected = torch.nonzero(carrier_mask, as_tuple=False).flatten()
    if len(selected) > max_carriers:
        order = relay_capacity.index_select(0, selected).argsort(
            descending=True, stable=True
        )
        carrier_mask[:] = False
        carrier_mask[selected[order[:max_carriers]]] = True

    if build_endpoints:
        upstream = torch.zeros((sources, sources), dtype=torch.bool)
        downstream = torch.zeros_like(upstream)
        for event in torch.nonzero(carrier_mask, as_tuple=False).flatten().tolist():
            carrier = int(prediction[event])
            carrier_row = carrier - routes.row_start
            upstream[carrier] = evidence & (routes.early_absolute_map[carrier_row] > 0)

            future = (query - carrier >= horizon_low) & (
                query - carrier <= horizon_high
            )
            future_rows = torch.nonzero(future, as_tuple=False).flatten()
            future_targets = query.index_select(0, future_rows)
            active = routes.late_absolute_map[future_rows, carrier] > 0
            downstream[future_targets[active], carrier] = True
    else:
        upstream = downstream = torch.empty((0, 0), dtype=torch.bool)

    return Rhythm(
        query_position=query,
        prediction_position=prediction,
        functional_reach=functional_reach,
        future_influence=future_influence,
        future_delivery=future_delivery,
        evidence_uptake=evidence_uptake,
        evidence_binding=evidence_binding,
        relay_capacity=relay_capacity,
        relay_mass=relay_mass,
        carrier_mask=carrier_mask,
        upstream_edges=upstream,
        downstream_edges=downstream,
    )


def top_quantile(values: Tensor, eligible: Tensor, quantile: float) -> Tensor:
    """Select values at or above a quantile without inventing a fallback."""

    selected = torch.zeros_like(eligible)
    if eligible.any():
        threshold = torch.quantile(values[eligible].float(), quantile)
        selected = eligible & (values >= threshold)
    return selected


def relay_diagnostics(
    u_delta: Tensor,
    d_delta: Tensor,
    ud_delta: Tensor,
) -> Tensor:
    """Return raw U, D, and U/D interaction diagnostics.

    All deltas are intervention minus full.  The last column is the four-cell
    difference-in-differences ``UD - U - D``; it is not a mediation proof.
    """

    return torch.stack(
        (
            u_delta,
            d_delta,
            ud_delta - u_delta - d_delta,
        ),
        dim=-1,
    )
