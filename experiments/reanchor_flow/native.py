"""Causal orchestration for one native evidence-to-target corridor."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .corridor import (
    CarrierEffect,
    CorridorEffect,
    RootEffect,
    complete_mediation_confirmed,
    confirm_carriers,
    confirm_corridor,
    intervention_tolerance,
    rerun_margin,
    select_root,
)
from .flow import FlowEdges, FlowSignal, PairedFlow, margin, stage_trace
from .native_flow import attach_cut_edge_codes, native_flow_screen
from .native_world import (
    NativeWorld,
    gated_forward_cache,
    source_gate,
)
from .throughput import FlowThroughput, compute_throughput
from .worlds import TargetContrast


@dataclass(frozen=True)
class NativeTargetAudit:
    """Native transport screen followed by source-cut causal confirmation."""

    world: NativeWorld
    flow: PairedFlow
    transport_throughput: FlowThroughput
    throughput: FlowThroughput
    corridor: FlowEdges
    effect: CorridorEffect
    corridor_confirmed: bool
    roots: tuple[RootEffect, ...]
    all_evidence_cut_margin: float
    selected_root_unit_id: int
    selected_root_effect: RootEffect
    selected_root_confirmed: bool
    carriers: tuple[CarrierEffect, ...]


def confirm_native_roots(
    model,
    flow: PairedFlow,
    world: NativeWorld,
    throughput: FlowThroughput,
    *,
    limit: int,
) -> tuple[tuple[RootEffect, ...], float]:
    """Test source Value-message-cut necessity and single-unit sufficiency."""

    candidates = list(world.evidence_unit_id)
    candidates.sort(
        key=lambda unit_id: float(throughput.unit_mass[unit_id]),
        reverse=True,
    )
    evaluated = set(candidates if limit == 0 else candidates[:limit])
    all_cut = rerun_margin(
        model,
        flow.clean_cache,
        source_gate(world, world.evidence_unit_id),
        flow.target,
    )
    effects = []
    for unit_id in candidates:
        route_mass = float(throughput.unit_mass[unit_id])
        selected_edge = flow.edges.source_unit == unit_id
        functional_score = float(flow.edges.clean_target_score[selected_edge].sum())
        if unit_id not in evaluated:
            effects.append(
                RootEffect(
                    unit_id,
                    route_mass,
                    functional_score,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    False,
                )
            )
            continue
        cut_margin = rerun_margin(
            model,
            flow.clean_cache,
            source_gate(world, (unit_id,)),
            flow.target,
        )
        other_units = tuple(
            candidate for candidate in world.evidence_unit_id if candidate != unit_id
        )
        only_unit = rerun_margin(
            model,
            flow.clean_cache,
            source_gate(world, other_units),
            flow.target,
        )
        necessity = flow.clean_margin - cut_margin
        sufficiency = only_unit - all_cut
        effects.append(
            RootEffect(
                unit_id,
                route_mass,
                functional_score,
                necessity,
                sufficiency,
                min(necessity, sufficiency),
                True,
            )
        )
    return tuple(effects), all_cut


def audit_native_target(
    model,
    world: NativeWorld,
    target: TargetContrast,
    signal: FlowSignal | str,
    *,
    carrier_scope: str = "response",
    coverage: float = 0.9,
    query_chunk: int = 8,
    root_screen_limit: int = 4,
    carrier_limit: int = 2,
) -> NativeTargetAudit:
    """Run native roots, carrier mediation, and exact corridor tests."""

    signal = FlowSignal(signal)
    prefix = world.prefix(target)
    screen, gradients = native_flow_screen(
        model,
        prefix,
        target,
        signal,
        carrier_scope=carrier_scope,
        coverage=coverage,
        query_chunk=query_chunk,
    )
    transport_throughput = compute_throughput(
        screen,
        prefix.units.token_unit_id,
        prefix.units.count,
        prefix.evidence_unit_id,
    )
    support_score = torch.where(
        screen.edges.clean_target_score > 0,
        screen.edges.score,
        torch.zeros_like(screen.edges.score),
    )
    support_screen = replace(
        screen,
        edges=replace(screen.edges, score=support_score),
    )
    candidate_throughput = compute_throughput(
        support_screen,
        prefix.units.token_unit_id,
        prefix.units.count,
        prefix.evidence_unit_id,
    )
    roots, all_cut_margin = confirm_native_roots(
        model,
        screen,
        prefix,
        candidate_throughput,
        limit=root_screen_limit,
    )
    selected_root = select_root(roots)
    selected_effect = next(
        effect for effect in roots if effect.unit_id == selected_root
    )
    tolerance = intervention_tolerance(model)
    selected_root_confirmed = bool(
        selected_effect.evaluated
        and selected_effect.route_mass > 0
        and selected_effect.causal_score > tolerance
    )

    root_gate = source_gate(prefix, (selected_root,))
    root_cut = gated_forward_cache(model, screen.clean_cache, root_gate)
    edges = attach_cut_edge_codes(
        model,
        screen.edges,
        root_cut,
        gradients,
        root_gate.source_mask,
    )
    flow = replace(
        screen,
        corrupt_margin=margin(model, root_cut, target),
        edges=edges,
        stages=stage_trace(screen.clean_cache, root_cut, gradients),
        corrupt_cache=root_cut,
        corrupt_source_mask=root_gate.source_mask,
    )

    support_flow = replace(
        flow,
        edges=replace(flow.edges, score=support_score),
    )
    throughput = compute_throughput(
        support_flow,
        prefix.units.token_unit_id,
        prefix.units.count,
        (selected_root,),
    )
    corridor = flow.edges.select(throughput.edge > 0)
    effect = confirm_corridor(model, flow, corridor)
    corridor_confirmed = bool(
        selected_root_confirmed
        and effect.restoration_valid
        and complete_mediation_confirmed(
            effect.necessity,
            effect.sufficiency,
            effect.blocked_sufficiency,
            direction=1.0,
            tolerance=tolerance,
        )
    )
    carriers = confirm_carriers(
        model,
        flow,
        throughput,
        prefix.units.positions((selected_root,)),
        limit=carrier_limit,
        effect_direction=1.0,
    )
    return NativeTargetAudit(
        prefix,
        flow,
        transport_throughput,
        throughput,
        corridor,
        effect,
        corridor_confirmed,
        roots,
        all_cut_margin,
        selected_root,
        selected_effect,
        selected_root_confirmed,
        carriers,
    )
