"""Plan one native evidence route, then optionally confirm the frozen plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from .corridor import (
    CarrierEffect,
    CorridorEffect,
    PlannedCarrier,
    RootEffect,
    complete_mediation_confirmed,
    confirm_corridor,
    confirm_planned_carriers,
    intervention_tolerance,
    rerun_margin,
)
from .flow import FlowEdges, FlowSignal, PairedFlow, margin, stage_trace
from .native_flow import attach_cut_edge_codes, native_flow_screen
from .native_world import NativeWorld, gated_forward_cache, source_gate
from .route_model import HeadResolvedRouteModel, RouteDynamics
from .route_plan import AuditPlan, RouteBudget, plan_audit
from .throughput import FlowThroughput
from .worlds import TargetContrast


@dataclass(frozen=True)
class NativeTargetAudit:
    """A graph-selected route and its explicitly separated exact checks."""

    world: NativeWorld
    flow: PairedFlow
    plan: AuditPlan
    throughput: FlowThroughput
    corridor: FlowEdges
    effect: CorridorEffect
    corridor_evaluated: bool
    corridor_confirmed: bool
    roots: tuple[RootEffect, ...]
    all_evidence_cut_margin: float
    selected_root_unit_id: int
    selected_root_effect: RootEffect
    selected_root_confirmed: bool
    carriers: tuple[CarrierEffect, ...]
    carrier_evaluated_count: int
    dynamics: RouteDynamics


def _phase(callback: Callable[[str], None] | None, name: str) -> None:
    if callback is not None:
        callback(name)


def _unevaluated_corridor(
    flow: PairedFlow,
    corridor: FlowEdges,
    tolerance: float,
) -> CorridorEffect:
    missing = float("nan")
    return CorridorEffect(
        corridor.count,
        flow.pair_effect,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        tolerance,
        False,
    )


def _planned_carriers(
    flow: PairedFlow,
    plan: AuditPlan,
    tolerance: float,
) -> tuple[CarrierEffect, ...]:
    missing = float("nan")
    carriers = []
    for hub in plan.hubs:
        state_delta = (
            flow.clean_cache.layer_input[hub.layer][hub.position].float()
            - flow.corrupt_cache.layer_input[hub.layer][hub.position].float()
        )
        carriers.append(
            CarrierEffect(
                hub.layer,
                hub.position,
                hub.route_mass,
                float(state_delta.norm()),
                hub.signed_action,
                missing,
                missing,
                missing,
                missing,
                missing,
                tolerance,
                False,
            )
        )
    return tuple(carriers)


def _root_effects(
    model,
    flow: PairedFlow,
    world: NativeWorld,
    plan: AuditPlan,
    *,
    confirm: bool,
) -> tuple[tuple[RootEffect, ...], float]:
    """Attach exact values to the selected root without re-ranking it."""

    selected = plan.selected_root_unit_id
    missing = float("nan")
    necessity = flow.pair_effect if confirm else missing
    all_cut_margin = missing
    sufficiency = missing
    causal_score = missing
    effect_direction = 1.0 if flow.pair_effect >= 0 else -1.0
    if confirm:
        if tuple(world.evidence_unit_id) == (selected,):
            all_cut_margin = flow.corrupt_margin
            only_selected_margin = flow.clean_margin
        else:
            all_cut_margin = rerun_margin(
                model,
                flow.clean_cache,
                source_gate(world, world.evidence_unit_id),
                flow.target,
            )
            other = tuple(
                unit_id for unit_id in world.evidence_unit_id if unit_id != selected
            )
            only_selected_margin = rerun_margin(
                model,
                flow.clean_cache,
                source_gate(world, other),
                flow.target,
            )
        sufficiency = only_selected_margin - all_cut_margin
        causal_score = min(
            effect_direction * necessity,
            effect_direction * sufficiency,
        )

    effects = []
    for score in plan.roots:
        is_selected = score.unit_id == selected
        effects.append(
            RootEffect(
                score.unit_id,
                score.route_mass,
                score.signed_action,
                necessity if is_selected else missing,
                sufficiency if is_selected else missing,
                causal_score if is_selected else missing,
                bool(confirm and is_selected),
            )
        )
    return tuple(effects), all_cut_margin


def audit_native_target(
    model,
    world: NativeWorld,
    target: TargetContrast,
    signal: FlowSignal | str,
    *,
    carrier_scope: str = "response",
    coverage: float = 0.9,
    query_chunk: int = 8,
    route_budget: RouteBudget | None = None,
    local_window: int = 10,
    on_phase: Callable[[str], None] | None = None,
) -> NativeTargetAudit:
    """Discover a bounded graph, then validate only its frozen plan.

    Root, hub, and corridor identities depend only on the native message graph
    and the fixed target-margin gradient. Exact interventions are optional
    measurements and can never change those identities.
    """

    signal = FlowSignal(signal)
    prefix = world.prefix(target)
    budget = route_budget or RouteBudget()

    _phase(on_phase, "capture")
    screen, gradients = native_flow_screen(
        model,
        prefix,
        target,
        signal,
        carrier_scope=carrier_scope,
        coverage=coverage,
        query_chunk=query_chunk,
        max_rows=budget.max_rows,
        max_edges_per_head_row=budget.edges_per_head,
        local_window=local_window,
    )
    _phase(on_phase, "plan")
    plan = plan_audit(screen, prefix, budget)
    selected_root = plan.selected_root_unit_id

    # This single selected-root run creates the paired state needed to measure
    # integration. It happens only after the graph plan has been frozen.
    _phase(on_phase, "root-cut")
    root_gate = source_gate(prefix, (selected_root,))
    root_cut = gated_forward_cache(model, screen.clean_cache, root_gate)
    edges = attach_cut_edge_codes(
        model,
        screen.edges,
        root_cut,
        gradients,
        root_gate.source_mask,
        clean_cache=screen.clean_cache,
        query_chunk=query_chunk,
    )
    flow = replace(
        screen,
        corrupt_margin=margin(model, root_cut, target),
        edges=edges,
        stages=stage_trace(screen.clean_cache, root_cut, gradients),
        corrupt_cache=root_cut,
        corrupt_source_mask=root_gate.source_mask,
    )
    tolerance = intervention_tolerance(model)
    roots, all_cut_margin = _root_effects(
        model,
        flow,
        prefix,
        plan,
        confirm=budget.confirm,
    )
    selected_effect = next(
        effect for effect in roots if effect.unit_id == selected_root
    )
    effect_direction = 1.0 if flow.pair_effect >= 0 else -1.0
    selected_root_confirmed = bool(
        selected_effect.evaluated
        and selected_effect.route_mass > 0
        and selected_effect.causal_score > tolerance
    )

    corridor = flow.edges.select(plan.corridor_edge_index)
    effect = _unevaluated_corridor(flow, corridor, tolerance)
    corridor_evaluated = False
    carriers = _planned_carriers(flow, plan, tolerance)
    carrier_evaluated_count = 0
    if budget.confirm and corridor.count:
        _phase(on_phase, "confirm-corridor")
        effect = confirm_corridor(model, flow, corridor)
        corridor_evaluated = True
    corridor_confirmed = bool(
        corridor_evaluated
        and selected_root_confirmed
        and effect.restoration_valid
        and complete_mediation_confirmed(
            effect.necessity,
            effect.sufficiency,
            effect.blocked_sufficiency,
            direction=effect_direction,
            tolerance=tolerance,
        )
    )
    if budget.confirm and plan.hubs:
        _phase(on_phase, "confirm-hub")
        chosen = plan.hubs[0]
        carriers = (
            confirm_planned_carriers(
                model,
                flow,
                (
                    PlannedCarrier(
                        chosen.layer,
                        chosen.position,
                        chosen.route_mass,
                        chosen.signed_action,
                    ),
                ),
                effect_direction=effect_direction,
            )
            + carriers[1:]
        )
        carrier_evaluated_count = 1

    _phase(on_phase, "analyze")
    dynamics = HeadResolvedRouteModel(local_window).analyze(
        model,
        flow,
        prefix,
        root_unit_id=selected_root,
    )
    _phase(on_phase, "complete")
    return NativeTargetAudit(
        world=prefix,
        flow=flow,
        plan=plan,
        throughput=plan.throughput,
        corridor=corridor,
        effect=effect,
        corridor_evaluated=corridor_evaluated,
        corridor_confirmed=corridor_confirmed,
        roots=roots,
        all_evidence_cut_margin=all_cut_margin,
        selected_root_unit_id=selected_root,
        selected_root_effect=selected_effect,
        selected_root_confirmed=selected_root_confirmed,
        carriers=carriers,
        carrier_evaluated_count=carrier_evaluated_count,
        dynamics=dynamics,
    )
