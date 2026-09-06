from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiments.reanchor_flow import throughput as throughput_module
from experiments.reanchor_flow.flow import FlowEdges
from experiments.reanchor_flow.route_plan import FlowKernel, RouteBudget, plan_audit
from experiments.reanchor_flow.worlds import SourceUnits, TargetContrast


def _flow_and_world(*, zero_action: bool = False):
    layer = torch.tensor([0, 0, 0, 2, 2], dtype=torch.int16)
    head = torch.tensor([0, 1, 0, 0, 1], dtype=torch.int16)
    source = torch.tensor([0, 1, 2, 3, 1], dtype=torch.int32)
    target = torch.tensor([3, 3, 3, 4, 4], dtype=torch.int32)
    transport = torch.tensor([4.0, 1.0, 1.0, 4.0, 1.0])
    action = torch.zeros(5) if zero_action else torch.tensor([4.0, 1.0, -2.0, 4.0, 1.0])
    nan = torch.full((5,), float("nan"))
    code = torch.zeros(5, 1)
    vector = torch.empty(5, 0)
    token_unit = torch.arange(5)
    edges = FlowEdges(
        layer=layer,
        head=head,
        source=source,
        target=target,
        source_unit=token_unit.index_select(0, source.long()).to(torch.int32),
        attention_clean=transport,
        attention_corrupt=transport.clone(),
        score=transport,
        clean_target_score=action,
        corrupt_target_score=action.clone(),
        selector_score=nan,
        content_score=nan.clone(),
        clean_message_norm=transport,
        corrupt_message_norm=transport.clone(),
        delta_message_norm=torch.zeros(5),
        clean_code=code,
        corrupt_code=code.clone(),
        clean_message_vector=vector,
        corrupt_message_vector=vector.clone(),
        delta_message_vector=vector.clone(),
    )
    rows = torch.arange(5)
    row_total = torch.zeros(3, 2, 5)
    for index in range(edges.count):
        row_total[int(layer[index]), int(head[index]), int(target[index])] += transport[
            index
        ]
    flow = SimpleNamespace(
        edges=edges,
        row_position=rows,
        row_total=row_total,
        row_retained=row_total.clone(),
        residual_weight=torch.zeros(3, 5),
        clean_cache=SimpleNamespace(layer_count=3),
        target=TargetContrast(4, 7, 8, "test"),
    )
    world = SimpleNamespace(
        response_start=3,
        evidence_unit_id=(0, 1),
        units=SourceUnits(
            token_unit,
            ("evidence:0", "evidence:1", "prompt", "response:3", "response:4"),
            ("passage", "passage", "other_prompt", "response", "response"),
        ),
    )
    return flow, world


def test_route_budget_has_the_registered_defaults_and_small_validation() -> None:
    assert RouteBudget() == RouteBudget(
        edges_per_head=2,
        max_rows=256,
        root_candidates=4,
        hub_candidates=8,
        corridor_edges=64,
        confirm=False,
    )
    with pytest.raises(ValueError, match="edges_per_head"):
        RouteBudget(edges_per_head=0)


def test_plan_selects_roots_without_intervention_and_keeps_head_edges() -> None:
    flow, world = _flow_and_world()
    budget = RouteBudget(
        edges_per_head=1,
        max_rows=4,
        root_candidates=2,
        hub_candidates=4,
        corridor_edges=4,
    )
    plan = plan_audit(flow, world, budget)

    assert plan.budget == budget
    assert plan.selected_root_unit_id == 0
    assert not plan.root_selection_fallback
    assert tuple(root.unit_id for root in plan.roots) == (0, 1)
    assert plan.roots[0].score > plan.roots[1].score
    assert plan.roots[0].absolute_action >= abs(plan.roots[0].signed_action)
    assert 0 <= plan.roots[0].functional_agreement <= 1
    assert plan.throughput.root_unit_id == (0,)
    assert plan.backbone_edge_index.tolist() == [0, 3]
    assert plan.backbone_position.tolist() == [0, 3, 3, 4]

    selected = plan.corridor_edge_index.tolist()
    assert 0 in selected
    assert 3 in selected
    groups = [
        (
            int(flow.edges.layer[index]),
            int(flow.edges.head[index]),
            int(flow.edges.target[index]),
        )
        for index in selected
    ]
    assert len(groups) == len(set(groups))


def test_plan_reuses_one_target_reverse_pass_for_all_roots(monkeypatch) -> None:
    flow, world = _flow_and_world()
    actual = throughput_module.transition_probabilities
    calls = 0

    def traced(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(throughput_module, "transition_probabilities", traced)
    plan_audit(flow, world, RouteBudget(corridor_edges=4))

    assert calls == 1


def test_planner_retains_only_a_fresh_selected_root_condition(monkeypatch) -> None:
    flow, world = _flow_and_world()
    actual = FlowKernel.condition
    calls = []

    def traced(self, flow, token_unit_id, root_unit_id):
        result = actual(self, flow, token_unit_id, root_unit_id)
        calls.append((root_unit_id, result))
        return result

    monkeypatch.setattr(FlowKernel, "condition", traced)
    plan = plan_audit(flow, world, RouteBudget(corridor_edges=4))

    assert [root for root, _ in calls] == [(0,), (1,), (0,)]
    assert plan.throughput is calls[-1][1]
    assert all(plan.throughput is not result for _, result in calls[:-1])


def test_corridor_crosses_a_layer_without_a_message_edge_and_hubs_use_nms() -> None:
    flow, world = _flow_and_world()
    plan = plan_audit(
        flow,
        world,
        RouteBudget(max_rows=3, hub_candidates=8, corridor_edges=2),
    )

    assert plan.corridor_edge_index.tolist() == [0, 3]
    response_hubs = [hub for hub in plan.hubs if hub.position == 3]
    assert response_hubs
    assert len(response_hubs) == 1
    assert response_hubs[0].layer in {1, 2}
    assert response_hubs[0].route_mass > 0
    assert response_hubs[0].score > 0
    assert all(hub.position < flow.target.query_position for hub in plan.hubs)


def test_zero_functional_action_falls_back_to_route_mass() -> None:
    flow, world = _flow_and_world(zero_action=True)
    plan = plan_audit(
        flow,
        world,
        RouteBudget(root_candidates=1, hub_candidates=0, corridor_edges=2),
    )

    assert plan.selected_root_unit_id == 0
    assert plan.root_selection_fallback
    assert len(plan.roots) == 1
    assert plan.roots[0].score == 0
    assert plan.roots[0].route_mass == pytest.approx(plan.throughput.root_mass)
    assert plan.hubs == ()


def test_signed_root_action_retains_cross_head_disagreement() -> None:
    flow, world = _flow_and_world()
    opposed = replace(
        flow.edges,
        clean_target_score=torch.tensor([4.0, 1.0, -2.0, -4.0, 1.0]),
    )
    opposed_flow = SimpleNamespace(**vars(flow))
    opposed_flow.edges = opposed
    plan = plan_audit(
        opposed_flow,
        world,
        RouteBudget(root_candidates=2, hub_candidates=0, corridor_edges=2),
    )

    root = next(item for item in plan.roots if item.unit_id == 0)
    assert root.absolute_action >= abs(root.signed_action)
    assert root.functional_agreement < 1
