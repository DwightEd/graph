from __future__ import annotations

from dataclasses import replace

import torch

from experiments.reanchor_flow.flow import SourceLocationBuckets
from experiments.reanchor_flow.reanchor_timeline import ReanchorTimelineAuditor
from experiments.reanchor_flow.tests.test_route_model import _audit


def _location_with_prompt_switch(audit, dynamics) -> SourceLocationBuckets:
    layers, heads, rows = dynamics.head_transport.shape[:3]
    shape = (layers, heads, rows, 4)
    transport = torch.zeros(shape)
    action = torch.zeros(shape)
    position = torch.full(shape, -1, dtype=torch.int32)
    unit = torch.full(shape, -1, dtype=torch.int32)
    peak = torch.zeros(shape)
    source_action = torch.zeros(shape)
    attention = torch.zeros(shape)

    layer, head = 1, 2
    previous = int(torch.nonzero(dynamics.row_position == 4)[0])
    current = int(torch.nonzero(dynamics.row_position == 5)[0])
    root_position = int(audit.world.units.positions((audit.selected_root_unit_id,))[0])
    root_unit = int(audit.world.units.token_unit_id[root_position])

    # Bucket order: prompt evidence, other prompt, remote response, recent local.
    transport[layer, head, previous, 0] = 0.5
    transport[layer, head, previous, 3] = 4.0
    transport[layer, head, current, 0] = 3.0
    transport[layer, head, current, 3] = 0.2
    action[layer, head, current, 0] = 1.5
    position[layer, head, :, 0] = root_position
    unit[layer, head, :, 0] = root_unit
    position[layer, head, previous, 3] = 4
    unit[layer, head, previous, 3] = int(audit.world.units.token_unit_id[4])
    peak.copy_(transport)
    source_action.copy_(action)
    attention.copy_(transport)
    return SourceLocationBuckets(
        local_window=dynamics.local_window,
        attention=attention,
        transport=transport,
        downstream_action=action,
        source_position=position,
        source_unit_id=unit,
        source_attention=attention.clone(),
        source_transport=peak,
        source_downstream_action=source_action,
    )


def test_timeline_finds_head_specific_local_to_prompt_switch() -> None:
    _, audit, dynamics = _audit()
    location = _location_with_prompt_switch(audit, dynamics)
    flow = replace(audit.flow, source_location=location)

    timeline = ReanchorTimelineAuditor.audit(flow, dynamics, audit.world, limit=8)

    assert len(timeline.candidates) == 1
    candidate = timeline.candidates[0]
    assert (candidate.layer, candidate.head, candidate.position) == (1, 2, 5)
    assert candidate.source_kind == "prompt_evidence"
    assert candidate.previous_local_source_position == 4
    assert candidate.previous_local_transport == 4
    assert candidate.prompt_transport == 3
    assert candidate.dominance_flip
    assert candidate.current_target_match
    assert candidate.long_range_immediate_action == 1.5
    assert candidate.bucket_immediate_action == 1.5
    assert candidate.source_immediate_action == 1.5
    expected_delta = 3 / 3.2 - 0.5 / 4.5
    assert abs(candidate.switch_delta - expected_delta) < 1e-6
    expected_rise = (3 - 0.5) / (3 + 0.5)
    expected_fall = (4 - 0.2) / (4 + 0.2)
    assert abs(candidate.score - (expected_rise * expected_fall) ** 0.5) < 1e-6
    assert int(torch.count_nonzero(timeline.score)) == 1


def test_timeline_reads_anchor_lineage_from_source_position_at_event_layer() -> None:
    _, audit, dynamics = _audit()
    root_position = int(audit.world.units.positions((audit.selected_root_unit_id,))[0])
    node_register = dynamics.node_register.clone()
    node_register[:, root_position] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    node_register[1, root_position] = torch.tensor([0.625, 0.0, 0.0, 0.375])
    node_register[1, 5] = torch.tensor([0.25, 0.0, 0.0, 0.75])
    dynamics = replace(dynamics, node_register=node_register)
    location = _location_with_prompt_switch(audit, dynamics)

    timeline = ReanchorTimelineAuditor.audit(
        replace(audit.flow, source_location=location),
        dynamics,
        audit.world,
        limit=1,
    )

    candidate = timeline.candidates[0]
    assert (candidate.layer, candidate.source_position) == (1, root_position)
    assert candidate.source_evidence_fraction == 0.625


def test_timeline_selection_does_not_depend_on_target_gradient_action() -> None:
    _, audit, dynamics = _audit()
    location = _location_with_prompt_switch(audit, dynamics)
    location = replace(
        location,
        downstream_action=torch.zeros_like(location.downstream_action),
        source_downstream_action=torch.zeros_like(location.source_downstream_action),
    )
    flow = replace(audit.flow, source_location=location)

    candidate = ReanchorTimelineAuditor.audit(
        flow, dynamics, audit.world, limit=1
    ).candidates[0]

    assert candidate.score > 0
    assert candidate.long_range_downstream_action == 0
    assert candidate.source_downstream_action == 0
    assert candidate.long_range_immediate_action == 0
    assert candidate.source_immediate_action == 0


def test_anchor_kind_uses_bucket_total_not_one_source_peak() -> None:
    _, audit, dynamics = _audit()
    location = _location_with_prompt_switch(audit, dynamics)
    current = int(torch.nonzero(dynamics.row_position == 5)[0])
    transport = location.transport.clone()
    transport[1, 2, current, 1] = 4.0
    source_transport = location.source_transport.clone()
    source_transport[1, 2, current, 1] = 1.0
    source_position = location.source_position.clone()
    source_position[1, 2, current, 1] = 0
    source_unit = location.source_unit_id.clone()
    source_unit[1, 2, current, 1] = int(audit.world.units.token_unit_id[0])
    location = replace(
        location,
        transport=transport,
        source_transport=source_transport,
        source_position=source_position,
        source_unit_id=source_unit,
    )
    flow = replace(audit.flow, source_location=location)

    candidate = ReanchorTimelineAuditor.audit(
        flow, dynamics, audit.world, limit=1
    ).candidates[0]

    assert candidate.source_kind == "other_prompt"
    assert candidate.source_position == 0
    assert candidate.source_evidence_fraction == 0
    assert torch.isnan(torch.tensor(candidate.selected_root_integration_action))


def test_timeline_requires_prepruning_source_buckets() -> None:
    _, audit, dynamics = _audit()
    with torch.no_grad():
        flow = replace(audit.flow, source_location=None)

    try:
        ReanchorTimelineAuditor.audit(flow, dynamics, audit.world)
    except ValueError as error:
        assert "full-row" in str(error)
    else:
        raise AssertionError("sparse edges must not masquerade as a full timeline")
