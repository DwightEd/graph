from __future__ import annotations

from dataclasses import replace

import torch

from experiments.reanchor_flow.native import audit_native_target
from experiments.reanchor_flow.route_model import (
    CHANNEL_NAMES,
    EVIDENCE,
    RESPONSE,
    UNOBSERVED,
    HeadResolvedRouteModel,
)
from experiments.reanchor_flow.tests.test_native import native_world
from experiments.reanchor_flow.throughput import transition_probabilities

from .etcc_helpers import tiny_model


def _audit(coverage: float = 1.0):
    model = tiny_model()
    world = native_world(model)
    audit = audit_native_target(
        model,
        world,
        world.targets[0],
        "message",
        carrier_scope="all",
        coverage=coverage,
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=1,
    )
    dynamics = HeadResolvedRouteModel(local_window=2).analyze(
        model, audit.flow, audit.world
    )
    return model, audit, dynamics


def test_route_model_keeps_layer_and_head_axes_and_conserves_registers() -> None:
    model, audit, dynamics = _audit()
    layers = model.config.num_hidden_layers
    heads = model.config.num_attention_heads
    rows = len(audit.flow.row_position)
    tokens = len(audit.world.units.token_unit_id)

    assert CHANNEL_NAMES == (
        "evidence",
        "other_prompt",
        "response",
        "unobserved",
    )
    assert dynamics.node_register.shape == (layers + 1, tokens, 4)
    assert dynamics.edge_register.shape == (audit.flow.edges.count, 4)
    assert dynamics.head_transport.shape == (layers, heads, rows, 4)
    assert dynamics.head_gradient_action.shape == (layers, heads, rows, 4)
    assert dynamics.head_direct_evidence.shape == (layers, heads, rows, 2)
    assert dynamics.head_local_response.shape == (layers, heads, rows, 2)
    assert dynamics.head_integration.shape == (layers, heads, rows, 4)
    assert dynamics.head_backward_distance.shape == (layers, heads, rows)
    assert dynamics.head_span.shape == (layers, heads)
    assert dynamics.source_reuse.shape == (layers, heads, tokens, 2)
    assert dynamics.stage_presence.shape == (layers, len(dynamics.stage_position), 3)
    assert dynamics.stage_gradient_action.shape == dynamics.stage_presence.shape

    torch.testing.assert_close(
        dynamics.node_register.sum(-1),
        torch.ones(layers + 1, tokens),
        atol=2e-5,
        rtol=2e-5,
    )
    probability, _ = transition_probabilities(audit.flow, tokens)
    torch.testing.assert_close(
        dynamics.edge_register.sum(-1), probability, atol=2e-6, rtol=2e-6
    )
    assert bool((dynamics.head_integration[..., 2] <= 1 + 2e-5).all())


def test_route_channels_distinguish_direct_evidence_from_local_response() -> None:
    _, audit, dynamics = _audit()
    initial = dynamics.node_register[0]
    evidence_positions = audit.world.units.positions(audit.world.evidence_unit_id)
    assert bool((initial[evidence_positions, EVIDENCE] == 1).all())
    response_position = torch.arange(
        audit.world.response_start, len(audit.world.units.token_unit_id)
    )
    assert bool((initial[response_position, RESPONSE] == 1).all())

    direct = dynamics.head_direct_evidence[..., 0]
    local = dynamics.head_local_response[..., 0]
    assert float(direct.sum()) > 0
    assert float(local.sum()) > 0


def test_pruned_transport_is_explicitly_unobserved_not_renormalized() -> None:
    _, _, dynamics = _audit(coverage=0.5)
    assert float(dynamics.node_register[1:, :, UNOBSERVED].sum()) > 0
    torch.testing.assert_close(
        dynamics.node_register.sum(-1),
        torch.ones_like(dynamics.node_register[..., 0]),
        atol=2e-5,
        rtol=2e-5,
    )


def test_message_aggregation_detects_cancellation_inside_one_head() -> None:
    model, audit, _ = _audit()
    edges = audit.flow.edges
    coordinate = torch.stack(
        (edges.layer.long(), edges.head.long(), edges.target.long()), dim=1
    )
    _, inverse, count = torch.unique(
        coordinate, dim=0, return_inverse=True, return_counts=True
    )
    group = int(torch.nonzero(count >= 2, as_tuple=False)[0])
    selected = torch.nonzero(inverse == group, as_tuple=False).flatten()[:2]
    layer, head, position = coordinate[selected[0]].tolist()

    clean_code = torch.zeros_like(edges.clean_code)
    corrupt_code = torch.zeros_like(edges.corrupt_code)
    vector = torch.ones(clean_code.shape[1])
    clean_code[selected[0]] = vector
    clean_code[selected[1]] = -vector
    delta_norm = torch.zeros_like(edges.delta_message_norm)
    delta_norm[selected] = 1
    zero_action = torch.zeros_like(edges.clean_target_score)
    modified = replace(
        audit.flow,
        edges=replace(
            edges,
            clean_code=clean_code,
            corrupt_code=corrupt_code,
            delta_message_norm=delta_norm,
            clean_target_score=zero_action,
            corrupt_target_score=zero_action,
        ),
    )
    dynamics = HeadResolvedRouteModel(local_window=2).analyze(
        model, modified, audit.world
    )
    slot = int(torch.nonzero(dynamics.row_position == position, as_tuple=False)[0])
    budget, net, coherence, _ = dynamics.head_integration[layer, head, slot]
    assert float(budget) == 2
    assert float(net) < 1e-6
    assert float(coherence) < 1e-6


def test_reanchor_candidates_are_exact_head_nodes_not_head_means() -> None:
    model, audit, dynamics = _audit()
    analyzer = HeadResolvedRouteModel(local_window=2)
    events = analyzer.reanchor_events(dynamics, limit=8)
    assert events
    assert len(events) <= 8
    for event in events:
        assert 0 <= event.layer < model.config.num_hidden_layers
        assert 0 <= event.head < model.config.num_attention_heads
        assert event.position in dynamics.row_position.tolist()
        assert event.score == abs(event.evidence_gradient_action)
        assert event.evidence_transport > 0

    local, global_ = analyzer.head_groups(dynamics)
    assert local.shape == global_.shape == (
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
    )
    assert bool(local.any())
    assert bool(global_.any())

    compact = analyzer.compact_arrays(
        dynamics, response_start=audit.world.response_start, limit=8
    )
    assert int(compact["route_model_schema"]) == 1
    assert compact["route_head_span"].shape == local.shape
    assert len(compact["reanchor_event_head"]) == len(events)
    assert len(compact["local_event_head"]) <= 8
    assert len(compact["silent_event_head"]) <= 8
    assert len(compact["reuse_event_head"]) <= 8
