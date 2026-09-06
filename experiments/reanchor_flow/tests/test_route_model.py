from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

from experiments.reanchor_flow import route_model
from experiments.reanchor_flow.native import audit_native_target
from experiments.reanchor_flow.route_model import (
    CHANNEL_NAMES,
    EVIDENCE,
    RESPONSE,
    UNOBSERVED,
    HeadResolvedRouteModel,
)
from experiments.reanchor_flow.route_plan import RouteBudget
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
        route_budget=RouteBudget(
            edges_per_head=2,
            max_rows=32,
            root_candidates=2,
            hub_candidates=1,
            corridor_edges=8,
        ),
    )
    dynamics = HeadResolvedRouteModel(local_window=2).analyze(
        model,
        audit.flow,
        audit.world,
        root_unit_id=audit.selected_root_unit_id,
    )
    return model, audit, dynamics


def _synthetic_flow(
    audit,
    *,
    layer: Tensor,
    head: Tensor,
    source: Tensor,
    target: Tensor,
    code: Tensor,
    action: Tensor,
):
    count = len(layer)
    template = audit.flow.edges.select(torch.arange(count))
    scalar = torch.ones(count)
    zero = torch.zeros(count)
    vector = torch.empty(count, 0)
    edges = replace(
        template,
        layer=layer.to(torch.int16),
        head=head.to(torch.int16),
        source=source.to(torch.int32),
        target=target.to(torch.int32),
        source_unit=audit.world.units.token_unit_id.index_select(0, source.long()).to(
            torch.int32
        ),
        attention_clean=scalar,
        attention_corrupt=zero,
        score=scalar,
        clean_target_score=action.float(),
        corrupt_target_score=zero,
        selector_score=torch.full((count,), float("nan")),
        content_score=torch.full((count,), float("nan")),
        clean_message_norm=scalar,
        corrupt_message_norm=zero,
        delta_message_norm=scalar,
        clean_code=code.float(),
        corrupt_code=torch.zeros_like(code, dtype=torch.float32),
        clean_message_vector=vector,
        corrupt_message_vector=vector.clone(),
        delta_message_vector=vector.clone(),
    )
    layers = audit.flow.clean_cache.layer_count
    heads = audit.flow.row_total.shape[1]
    row_position = torch.unique(target.long(), sorted=True)
    lookup = {int(position): slot for slot, position in enumerate(row_position)}
    row_total = torch.zeros(layers, heads, len(row_position))
    for edge_layer, edge_head, edge_target in zip(
        layer.tolist(), head.tolist(), target.tolist(), strict=True
    ):
        row_total[edge_layer, edge_head, lookup[edge_target]] += 1
    return replace(
        audit.flow,
        edges=edges,
        row_position=row_position,
        row_total=row_total,
        row_retained=row_total.clone(),
        residual_weight=torch.zeros(layers, len(row_position)),
        stages=None,
    )


def _stage_flow(audit, mlp_sign: float):
    flow = audit.flow
    assert flow.stages is not None
    position = flow.stages.position.long()
    layers = flow.clean_cache.layer_count

    def stage_vector(value: float) -> Tensor:
        result = torch.zeros_like(flow.clean_cache.final_hidden)
        result[position, 0] = value
        return result

    clean = replace(
        flow.clean_cache,
        layer_input={layer: stage_vector(1.0) for layer in range(layers)},
        final_hidden=stage_vector(1.0),
        attention_write={layer: stage_vector(1.0) for layer in range(layers)},
        mlp_write={layer: stage_vector(mlp_sign) for layer in range(layers)},
    )
    cut = replace(
        flow.corrupt_cache,
        layer_input={layer: stage_vector(0.0) for layer in range(layers)},
        final_hidden=stage_vector(0.0),
        attention_write={layer: stage_vector(0.0) for layer in range(layers)},
        mlp_write={layer: stage_vector(0.0) for layer in range(layers)},
    )
    shape = (layers, len(position))
    stages = replace(
        flow.stages,
        state_delta_norm=torch.ones(shape),
        state_score=torch.ones(shape),
        attention_delta_norm=torch.ones(shape),
        attention_score=torch.ones(shape),
        mlp_delta_norm=torch.ones(shape),
        mlp_score=torch.full(shape, mlp_sign),
    )
    return replace(flow, clean_cache=clean, corrupt_cache=cut, stages=stages)


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
    assert dynamics.root_unit_id == audit.selected_root_unit_id
    assert dynamics.edge_register.shape == (audit.flow.edges.count, 4)
    assert dynamics.head_transport.shape == (layers, heads, rows, 4)
    assert dynamics.head_gradient_action.shape == (layers, heads, rows, 4)
    assert dynamics.head_direct_evidence.shape == (layers, heads, rows, 2)
    assert dynamics.head_integration.shape == (layers, heads, rows, 4)
    assert dynamics.cross_head_vector_coherence.shape == (layers, rows)
    assert dynamics.cross_head_functional_agreement.shape == (layers, rows)
    assert dynamics.layer_integration.shape == (layers, rows, 4)
    assert dynamics.evidence_source_reuse.shape == (layers, heads, tokens, 2)
    assert dynamics.response_source_reuse.shape == (layers, heads, tokens, 2)
    assert dynamics.stage_displacement.shape == (
        layers,
        len(dynamics.stage_position),
        3,
    )
    assert dynamics.stage_gradient_action.shape == dynamics.stage_displacement.shape
    assert dynamics.attention_mlp_vector_cosine.shape == (
        layers,
        len(dynamics.stage_position),
    )
    assert dynamics.attention_mlp_functional_agreement.shape == (
        layers,
        len(dynamics.stage_position),
    )
    assert dynamics.state_continuity.shape == (
        layers,
        len(dynamics.stage_position),
    )

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
    assert bool(
        (
            dynamics.head_direct_evidence[..., 0]
            <= dynamics.head_transport[..., EVIDENCE] + 2e-5
        ).all()
    )


def test_route_channels_distinguish_direct_evidence_from_response_provenance() -> None:
    _, audit, dynamics = _audit()
    initial = dynamics.node_register[0]
    evidence_positions = audit.world.units.positions((audit.selected_root_unit_id,))
    assert bool((initial[evidence_positions, EVIDENCE] == 1).all())
    other_evidence = tuple(
        unit_id
        for unit_id in audit.world.evidence_unit_id
        if unit_id != audit.selected_root_unit_id
    )
    if other_evidence:
        other_position = audit.world.units.positions(other_evidence)
        assert bool((initial[other_position, EVIDENCE] == 1).all())
    response_position = torch.arange(
        audit.world.response_start, len(audit.world.units.token_unit_id)
    )
    assert bool((initial[response_position, RESPONSE] == 1).all())

    direct = dynamics.head_direct_evidence[..., 0]
    response = dynamics.head_transport[..., RESPONSE]
    assert float(direct.sum()) > 0
    assert float(response.sum()) > 0


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
        model,
        modified,
        audit.world,
        root_unit_id=audit.selected_root_unit_id,
    )
    slot = int(torch.nonzero(dynamics.row_position == position, as_tuple=False)[0])
    budget, net, coherence, _ = dynamics.head_integration[layer, head, slot]
    assert float(budget) == 2
    assert float(net) < 1e-6
    assert float(coherence) < 1e-6


def test_head_integration_materializes_code_deltas_in_bounded_chunks(
    monkeypatch,
) -> None:
    model, audit, _ = _audit()
    count = 5
    root = int(audit.world.units.positions((audit.selected_root_unit_id,))[0])
    target = audit.flow.target.query_position
    head_dim = audit.flow.edges.clean_code.shape[1]
    code = torch.arange(count * head_dim, dtype=torch.float32).view(count, head_dim)
    flow = _synthetic_flow(
        audit,
        layer=torch.zeros(count, dtype=torch.long),
        head=torch.arange(count) % model.config.num_attention_heads,
        source=torch.full((count,), root),
        target=torch.full((count,), target),
        code=code,
        action=torch.ones(count),
    )
    analyzer = HeadResolvedRouteModel()
    _, edge_slot = analyzer._slots(flow, len(audit.world.units.token_unit_id))
    code_subtractions: list[torch.Size] = []

    class RecordCodeSubtractions(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.sub.Tensor and args[0].ndim == 2:
                code_subtractions.append(args[0].shape)
            return func(*args, **(kwargs or {}))

    monkeypatch.setattr(route_model, "EDGE_CHUNK", 2)
    with RecordCodeSubtractions():
        analyzer._head_integration(
            model,
            flow,
            edge_slot,
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
        )

    assert code_subtractions == [
        torch.Size((2, head_dim)),
        torch.Size((2, head_dim)),
        torch.Size((1, head_dim)),
    ]


def test_selected_root_must_match_the_root_cut_estimand() -> None:
    model, audit, _ = _audit()
    other_root = next(
        unit_id
        for unit_id in audit.world.evidence_unit_id
        if unit_id != audit.selected_root_unit_id
    )
    with pytest.raises(ValueError, match="root-cut cache"):
        HeadResolvedRouteModel().analyze(
            model,
            audit.flow,
            audit.world,
            root_unit_id=other_root,
        )


def test_reduced_scope_preserves_prompt_evidence_on_residual_path() -> None:
    model = tiny_model()
    world = native_world(model)
    audit = audit_native_target(
        model,
        world,
        world.targets[0],
        "message",
        carrier_scope="response",
        coverage=1.0,
        query_chunk=2,
        route_budget=RouteBudget(
            edges_per_head=2,
            max_rows=32,
            root_candidates=2,
            hub_candidates=0,
            corridor_edges=8,
        ),
    )
    root_position = audit.world.units.positions((audit.selected_root_unit_id,))
    represented = set(audit.flow.row_position.tolist())
    omitted = torch.tensor(
        [position for position in root_position.tolist() if position not in represented]
    )
    assert len(omitted)
    dynamics = audit.dynamics
    assert bool((dynamics.node_register[0, omitted, EVIDENCE] == 1).all())
    assert bool((dynamics.node_register[1:, omitted, EVIDENCE] == 1).all())
    assert bool((dynamics.node_register[:, omitted, UNOBSERVED] == 0).all())
    _, residual = transition_probabilities(
        audit.flow, len(audit.world.units.token_unit_id)
    )
    assert bool((residual[:, omitted] == 1).all())


def test_late_head_read_from_any_prompt_evidence_keeps_evidence_lineage() -> None:
    model = tiny_model()
    world = native_world(model)
    audit = audit_native_target(
        model,
        world,
        world.targets[0],
        "message",
        carrier_scope="response",
        coverage=1.0,
        query_chunk=2,
        route_budget=RouteBudget(
            edges_per_head=2,
            max_rows=32,
            root_candidates=2,
            hub_candidates=0,
            corridor_edges=8,
        ),
    )
    other_unit = next(
        unit_id
        for unit_id in audit.world.evidence_unit_id
        if unit_id != audit.selected_root_unit_id
    )
    evidence_position = int(audit.world.units.positions((other_unit,))[0])
    target = audit.flow.target.query_position
    layer = model.config.num_hidden_layers - 1
    head = model.config.num_attention_heads - 1
    code = torch.ones(1, audit.flow.edges.clean_code.shape[1])
    flow = _synthetic_flow(
        audit,
        layer=torch.tensor([layer]),
        head=torch.tensor([head]),
        source=torch.tensor([evidence_position]),
        target=torch.tensor([target]),
        code=code,
        action=torch.tensor([2.0]),
    )

    dynamics = HeadResolvedRouteModel(local_window=2).analyze(
        model,
        flow,
        audit.world,
        root_unit_id=audit.selected_root_unit_id,
    )

    slot = int(torch.nonzero(dynamics.row_position == target)[0])
    assert float(dynamics.node_register[layer, evidence_position, EVIDENCE]) == 1
    assert float(dynamics.head_transport[layer, head, slot, EVIDENCE]) == 1
    assert float(dynamics.head_gradient_action[layer, head, slot, EVIDENCE]) == 2
    assert float(dynamics.head_direct_evidence[layer, head, slot, 0]) == 0


def test_response_hub_preserves_evidence_lineage_for_reanchor_and_reuse() -> None:
    model, audit, _ = _audit()
    root = int(audit.world.units.positions((audit.selected_root_unit_id,))[0])
    hub = audit.world.response_start
    target = audit.flow.target.query_position
    head_dim = audit.flow.edges.clean_code.shape[1]
    code = torch.zeros(2, head_dim)
    code[:, 0] = 1
    flow = _synthetic_flow(
        audit,
        layer=torch.tensor([0, 1]),
        head=torch.tensor([0, 1]),
        source=torch.tensor([root, hub]),
        target=torch.tensor([hub, target]),
        code=code,
        action=torch.tensor([1.0, 2.0]),
    )
    dynamics = HeadResolvedRouteModel(local_window=2).analyze(
        model,
        flow,
        audit.world,
        root_unit_id=audit.selected_root_unit_id,
    )
    target_slot = int(torch.nonzero(dynamics.row_position == target)[0])
    assert float(dynamics.head_transport[1, 1, target_slot, EVIDENCE]) == 1
    assert float(dynamics.head_direct_evidence[1, 1, target_slot, 0]) == 0
    assert float(dynamics.evidence_source_reuse[1, 1, hub, 0]) == 1
    assert float(dynamics.response_source_reuse[1, 1, hub, 0]) == 0


def test_cross_head_vector_coherence_and_functional_agreement_keep_heads() -> None:
    _, audit, _ = _audit()
    model = tiny_model()
    root = int(audit.world.units.positions((audit.selected_root_unit_id,))[0])
    target = audit.flow.target.query_position
    head_dim = audit.flow.edges.clean_code.shape[1]
    code = torch.zeros(2, head_dim)
    code[:, 0] = 1
    base = _synthetic_flow(
        audit,
        layer=torch.tensor([0, 0]),
        head=torch.tensor([0, 1]),
        source=torch.tensor([root, root]),
        target=torch.tensor([target, target]),
        code=code,
        action=torch.tensor([1.0, 1.0]),
    )
    output = model.model.layers[0].self_attn.o_proj.weight
    with torch.no_grad():
        output.zero_()
        output[:head_dim, :head_dim] = torch.eye(head_dim)
        output[:head_dim, head_dim : 2 * head_dim] = torch.eye(head_dim)
    analyzer = HeadResolvedRouteModel()
    aligned = analyzer.analyze(
        model,
        base,
        audit.world,
        root_unit_id=audit.selected_root_unit_id,
    )
    assert float(aligned.cross_head_vector_coherence[0, 0]) == 1
    assert float(aligned.cross_head_functional_agreement[0, 0]) == 1
    torch.testing.assert_close(aligned.head_integration[0, :2, 0, 3], torch.ones(2))

    with torch.no_grad():
        output[:head_dim, head_dim : 2 * head_dim] = -torch.eye(head_dim)
    opposed = replace(
        base,
        edges=replace(
            base.edges,
            clean_target_score=torch.tensor([1.0, -1.0]),
        ),
    )
    cancelled = analyzer.analyze(
        model,
        opposed,
        audit.world,
        root_unit_id=audit.selected_root_unit_id,
    )
    assert float(cancelled.cross_head_vector_coherence[0, 0]) < 1e-6
    assert float(cancelled.cross_head_functional_agreement[0, 0]) == 0
    torch.testing.assert_close(
        cancelled.head_integration[0, :2, 0, 3],
        torch.tensor([1.0, -1.0]),
    )


def test_module_conflict_and_state_continuity_are_stability_not_grounding() -> None:
    model, audit, _ = _audit()
    analyzer = HeadResolvedRouteModel()
    conflict = analyzer.analyze(
        model,
        _stage_flow(audit, -1.0),
        audit.world,
        root_unit_id=audit.selected_root_unit_id,
    )
    torch.testing.assert_close(
        conflict.attention_mlp_vector_cosine,
        -torch.ones_like(conflict.attention_mlp_vector_cosine),
    )
    torch.testing.assert_close(
        conflict.attention_mlp_functional_agreement,
        torch.zeros_like(conflict.attention_mlp_functional_agreement),
    )
    torch.testing.assert_close(
        conflict.state_continuity,
        torch.ones_like(conflict.state_continuity),
    )

    aligned = analyzer.analyze(
        model,
        _stage_flow(audit, 1.0),
        audit.world,
        root_unit_id=audit.selected_root_unit_id,
    )
    torch.testing.assert_close(
        aligned.attention_mlp_vector_cosine,
        torch.ones_like(aligned.attention_mlp_vector_cosine),
    )
    torch.testing.assert_close(
        aligned.attention_mlp_functional_agreement,
        torch.ones_like(aligned.attention_mlp_functional_agreement),
    )


def test_stage_dynamics_never_stacks_full_hidden_vectors(monkeypatch) -> None:
    _, audit, _ = _audit()
    flow = _stage_flow(audit, -1.0)
    hidden_size = flow.clean_cache.final_hidden.shape[-1]
    original_stack = torch.stack
    stacked_inputs: list[tuple[int, ...]] = []

    def record_stack(values, *args, **kwargs):
        values = tuple(values)
        if values:
            stacked_inputs.append(tuple(values[0].shape))
        return original_stack(values, *args, **kwargs)

    monkeypatch.setattr(route_model.torch, "stack", record_stack)
    HeadResolvedRouteModel._stage_dynamics(
        flow,
        flow.clean_cache.layer_count,
    )

    assert not any(
        len(shape) == 2 and shape[-1] == hidden_size for shape in stacked_inputs
    )
