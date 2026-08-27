from unittest.mock import patch

import pytest
import torch

from experiments.directed_route_hypergraph.config import LearningConfig, ModelConfig
from experiments.directed_route_hypergraph.corruption import corrupt_graph
from experiments.directed_route_hypergraph.learning import (
    endpoint_layout_loss,
    held_out_endpoint_loss,
    sample_held_out_endpoints,
    self_supervised_loss,
)
from experiments.directed_route_hypergraph.model import DirectedRouteHypergraphEncoder
from experiments.grounded_route.aggregation import lag_bucket
from experiments.grounded_route.tests.helpers import make_graph


def make_model(graph):
    return DirectedRouteHypergraphEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(dropout=0.0),
    )


def endpoint_config(graph, **updates):
    values = dict(
        positive_edges_per_graph=graph.edge_count,
        holdout_fraction=1.0,
        negative_count=1,
        negative_attempt_factor=64,
        incidence_dropout=0.0,
        head_dropout=0.0,
        flow_weight=0.0,
        layout_weight=0.0,
        variance_weight=0.0,
        kl_weight=0.0,
    )
    values.update(updates)
    return LearningConfig(**values)


def test_heldout_sampler_matches_role_and_lag_and_returns_typed_nonedges():
    graph = make_graph()
    pairs = sample_held_out_endpoints(
        graph,
        endpoint_config(graph),
        torch.Generator().manual_seed(3),
    )

    assert pairs.count > 0
    edge = pairs.edge
    positive_source = graph.edges.source[edge]
    negative_source = pairs.negative_source
    target = graph.edges.target[edge]
    assert torch.equal(
        positive_source < graph.response_start,
        negative_source < graph.response_start,
    )
    assert torch.equal(
        lag_bucket(target - positive_source, 63),
        lag_bucket(target - negative_source, 63),
    )
    assert bool((negative_source < target).all())

    observed = {
        (int(layer), int(head), int(target), int(source))
        for source, target, layer, head in zip(
            graph.edges.source,
            graph.edges.target,
            graph.edges.layer,
            graph.edges.head,
            strict=True,
        )
    }
    for pair, negative in zip(edge.tolist(), negative_source.tolist(), strict=True):
        key = (
            int(graph.edges.layer[pair]),
            int(graph.edges.head[pair]),
            int(graph.edges.target[pair]),
            negative,
        )
        assert key not in observed


def test_heldout_endpoints_are_absent_from_student_without_mutating_clean_graph():
    graph = make_graph()
    original = tuple(tensor.clone() for tensor in (
        graph.edges.source,
        graph.edges.target,
        graph.edges.layer,
        graph.edges.head,
        graph.edges.weight,
        graph.unresolved,
    ))
    pairs = sample_held_out_endpoints(
        graph,
        endpoint_config(graph),
        torch.Generator().manual_seed(5),
    )
    heldout = torch.unique(pairs.edge)
    result = corrupt_graph(
        graph,
        incidence_dropout=0.0,
        head_dropout=0.0,
        generator=torch.Generator().manual_seed(7),
        forced_edge=heldout,
    )

    assert heldout.numel() > 0
    student_endpoints = {
        (int(layer), int(head), int(target), int(source))
        for source, target, layer, head in zip(
            result.graph.edges.source,
            result.graph.edges.target,
            result.graph.edges.layer,
            result.graph.edges.head,
            strict=True,
        )
    }
    for edge in heldout.tolist():
        key = (
            int(graph.edges.layer[edge]),
            int(graph.edges.head[edge]),
            int(graph.edges.target[edge]),
            int(graph.edges.source[edge]),
        )
        assert key not in student_endpoints
    assert result.graph.edge_count == graph.edge_count - heldout.numel()
    assert torch.equal(result.graph.unresolved, graph.unresolved)
    for current, expected in zip(
        (
            graph.edges.source,
            graph.edges.target,
            graph.edges.layer,
            graph.edges.head,
            graph.edges.weight,
            graph.unresolved,
        ),
        original,
        strict=True,
    ):
        assert torch.equal(current, expected)


def test_heldout_endpoint_bpr_backpropagates_through_final_encoder_state():
    torch.manual_seed(11)
    graph = make_graph()
    pairs = sample_held_out_endpoints(
        graph,
        endpoint_config(graph),
        torch.Generator().manual_seed(13),
    )
    result = corrupt_graph(
        graph,
        incidence_dropout=0.0,
        head_dropout=0.0,
        generator=torch.Generator().manual_seed(17),
        forced_edge=torch.unique(pairs.edge),
    )
    model = make_model(graph).train()
    output = model(result.graph, masked_mass=result.masked_mass)
    objective = held_out_endpoint_loss(model, output, graph, pairs)

    objective.loss.backward()

    assert objective.pair_count == pairs.count
    assert objective.heldout_edge_count == torch.unique(pairs.edge).numel()
    assert torch.isfinite(objective.loss) and float(objective.loss.detach()) > 0
    query_gradient = model.route_query[-1].weight.grad
    key_gradient = model.route_key[-1].weight.grad
    encoder_gradient = model.source_to_hyperedge.source_projection[0][-1].weight.grad
    assert query_gradient is not None and bool(query_gradient.abs().sum() > 0)
    assert key_gradient is not None and bool(key_gradient.abs().sum() > 0)
    assert encoder_gradient is not None and bool(encoder_gradient.abs().sum() > 0)


def test_endpoint_only_objective_bypasses_clean_flow_and_layout_teachers():
    torch.manual_seed(19)
    graph = make_graph()
    model = make_model(graph).train()
    config = endpoint_config(graph)
    with patch(
        "experiments.directed_route_hypergraph.learning.ordered_flow",
        side_effect=AssertionError("flow teacher must be bypassed"),
    ), patch(
        "experiments.directed_route_hypergraph.learning.ordered_endpoint_layout",
        side_effect=AssertionError("layout teacher must be bypassed"),
    ):
        output = self_supervised_loss(
            model,
            graph,
            config,
            torch.Generator().manual_seed(23),
        )
    output.loss.backward()

    assert output.pair_count > 0
    assert output.heldout_edge_count > 0
    assert output.masked_edge_count == output.heldout_edge_count
    assert output.masked_mass_total > 0
    assert torch.allclose(output.loss, output.endpoint)
    assert output.flow.item() == 0.0
    assert output.layout.item() == 0.0
    assert output.layout_row_count == 0
    assert output.layout_self_row_count == 0
    assert output.layout_external_row_count == 0
    assert model.route_query[-1].weight.grad is not None
    assert model.flow_readout[-1].weight.grad is None
    assert model.layout_query[-1].weight.grad is None
    assert model.layout_key[-1].weight.grad is None


def test_balanced_endpoint_layout_loss_is_batching_invariant():
    torch.manual_seed(29)
    graph = make_graph()
    model = make_model(graph).eval()
    output = model(graph)

    one = endpoint_layout_loss(model, output, graph, rows_per_batch=1)
    all_rows = endpoint_layout_loss(model, output, graph, rows_per_batch=10_000)

    assert torch.allclose(one.loss, all_rows.loss, atol=1e-6, rtol=1e-6)
    assert torch.allclose(one.sink, all_rows.sink, atol=1e-6, rtol=1e-6)
    assert torch.allclose(one.self_mass, all_rows.self_mass, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        one.external_endpoint,
        all_rows.external_endpoint,
        atol=1e-6,
        rtol=1e-6,
    )
    assert one.candidate_count == all_rows.candidate_count
    assert one.row_count == graph.response_count
    assert one.self_row_count == graph.response_count
    assert one.external_row_count == graph.response_count


def test_endpoint_layout_loss_alone_updates_encoder_and_layout_decoder():
    torch.manual_seed(31)
    graph = make_graph()
    model = make_model(graph).train()
    output = model(graph)
    objective = endpoint_layout_loss(model, output, graph, rows_per_batch=3)

    objective.loss.backward()

    encoder_gradient = model.source_to_hyperedge.source_projection[0][-1].weight.grad
    query_gradient = model.layout_query[-1].weight.grad
    key_gradient = model.layout_key[-1].weight.grad
    assert encoder_gradient is not None and bool(encoder_gradient.abs().sum() > 0)
    assert query_gradient is not None and bool(query_gradient.abs().sum() > 0)
    assert key_gradient is not None and bool(key_gradient.abs().sum() > 0)


def test_endpoint_layout_limit_fails_before_allocating_an_oversized_target():
    graph = make_graph()
    model = make_model(graph).eval()
    output = model(graph)

    with pytest.raises(ValueError, match="layout_max_elements"):
        endpoint_layout_loss(model, output, graph, max_elements=1)


def test_endpoint_layout_relay_work_limit_fails_before_teacher_rollout():
    graph = make_graph()
    model = make_model(graph).eval()
    output = model(graph)

    with patch(
        "experiments.directed_route_hypergraph.learning.ordered_endpoint_layout",
        side_effect=AssertionError("teacher must not run above the work limit"),
    ), pytest.raises(ValueError, match="layout_max_work_elements"):
        endpoint_layout_loss(model, output, graph, max_work_elements=1)
