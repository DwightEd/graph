from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from experiments.directed_route_hypergraph.config import LearningConfig, ModelConfig
from experiments.directed_route_hypergraph.learning import (
    SelectedRows,
    endpoint_layout_loss,
    row_candidates,
    row_distribution_loss,
    sample_rows,
    self_supervised_loss,
)
from experiments.directed_route_hypergraph.model import DirectedRouteHypergraphEncoder
from experiments.grounded_route.graph import TokenEdges
from experiments.grounded_route.tests.helpers import make_graph


def make_model(graph):
    return DirectedRouteHypergraphEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(dropout=0.0),
    )


def remove_first_row_endpoints(graph):
    selected = (
        (graph.edges.target == graph.response_start)
        & (graph.edges.layer == 0)
        & (graph.edges.head == 0)
    )
    removed_mass = graph.edges.weight[selected].sum()
    keep = ~selected
    unresolved = graph.unresolved.clone()
    unresolved[0, 0, 0] += removed_mass
    return replace(
        graph,
        edges=TokenEdges(
            source=graph.edges.source[keep],
            target=graph.edges.target[keep],
            layer=graph.edges.layer[keep],
            head=graph.edges.head[keep],
            weight=graph.edges.weight[keep],
        ),
        unresolved=unresolved,
    ).check()


def test_row_sampling_covers_all_rows_and_subsamples_without_replacement():
    graph = make_graph()
    total = graph.response_count * graph.layer_count * graph.head_count
    all_rows = sample_rows(graph, total + 1, torch.Generator().manual_seed(3))
    subset = sample_rows(graph, 7, torch.Generator().manual_seed(5))

    assert torch.equal(all_rows.row, torch.arange(total))
    assert subset.count == 7
    assert len(torch.unique(subset.row)) == 7
    assert bool((subset.row >= 0).all()) and bool((subset.row < total).all())


def test_candidates_are_only_retained_endpoints_self_and_unresolved():
    graph = make_graph()
    total = graph.response_count * graph.layer_count * graph.head_count
    selected = sample_rows(graph, total, torch.Generator().manual_seed(7))
    candidates = row_candidates(graph, selected)
    mass = torch.zeros(total)
    mass.index_add_(0, candidates.group, candidates.weight)

    assert candidates.endpoint_count == graph.edge_count
    assert candidates.count == graph.edge_count + 2 * total
    assert torch.equal(candidates.source, graph.edges.source)
    assert torch.allclose(mass, torch.ones(total), atol=1e-6, rtol=1e-6)


def test_self_supervised_objective_backpropagates_without_labels_or_nonedges():
    torch.manual_seed(11)
    graph = make_graph()
    total = graph.response_count * graph.layer_count * graph.head_count
    model = make_model(graph).train()
    output = self_supervised_loss(
        model,
        graph,
        LearningConfig(rows_per_graph=total, variance_weight=0.05),
        torch.Generator().manual_seed(13),
    )
    output.loss.backward()

    assert output.row_count == total
    assert output.candidate_count == graph.edge_count + 2 * total
    assert torch.isfinite(output.loss)
    assert float(output.row.detach()) > 0
    assert float(output.flow.detach()) > 0
    assert float(output.layout.detach()) > 0
    assert torch.allclose(
        output.layout,
        output.layout_sink + output.layout_self + output.layout_external,
    )
    assert output.layout_row_count == graph.response_count
    assert output.layout_self_row_count == graph.response_count
    assert output.layout_external_row_count == graph.response_count
    assert output.layout_candidate_count > output.layout_row_count
    assert model.route_query[-1].weight.grad is not None
    assert model.bucket_key.grad is not None
    assert model.flow_readout[-1].weight.grad is not None
    assert model.layout_query[-1].weight.grad is not None
    assert model.layout_key[-1].weight.grad is not None
    assert model.layout_unresolved_key.grad is not None
    assert model.source_to_hyperedge.source_projection[0][-1].weight.grad is not None


def test_empty_retained_row_still_has_self_and_unresolved_likelihood():
    torch.manual_seed(17)
    graph = remove_first_row_endpoints(make_graph())
    model = make_model(graph).train()
    output = model(graph, return_layer_input=True)
    selected = SelectedRows(torch.tensor([0], dtype=torch.long))
    candidates = row_candidates(graph, selected)
    objective = row_distribution_loss(model, output, graph, selected)

    assert candidates.endpoint_count == 0
    assert candidates.count == 2
    assert torch.allclose(candidates.weight.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isfinite(objective.loss)
    assert float(objective.loss.detach()) > 0
    objective.loss.backward()
    assert model.bucket_key.grad is not None


def test_balanced_endpoint_layout_loss_is_batching_invariant():
    torch.manual_seed(19)
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
    torch.manual_seed(29)
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


def test_zero_flow_and_layout_weights_bypass_both_teacher_targets():
    torch.manual_seed(31)
    graph = make_graph()
    model = make_model(graph).train()
    config = LearningConfig(flow_weight=0.0, layout_weight=0.0)
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
            torch.Generator().manual_seed(37),
        )
    output.loss.backward()

    assert output.flow.item() == 0.0
    assert output.layout.item() == 0.0
    assert output.layout_row_count == 0
    assert output.layout_self_row_count == 0
    assert output.layout_external_row_count == 0
    assert model.flow_readout[-1].weight.grad is None
    assert model.layout_query[-1].weight.grad is None
    assert model.layout_key[-1].weight.grad is None


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
