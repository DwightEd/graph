from dataclasses import replace

import torch

from experiments.directed_route_hypergraph.config import LearningConfig, ModelConfig
from experiments.directed_route_hypergraph.learning import (
    SelectedRows,
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
    assert model.route_query[-1].weight.grad is not None
    assert model.bucket_key.grad is not None
    assert model.flow_readout[-1].weight.grad is not None
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
