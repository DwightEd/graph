import torch

from experiments.directed_route_hypergraph.hypergraph import layer_hypergraph
from experiments.grounded_route.tests.helpers import make_graph


def test_incidence_maps_sources_to_rows_and_rows_to_targets():
    graph = make_graph()
    layer = 1
    view = layer_hypergraph(graph, layer, "cpu")
    edges = graph.layer_edges(layer)

    assert view.hyperedge_count == graph.response_count * graph.head_count
    assert view.incidence_count == edges.count
    assert view.source_to_hyperedge.shape == (2, edges.count)
    assert view.hyperedge_to_target.shape == (2, view.hyperedge_count)
    assert torch.equal(view.source, edges.source)
    assert torch.equal(view.target[view.hyperedge], edges.target)
    assert torch.equal(view.head[view.hyperedge], edges.head)
    assert torch.equal(view.hyperedge_to_target[1], view.target)


def test_every_hyperedge_conserves_retained_self_and_unresolved_mass():
    graph = make_graph()
    view = layer_hypergraph(graph, 0, "cpu")
    retained = torch.zeros(view.hyperedge_count)
    retained.index_add_(0, view.hyperedge, view.weight)

    assert torch.allclose(
        retained + view.diagonal + view.unresolved,
        torch.ones(view.hyperedge_count),
        atol=1e-6,
        rtol=1e-6,
    )
