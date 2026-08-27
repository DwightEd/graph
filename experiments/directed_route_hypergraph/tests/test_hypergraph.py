import pytest
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


def test_clean_view_has_no_artificial_mask_and_conserves_native_mass():
    graph = make_graph()
    view = layer_hypergraph(graph, 0, "cpu")
    retained = torch.zeros(view.hyperedge_count)
    retained.index_add_(0, view.hyperedge, view.weight)

    assert torch.equal(view.masked, torch.zeros_like(view.unresolved))
    assert torch.allclose(
        retained + view.diagonal + view.unresolved,
        torch.ones(view.hyperedge_count),
        atol=1e-6,
        rtol=1e-6,
    )


def test_explicit_mask_bucket_is_flattened_without_changing_unresolved():
    graph = make_graph()
    masked_mass = torch.zeros_like(graph.unresolved)
    masked_mass[:, 1] = torch.arange(
        graph.response_count * graph.head_count,
        dtype=masked_mass.dtype,
    ).view(graph.response_count, graph.head_count)
    view = layer_hypergraph(graph, 1, "cpu", masked_mass=masked_mass)

    assert torch.equal(view.masked, masked_mass[:, 1].reshape(-1))
    assert torch.equal(view.unresolved, graph.unresolved[:, 1].reshape(-1))
    assert not torch.equal(view.masked, view.unresolved)


def test_mask_bucket_requires_graph_shaped_mass():
    graph = make_graph()
    with pytest.raises(ValueError, match=r"\[R,L,H\]"):
        layer_hypergraph(
            graph,
            0,
            "cpu",
            masked_mass=torch.zeros(graph.response_count, graph.layer_count),
        )
