import torch

from experiments.directed_route_hypergraph.corruption import corrupt_graph
from experiments.directed_route_hypergraph.hypergraph import layer_hypergraph
from experiments.grounded_route.tests.helpers import make_graph


def test_zero_corruption_returns_the_clean_graph():
    graph = make_graph()
    corrupted = corrupt_graph(
        graph,
        incidence_dropout=0.0,
        head_dropout=0.0,
        generator=torch.Generator().manual_seed(3),
    )

    assert corrupted is graph


def test_hidden_incidence_mass_moves_to_unresolved_without_changing_rows():
    graph = make_graph()
    corrupted = corrupt_graph(
        graph,
        incidence_dropout=0.5,
        head_dropout=0.25,
        generator=torch.Generator().manual_seed(7),
    )

    assert corrupted.edge_count < graph.edge_count
    assert torch.equal(corrupted.diagonal, graph.diagonal)
    for layer in range(graph.layer_count):
        view = layer_hypergraph(corrupted, layer, "cpu")
        retained = torch.zeros(view.hyperedge_count)
        retained.index_add_(0, view.hyperedge, view.weight)
        assert torch.allclose(
            retained + view.diagonal + view.unresolved,
            torch.ones_like(retained),
            atol=1e-6,
            rtol=1e-6,
        )


def test_head_dropout_removes_complete_layer_head_channels():
    graph = make_graph()
    corrupted = corrupt_graph(
        graph,
        incidence_dropout=0.0,
        head_dropout=0.75,
        generator=torch.Generator().manual_seed(11),
    )
    clean_channels = graph.edges.layer * graph.head_count + graph.edges.head
    kept_channels = corrupted.edges.layer * graph.head_count + corrupted.edges.head
    removed_channels = set(clean_channels.tolist()) - set(kept_channels.tolist())

    assert removed_channels
    assert not removed_channels.intersection(kept_channels.tolist())
