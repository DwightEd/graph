from dataclasses import replace

import torch

from experiments.directed_route_hypergraph.corruption import corrupt_graph
from experiments.directed_route_hypergraph.hypergraph import layer_hypergraph
from experiments.grounded_route.graph import TokenEdges, endpoint_storage_keys
from experiments.grounded_route.tests.helpers import make_graph


def retained_mass(view):
    retained = torch.zeros(view.hyperedge_count, dtype=view.weight.dtype)
    retained.index_add_(0, view.hyperedge, view.weight)
    return retained


def test_zero_corruption_returns_clean_graph_and_an_empty_mask_bucket():
    graph = make_graph()
    result = corrupt_graph(
        graph,
        incidence_dropout=0.0,
        head_dropout=0.0,
        generator=torch.Generator().manual_seed(3),
    )

    assert result.graph is graph
    assert result.masked_edge.numel() == 0
    assert torch.equal(result.masked_mass, torch.zeros_like(graph.unresolved))


def test_hidden_mass_is_separate_from_native_unresolved_and_conserved():
    graph = make_graph()
    original_unresolved = graph.unresolved.clone()
    result = corrupt_graph(
        graph,
        incidence_dropout=0.5,
        head_dropout=0.25,
        generator=torch.Generator().manual_seed(7),
    )

    assert result.graph.edge_count < graph.edge_count
    assert torch.equal(result.graph.diagonal, graph.diagonal)
    assert torch.equal(result.graph.unresolved, original_unresolved)
    assert torch.equal(graph.unresolved, original_unresolved)
    assert bool((result.masked_mass > 0).any())
    for layer in range(graph.layer_count):
        view = layer_hypergraph(
            result.graph,
            layer,
            "cpu",
            masked_mass=result.masked_mass,
        )
        assert torch.allclose(
            retained_mass(view)
            + view.diagonal
            + view.unresolved
            + view.masked,
            torch.ones(view.hyperedge_count),
            atol=1e-6,
            rtol=1e-6,
        )


def test_repeated_forced_edge_is_masked_exactly_once():
    graph = make_graph()
    edge = 5
    expected_mass = graph.edges.weight[edge]
    expected_row = (
        graph.edges.target[edge] - graph.response_start,
        graph.edges.layer[edge],
        graph.edges.head[edge],
    )
    result = corrupt_graph(
        graph,
        incidence_dropout=0.0,
        head_dropout=0.0,
        generator=torch.Generator().manual_seed(9),
        forced_edge=torch.tensor([edge, edge, edge]),
    )

    assert result.graph.edge_count == graph.edge_count - 1
    assert torch.equal(result.masked_edge, torch.tensor([edge]))
    assert torch.allclose(result.masked_mass.sum(), expected_mass)
    assert torch.allclose(result.masked_mass[expected_row], expected_mass)
    assert torch.equal(result.graph.unresolved, graph.unresolved)
    assert graph.edge_count == make_graph().edge_count


def test_forced_endpoint_masks_all_duplicate_storage_entries():
    graph = make_graph()
    edge = 5
    edges = graph.edges
    weight = edges.weight.clone()
    weight[edge] *= 0.5
    duplicated = TokenEdges(
        source=torch.cat((edges.source, edges.source[edge : edge + 1])),
        target=torch.cat((edges.target, edges.target[edge : edge + 1])),
        layer=torch.cat((edges.layer, edges.layer[edge : edge + 1])),
        head=torch.cat((edges.head, edges.head[edge : edge + 1])),
        weight=torch.cat((weight, weight[edge : edge + 1])),
    )
    duplicated_graph = replace(graph, edges=duplicated).check().canonicalize()
    key = endpoint_storage_keys(duplicated_graph)
    duplicate_position = int(
        torch.nonzero(key[1:] == key[:-1], as_tuple=False)[0].item()
    ) + 1
    duplicate_key = key[duplicate_position]
    duplicate_index = torch.nonzero(
        key == duplicate_key,
        as_tuple=False,
    ).flatten()

    result = corrupt_graph(
        duplicated_graph,
        incidence_dropout=0.0,
        head_dropout=0.0,
        generator=torch.Generator().manual_seed(13),
        forced_edge=duplicate_index[:1],
    )

    remaining_key = endpoint_storage_keys(result.graph)
    assert int((key == duplicate_key).sum()) == 2
    assert not bool((remaining_key == duplicate_key).any())
    assert result.masked_edge.numel() == 2


def test_head_dropout_removes_complete_layer_head_channels():
    graph = make_graph()
    result = corrupt_graph(
        graph,
        incidence_dropout=0.0,
        head_dropout=0.75,
        generator=torch.Generator().manual_seed(11),
    )
    clean_channels = graph.edges.layer * graph.head_count + graph.edges.head
    kept_channels = (
        result.graph.edges.layer * graph.head_count + result.graph.edges.head
    )
    removed_channels = set(clean_channels.tolist()) - set(kept_channels.tolist())

    assert removed_channels
    assert not removed_channels.intersection(kept_channels.tolist())
