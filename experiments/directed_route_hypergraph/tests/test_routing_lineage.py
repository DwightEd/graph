from dataclasses import replace

import pytest
import torch

from experiments.directed_route_hypergraph.routing_lineage import (
    DIRECT,
    ENDOGENOUS,
    INDIRECT,
    UNRESOLVED,
    ordered_routing_lineage,
)
from experiments.grounded_route.graph import TokenEdges
from experiments.grounded_route.tests.helpers import make_graph


def lineage_graph(*, heads: int = 1):
    graph = make_graph(
        layers=2,
        heads=heads,
        response_start=1,
        response_count=4,
    )
    source = []
    target = []
    layer = []
    head = []
    weight = []
    for current_head in range(heads):
        source.extend((0, 1, 2))
        target.extend((1, 2, 3))
        layer.extend((0, 1, 1))
        head.extend((current_head,) * 3)
        weight.extend((1.0,) * 3)

    edges = TokenEdges(
        source=torch.tensor(source),
        target=torch.tensor(target),
        layer=torch.tensor(layer),
        head=torch.tensor(head),
        weight=torch.tensor(weight),
    )
    diagonal = torch.zeros((4, 2, heads))
    diagonal[1:3, 0] = 1.0
    diagonal[0, 1] = 1.0
    unresolved = torch.zeros_like(diagonal)
    unresolved[3, 0] = 1.0
    unresolved[3, 1] = 1.0
    return replace(
        graph,
        edges=edges,
        diagonal=diagonal,
        unresolved=unresolved,
    ).check().canonicalize()


def test_ordered_lineage_separates_direct_relay_endogenous_and_unresolved_mass():
    output = ordered_routing_lineage(lineage_graph())
    expected = torch.eye(4)[[DIRECT, INDIRECT, ENDOGENOUS, UNRESOLVED]]

    assert torch.equal(output.query_lineage, expected)
    assert torch.allclose(
        output.query_trace.sum(dim=-1),
        torch.ones_like(output.query_trace[..., 0]),
    )


def test_layer_order_changes_response_carrier_lineage():
    graph = lineage_graph()
    ordered = ordered_routing_lineage(graph, layer_order=(0, 1))
    reverse = ordered_routing_lineage(graph, layer_order=[1, 0])

    assert ordered.query_lineage[1, INDIRECT] == 1.0
    assert reverse.query_lineage[1, ENDOGENOUS] == 1.0
    assert not torch.equal(ordered.query_lineage, reverse.query_lineage)


def test_predecessor_query_alignment_omits_unobserved_boundary_predictions():
    graph = lineage_graph()
    output = ordered_routing_lineage(graph)

    assert torch.equal(output.predictor_response_index, torch.tensor([0, 1, 2]))
    assert torch.equal(output.token_response_index, torch.tensor([1, 2, 3]))
    assert torch.equal(output.token_id, graph.response_token_ids[1:])
    assert torch.equal(output.token_lineage, output.query_lineage[:-1])
    assert graph.response_token_ids[0].item() not in output.token_id.tolist()


def test_heads_are_uniformly_averaged_only_at_the_layer_transition():
    graph = make_graph(
        layers=1,
        heads=2,
        response_start=1,
        response_count=1,
    )
    edges = TokenEdges(
        source=torch.tensor([0]),
        target=torch.tensor([1]),
        layer=torch.tensor([0]),
        head=torch.tensor([0]),
        weight=torch.tensor([1.0]),
    )
    diagonal = torch.zeros((1, 1, 2))
    unresolved = torch.zeros_like(diagonal)
    unresolved[0, 0, 1] = 1.0
    graph = replace(
        graph,
        edges=edges,
        diagonal=diagonal,
        unresolved=unresolved,
    ).check().canonicalize()

    output = ordered_routing_lineage(graph)

    assert torch.equal(
        output.query_lineage[0],
        torch.tensor([0.5, 0.0, 0.0, 0.5]),
    )
    assert output.token_lineage.shape == (0, 4)


def test_layer_order_cannot_repeat_a_layer():
    with pytest.raises(ValueError, match="permutation"):
        ordered_routing_lineage(lineage_graph(), layer_order=(0, 0))
