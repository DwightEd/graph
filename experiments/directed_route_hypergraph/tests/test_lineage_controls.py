from collections import Counter
from dataclasses import replace

import pytest
import torch

from experiments.directed_route_hypergraph.lineage_controls import (
    LINEAGE_CONTROLS,
    apply_lineage_control,
    layer_order,
    response_carrier_edges,
)
from experiments.directed_route_hypergraph.routing_lineage import (
    initial_routing_lineage,
    ordered_routing_lineage,
    routing_lineage_step,
)
from experiments.grounded_route.controls import lag_bucket
from experiments.grounded_route.tests.helpers import make_rewirable_graph


def row_mass(graph):
    mass = torch.zeros_like(graph.diagonal)
    mass.index_put_(
        (
            graph.edges.target - graph.response_start,
            graph.edges.layer,
            graph.edges.head,
        ),
        graph.edges.weight,
        accumulate=True,
    )
    return mass


def typed_edges(graph, selected):
    edges = graph.edges
    return {
        (
            int(edges.source[index]),
            int(edges.target[index]),
            int(edges.layer[index]),
            int(edges.head[index]),
            float(edges.weight[index]),
        )
        for index in torch.nonzero(selected, as_tuple=False).flatten().tolist()
    }


def test_layer_controls_return_exact_seeded_orders():
    assert LINEAGE_CONTROLS == (
        "ordered",
        "reverse",
        "random_layer",
        "last_layer",
        "carrier_rewire",
    )
    assert layer_order("ordered", 6, 19) == (0, 1, 2, 3, 4, 5)
    assert layer_order("carrier_rewire", 6, 19) == (0, 1, 2, 3, 4, 5)
    assert layer_order("reverse", 6, 19) == (5, 4, 3, 2, 1, 0)
    assert layer_order("last_layer", 6, 19) == (5,)

    first = layer_order("random_layer", 6, 19)
    second = layer_order("random_layer", 6, 19)
    assert first == second
    assert sorted(first) == list(range(6))
    assert first != tuple(range(6))


def test_random_layer_identity_draw_is_changed_when_possible():
    identity_seed = next(
        seed
        for seed in range(100)
        if torch.equal(
            torch.randperm(2, generator=torch.Generator().manual_seed(seed)),
            torch.arange(2),
        )
    )

    assert layer_order("random_layer", 2, identity_seed) == (1, 0)
    assert layer_order("random_layer", 1, identity_seed) == (0,)


def test_last_layer_starts_from_roots_and_applies_only_the_final_layer():
    # Repeat the synthetic rows across three layers so preceding transitions
    # would change the result if the control accidentally composed them.
    graph = make_rewirable_graph()
    edges = graph.edges
    layer_count = 3
    repeated_edges = type(edges)(
        source=edges.source.repeat(layer_count),
        target=edges.target.repeat(layer_count),
        layer=torch.arange(layer_count).repeat_interleave(edges.count),
        head=edges.head.repeat(layer_count),
        weight=edges.weight.repeat(layer_count),
    )
    diagonal = graph.diagonal.repeat(1, layer_count, 1)
    unresolved = graph.unresolved.repeat(1, layer_count, 1)
    graph = replace(
        graph,
        layer_count=layer_count,
        edges=repeated_edges,
        diagonal=diagonal,
        unresolved=unresolved,
    ).check().canonicalize()

    controlled, order = apply_lineage_control(
        graph,
        "last_layer",
        seed=29,
    )
    output = ordered_routing_lineage(controlled, layer_order=order)
    full = ordered_routing_lineage(graph)
    expected = routing_lineage_step(
        graph,
        initial_routing_lineage(graph),
        graph.layer_count - 1,
    )

    assert order == (graph.layer_count - 1,)
    assert output.query_trace.shape[1] == 1
    assert torch.equal(output.query_lineage, expected)
    assert not torch.equal(output.query_lineage, full.query_lineage)


def test_carrier_rewire_preserves_direct_prompt_routes_and_row_nuisances():
    graph = make_rewirable_graph().canonicalize()
    controlled, order = apply_lineage_control(
        graph,
        "carrier_rewire",
        seed=29,
        carrier_rewire_passes=1,
    )

    original_carrier = response_carrier_edges(graph)
    changed_carrier = response_carrier_edges(controlled)
    assert order == (0,)
    assert typed_edges(graph, ~original_carrier) == typed_edges(
        controlled, ~changed_carrier
    )
    assert typed_edges(graph, original_carrier) != typed_edges(
        controlled, changed_carrier
    )

    original_edges = graph.edges
    changed_edges = controlled.edges
    original_response_source = original_edges.source[original_carrier]
    changed_response_source = changed_edges.source[changed_carrier]
    assert Counter(original_response_source.tolist()) == Counter(
        changed_response_source.tolist()
    )
    assert Counter(original_edges.target[original_carrier].tolist()) == Counter(
        changed_edges.target[changed_carrier].tolist()
    )
    assert torch.equal(
        torch.sort(
            lag_bucket(
                original_response_source,
                original_edges.target[original_carrier],
            )
        ).values,
        torch.sort(
            lag_bucket(
                changed_response_source,
                changed_edges.target[changed_carrier],
            )
        ).values,
    )
    assert bool(
        (changed_response_source < changed_edges.target[changed_carrier]).all()
    )
    assert torch.allclose(row_mass(controlled), row_mass(graph))
    assert torch.equal(controlled.diagonal, graph.diagonal)
    assert torch.equal(controlled.unresolved, graph.unresolved)


def test_controls_reject_unknown_names():
    with pytest.raises(ValueError, match="unknown lineage control"):
        layer_order("shuffle_everything", 3, 0)
