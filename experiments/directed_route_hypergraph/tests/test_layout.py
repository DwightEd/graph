from dataclasses import replace

import torch

from experiments.directed_route_hypergraph.config import ModelConfig
from experiments.directed_route_hypergraph.flow import ordered_flow
from experiments.directed_route_hypergraph.layout import ordered_endpoint_layout
from experiments.directed_route_hypergraph.model import DirectedRouteHypergraphEncoder
from experiments.grounded_route.graph import TokenEdges, build_graph
from experiments.grounded_route.tests.helpers import SparseSample, make_graph


def noncommuting_layers_graph():
    graph = make_graph(layers=2, heads=1, response_start=1, response_count=2)
    edges = TokenEdges(
        source=torch.tensor([0, 1]),
        target=torch.tensor([1, 2]),
        layer=torch.tensor([0, 1]),
        head=torch.zeros(2, dtype=torch.long),
        weight=torch.ones(2),
    )
    diagonal = torch.ones((2, 2, 1))
    diagonal[0, 0, 0] = 0.0
    diagonal[1, 1, 0] = 0.0
    return replace(
        graph,
        edges=edges,
        diagonal=diagonal,
        unresolved=torch.zeros_like(diagonal),
    ).check().canonicalize()


def test_ordered_endpoint_layout_conserves_mass_and_causality():
    graph = make_graph(layers=3, heads=4, response_start=3, response_count=7)
    layout = ordered_endpoint_layout(graph).distribution

    assert layout.shape == (graph.response_count, graph.token_count + 1)
    assert torch.isfinite(layout).all()
    assert bool((layout >= 0).all())
    assert torch.allclose(
        layout.sum(dim=1),
        torch.ones(graph.response_count),
        atol=1e-6,
        rtol=1e-6,
    )
    for response in range(graph.response_count):
        target = graph.response_start + response
        assert torch.equal(
            layout[response, target + 1 : graph.token_count],
            torch.zeros_like(layout[response, target + 1 : graph.token_count]),
        )


def test_sparse_layout_matches_dense_layer_matrix_product():
    graph = make_graph(layers=3, heads=2, response_start=2, response_count=4)
    residual_weight = 0.7
    sparse = ordered_endpoint_layout(
        graph,
        residual_weight=residual_weight,
    ).distribution
    endpoint_count = graph.token_count + 1
    unresolved = graph.token_count
    dense = torch.eye(endpoint_count)
    for layer in range(graph.layer_count):
        transition = torch.zeros((endpoint_count, endpoint_count))
        transition[: graph.response_start, : graph.response_start] = torch.eye(
            graph.response_start
        )
        transition[unresolved, unresolved] = 1.0
        edges = graph.canonicalize().layer_edges(layer)
        transition.index_put_(
            (edges.target, edges.source),
            edges.weight / graph.head_count / (residual_weight + 1.0),
            accumulate=True,
        )
        response = torch.arange(graph.response_start, graph.token_count)
        diagonal = graph.diagonal[:, layer].mean(1)
        transition[response, response] += (
            residual_weight + diagonal
        ) / (residual_weight + 1.0)
        transition[response, unresolved] = (
            graph.unresolved[:, layer].mean(1) / (residual_weight + 1.0)
        )
        dense = transition @ dense

    assert torch.allclose(
        sparse,
        dense[graph.response_start : graph.token_count],
        atol=1e-6,
        rtol=1e-6,
    )


def test_endpoint_layout_coarse_grains_exactly_to_prompt_response_unresolved_flow():
    graph = make_graph(layers=3, heads=2, response_start=3, response_count=5)
    for order in (None, tuple(reversed(range(graph.layer_count)))):
        layout = ordered_endpoint_layout(graph, layer_order=order).distribution
        grouped = torch.stack(
            (
                layout[:, : graph.response_start].sum(dim=1),
                layout[:, graph.response_start : graph.token_count].sum(dim=1),
                layout[:, -1],
            ),
            dim=1,
        )
        flow = ordered_flow(graph, layer_order=order).token_trace[:, -1]

        assert torch.allclose(grouped, flow, atol=1e-6, rtol=1e-6)


def test_one_layer_sink_and_residual_have_the_analytic_mass():
    graph = make_graph(layers=1, heads=2, response_start=1, response_count=1)
    edges = TokenEdges(
        source=torch.tensor([0, 0]),
        target=torch.tensor([1, 1]),
        layer=torch.zeros(2, dtype=torch.long),
        head=torch.tensor([0, 1]),
        weight=torch.tensor([0.2, 0.4]),
    )
    diagonal = torch.tensor([[[0.3, 0.1]]])
    unresolved = torch.tensor([[[0.5, 0.5]]])
    graph = replace(
        graph,
        edges=edges,
        diagonal=diagonal,
        unresolved=unresolved,
    ).check().canonicalize()

    layout = ordered_endpoint_layout(graph, residual_weight=1.0).distribution

    assert torch.allclose(layout[0], torch.tensor([0.15, 0.60, 0.25]))


def test_endpoint_layout_is_sensitive_to_actual_layer_order():
    graph = noncommuting_layers_graph()
    forward = ordered_endpoint_layout(
        graph,
        residual_weight=0.0,
        layer_order=(0, 1),
    ).distribution
    reverse = ordered_endpoint_layout(
        graph,
        residual_weight=0.0,
        layer_order=(1, 0),
    ).distribution

    assert torch.equal(forward[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(reverse[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert not torch.equal(forward, reverse)


def test_endpoint_layout_preserves_exact_prompt_endpoint_identity():
    graph = make_graph(layers=1, heads=1, response_start=3, response_count=1)
    changed_edges = TokenEdges(
        source=torch.tensor([1]),
        target=graph.edges.target,
        layer=graph.edges.layer,
        head=graph.edges.head,
        weight=graph.edges.weight,
    )
    changed = replace(graph, edges=changed_edges).check().canonicalize()
    original = ordered_endpoint_layout(graph, residual_weight=0.0).distribution
    rewired = ordered_endpoint_layout(changed, residual_weight=0.0).distribution

    assert torch.isclose(original[0, graph.edges.source[0]], graph.edges.weight[0])
    assert torch.isclose(rewired[0, 1], graph.edges.weight[0])
    assert not torch.equal(original, rewired)


def test_endpoint_layout_preserves_exact_response_relay_identity():
    graph = make_graph(layers=1, heads=1, response_start=2, response_count=3)
    history = graph.edges.source >= graph.response_start
    edge = int(torch.nonzero(history, as_tuple=False)[0].item())
    source = graph.edges.source.clone()
    source[edge] = source[edge] + 1
    changed = replace(
        graph,
        edges=TokenEdges(
            source=source,
            target=graph.edges.target,
            layer=graph.edges.layer,
            head=graph.edges.head,
            weight=graph.edges.weight,
        ),
    ).check().canonicalize()
    original = ordered_endpoint_layout(graph, residual_weight=0.0).distribution
    rewired = ordered_endpoint_layout(changed, residual_weight=0.0).distribution
    target = graph.edges.target[edge] - graph.response_start

    assert torch.isclose(
        original[target, graph.edges.source[edge]],
        graph.edges.weight[edge],
    )
    assert torch.isclose(rewired[target, source[edge]], graph.edges.weight[edge])
    assert not torch.equal(original, rewired)


def test_future_response_rows_do_not_change_prefix_layout():
    graph = make_graph(layers=3, heads=2, response_start=3, response_count=7)
    full = ordered_endpoint_layout(graph).distribution

    for count in (1, 3, graph.response_count - 1):
        prefix_graph = graph.truncate_response(count)
        prefix = ordered_endpoint_layout(prefix_graph).distribution
        assert torch.allclose(
            full[:count, : prefix_graph.token_count],
            prefix[:, : prefix_graph.token_count],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            full[:count, -1],
            prefix[:, -1],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.equal(
            full[:count, prefix_graph.token_count : graph.token_count],
            torch.zeros_like(
                full[:count, prefix_graph.token_count : graph.token_count]
            ),
        )


def test_layout_decoder_masks_future_endpoints_but_keeps_sink():
    torch.manual_seed(41)
    graph = make_graph(response_count=5)
    model = DirectedRouteHypergraphEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(dropout=0.0),
    ).eval()
    output = model(graph)
    selected = torch.tensor([0, 2, 4])
    logits = model.endpoint_layout_logits(output, graph, selected)

    assert logits.shape == (len(selected), graph.token_count + 1)
    assert torch.isfinite(logits).all()
    minimum = torch.finfo(logits.dtype).min
    for row, response in enumerate(selected.tolist()):
        target = graph.response_start + response
        assert bool((logits[row, target + 1 : graph.token_count] == minimum).all())
        assert bool((logits[row, : target + 1] != minimum).all())
        assert logits[row, -1] != minimum


def test_layout_decoder_logits_are_prefix_invariant():
    torch.manual_seed(43)
    graph = make_graph(layers=3, heads=2, response_count=6)
    model = DirectedRouteHypergraphEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(dropout=0.0),
    ).eval()
    full_output = model(graph)

    for count in (1, 3, graph.response_count - 1):
        prefix_graph = graph.truncate_response(count)
        prefix_output = model(prefix_graph)
        selected = torch.arange(count)
        full = model.endpoint_layout_logits(full_output, graph, selected)
        prefix = model.endpoint_layout_logits(
            prefix_output,
            prefix_graph,
            selected,
        )

        assert torch.allclose(
            full[:, : prefix_graph.token_count],
            prefix[:, :-1],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(full[:, -1], prefix[:, -1], atol=1e-6, rtol=1e-6)
        assert bool(
            (
                full[:, prefix_graph.token_count : graph.token_count]
                == torch.finfo(full.dtype).min
            ).all()
        )


def test_layout_supports_full_32_by_32_channel_geometry():
    graph = build_graph(SparseSample(layers=32, heads=32))
    layout = ordered_endpoint_layout(graph).distribution

    assert layout.shape == (graph.response_count, graph.token_count + 1)
    assert torch.allclose(
        layout.sum(dim=1),
        torch.ones(graph.response_count),
        atol=1e-6,
        rtol=1e-6,
    )
