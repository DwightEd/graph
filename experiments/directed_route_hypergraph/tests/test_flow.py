from dataclasses import replace

import torch

from experiments.directed_route_hypergraph.flow import (
    flow_step,
    initial_flow,
    ordered_flow,
)
from experiments.grounded_route.graph import TokenEdges
from experiments.grounded_route.tests.helpers import make_graph


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


def test_ordered_flow_conserves_probability_mass():
    output = ordered_flow(make_graph(), residual_weight=1.0)

    assert bool((output.head_trace >= 0).all())
    assert bool((output.token_trace >= 0).all())
    assert torch.allclose(
        output.head_trace.sum(dim=-1),
        torch.ones_like(output.head_trace[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        output.token_trace.sum(dim=-1),
        torch.ones_like(output.token_trace[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_flow_step_keeps_prompt_fixed_and_returns_aligned_provenance():
    graph = make_graph(layers=2, heads=3, response_start=3, response_count=4)
    state = initial_flow(graph)
    edges = graph.canonicalize().layer_edges(0, state.device)
    step = flow_step(graph, state, 0, residual_weight=1.0, edges=edges)

    assert step.provenance.shape == (edges.count, 3)
    assert step.head_flow.shape == (graph.response_count, graph.head_count, 3)
    assert step.token_state.shape == (graph.token_count, 3)
    assert torch.equal(step.provenance, state[edges.source])
    assert torch.equal(
        step.token_state[: graph.response_start],
        state[: graph.response_start],
    )


def test_ordered_flow_is_sensitive_to_layer_order():
    graph = noncommuting_layers_graph()
    forward = ordered_flow(
        graph,
        residual_weight=0.0,
        layer_order=(0, 1),
    )
    reverse = ordered_flow(
        graph,
        residual_weight=0.0,
        layer_order=(1, 0),
    )

    assert torch.equal(forward.token_trace[1, -1], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(reverse.token_trace[1, -1], torch.tensor([0.0, 1.0, 0.0]))
    assert not torch.equal(forward.token_trace, reverse.token_trace)


def test_response_source_can_relay_prompt_rooted_provenance():
    graph = noncommuting_layers_graph()
    state = initial_flow(graph)
    first = flow_step(graph, state, 0, residual_weight=0.0)
    second = flow_step(graph, first.token_state, 1, residual_weight=0.0)

    assert torch.equal(first.token_state[1], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(second.provenance[0], torch.tensor([1.0, 0.0, 0.0]))


def test_future_response_rows_do_not_change_prefix_flow():
    graph = make_graph(layers=3, heads=2, response_start=3, response_count=7)
    full = ordered_flow(graph)

    for count in (1, 3, graph.response_count - 1):
        prefix = ordered_flow(graph.truncate_response(count))
        assert torch.allclose(
            full.head_trace[:count],
            prefix.head_trace,
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            full.token_trace[:count],
            prefix.token_trace,
            atol=1e-6,
            rtol=1e-6,
        )
