from dataclasses import dataclass, replace

import torch

from experiments.information_flow.basis import source_basis
from experiments.information_flow.transport import (
    flow_embedding,
    sketch_tables,
    transport_layer,
)


@dataclass(frozen=True)
class Graph:
    response_start: int
    layer_count: int
    head_count: int
    token_ids: torch.Tensor
    node_embedding: torch.Tensor
    edge_index: torch.Tensor
    edge_layer: torch.Tensor
    edge_head: torch.Tensor
    edge_weight: torch.Tensor
    diagonal: torch.Tensor
    unresolved: torch.Tensor


def one_edge_graph():
    return Graph(
        response_start=1,
        layer_count=1,
        head_count=1,
        token_ids=torch.tensor([10, 11]),
        node_embedding=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_layer=torch.tensor([0]),
        edge_head=torch.tensor([0]),
        edge_weight=torch.tensor([0.75]),
        diagonal=torch.tensor([[[0.25]]]),
        unresolved=torch.zeros(1, 1, 1),
    )


def test_mean_transport_matches_row_weighted_state():
    output = flow_embedding(one_edge_graph(), mode="mean", checkpoints=1)
    expected = torch.tensor([[0.0, 1.0, 0.75, 0.25]])
    assert torch.allclose(output.embedding, expected)
    assert output.trajectory.shape == (1, 1, 2)


def test_sketch_keeps_head_identity():
    graph = Graph(
        response_start=1,
        layer_count=1,
        head_count=2,
        token_ids=torch.tensor([10, 11]),
        node_embedding=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=torch.tensor([[0, 0], [1, 1]]),
        edge_layer=torch.tensor([0, 0]),
        edge_head=torch.tensor([0, 1]),
        edge_weight=torch.tensor([1.0, 0.0]),
        diagonal=torch.zeros(1, 1, 2),
        unresolved=torch.tensor([[[0.0, 1.0]]]),
    )
    swapped = replace(graph, edge_head=torch.tensor([1, 0]))
    state = source_basis(2, 1, 2)
    permutation = torch.tensor([[0, 1], [0, 1]])
    sign = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
    first = transport_layer(graph, state, 0, permutation, sign, "sketch")
    second = transport_layer(swapped, state, 0, permutation, sign, "sketch")
    assert not torch.allclose(first, second)


def test_self_only_rows_are_identity_and_prompt_never_drifts():
    graph = replace(
        one_edge_graph(),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_layer=torch.empty(0, dtype=torch.long),
        edge_head=torch.empty(0, dtype=torch.long),
        edge_weight=torch.empty(0),
        diagonal=torch.ones(1, 1, 1),
    )
    state = source_basis(2, 1, 2)
    permutation, sign = sketch_tables(1, 1, 2, 19, state.device)
    updated = transport_layer(
        graph,
        state,
        0,
        permutation[0],
        sign[0],
        "sketch",
    )

    assert torch.equal(updated, state)
    output = flow_embedding(graph, mode="sketch", checkpoints=1, seed=19)
    assert torch.equal(output.trajectory[:, 0], state[1:])


def test_future_target_does_not_change_earlier_response_node():
    graph = Graph(
        response_start=1,
        layer_count=2,
        head_count=1,
        token_ids=torch.tensor([10, 11, 12]),
        node_embedding=torch.eye(3),
        edge_index=torch.tensor([[0, 0, 1, 0], [1, 2, 2, 1]]),
        edge_layer=torch.tensor([0, 0, 0, 1]),
        edge_head=torch.zeros(4, dtype=torch.long),
        edge_weight=torch.tensor([1.0, 0.5, 0.5, 1.0]),
        diagonal=torch.zeros(2, 2, 1),
        unresolved=torch.zeros(2, 2, 1),
    )
    changed_weight = graph.edge_weight.clone()
    changed_weight[1:3] = torch.tensor([0.9, 0.1])
    changed = replace(graph, edge_weight=changed_weight)
    first = flow_embedding(graph, mode="mean", checkpoints=2)
    second = flow_embedding(changed, mode="mean", checkpoints=2)
    assert torch.allclose(first.embedding[0], second.embedding[0])
    assert not torch.allclose(first.embedding[1], second.embedding[1])
