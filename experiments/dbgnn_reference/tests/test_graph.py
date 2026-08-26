import torch

from experiments.dbgnn_reference.graph import build_dbgnn_graph
from experiments.grounded_route.artifacts import EncodedTokenGraph


def encoded_graph() -> EncodedTokenGraph:
    source = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 3, 3])
    target = torch.tensor([2, 2, 2, 2, 3, 3, 3, 4, 5, 5])
    return EncodedTokenGraph(
        sample_id="sample",
        source_id="source",
        task_type="QA",
        response_start=2,
        layer_count=3,
        head_count=2,
        attention_floor=0.01,
        token_ids=torch.arange(6),
        node_embedding=torch.zeros(6, 3),
        edge_index=torch.stack((source, target)),
        edge_layer=torch.tensor([0, 0, 1, 0, 1, 1, 2, 2, 2, 2]),
        edge_head=torch.tensor([0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
        edge_weight=torch.tensor(
            [0.4, 0.2, 0.2, 0.8, 0.5, 0.5, 0.2, 0.6, 0.6, 0.2]
        ),
        diagonal=torch.zeros(4, 3, 2),
        unresolved=torch.zeros(4, 3, 2),
        lineage=torch.zeros(4, 3, 2, 3),
    )


def test_typed_edges_are_aggregated_into_fixed_first_order_input():
    graph = build_dbgnn_graph(encoded_graph())

    assert torch.equal(
        graph.edge_index_fo,
        torch.tensor(
            [[0, 1, 2, 2, 3], [2, 2, 3, 4, 5]],
        ),
    )
    assert torch.allclose(
        graph.edge_weight_fo,
        torch.tensor([2 / 15, 2 / 15, 0.20, 0.10, 2 / 15]),
    )
    assert graph.x_fo.shape == (6, 4)
    assert torch.equal(
        graph.x_fo[:, :2].argmax(dim=-1),
        torch.tensor([0, 0, 1, 1, 1, 1]),
    )
    assert torch.allclose(
        graph.x_fo[:, 2],
        torch.tensor([0.0, 0.5, 2 / 3, 0.75, 0.8, 5 / 6]),
    )
    assert torch.allclose(
        graph.x_fo[:, 3],
        torch.tensor([0.0, 0.0, 0.5, 2 / 3, 0.75, 0.8]),
    )


def test_order_two_nodes_paths_and_terminal_projection_are_exact():
    graph = build_dbgnn_graph(encoded_graph())

    assert graph.x_ho.shape == (5, 10)
    assert torch.equal(graph.ho_endpoints, graph.edge_index_fo)
    assert torch.equal(
        graph.edge_index,
        torch.tensor([[0, 0, 1, 2], [2, 3, 2, 4]]),
    )
    assert torch.allclose(
        graph.edge_weight,
        torch.tensor([0.16, 0.03, 0.20, 0.20]),
    )
    assert torch.equal(
        graph.edge_index_hon_to_fon,
        torch.tensor([[0, 1, 2, 3, 4], [2, 2, 3, 4, 5]]),
    )
    assert graph.num_nodes == 6
    assert graph.num_ho_nodes == 5
    assert graph.edge_weight_ho is graph.edge_weight


def test_larger_layer_wait_adds_only_chronological_compositions():
    graph = build_dbgnn_graph(encoded_graph(), delta_layers=2)

    assert torch.equal(
        graph.edge_index,
        torch.tensor([[0, 0, 1, 1, 2], [2, 3, 2, 3, 4]]),
    )
    assert torch.allclose(
        graph.edge_weight,
        torch.tensor([0.19, 0.12, 0.24, 0.12, 0.20]),
    )
    source_pair = graph.ho_endpoints[:, graph.edge_index[0]]
    target_pair = graph.ho_endpoints[:, graph.edge_index[1]]
    assert torch.equal(source_pair[1], target_pair[0])
    assert bool((target_pair[1] > source_pair[1]).all())


def test_graph_contract_moves_all_tensors():
    graph = build_dbgnn_graph(encoded_graph()).to("cpu")

    tensor_fields = (
        graph.x_ho,
        graph.x_fo,
        graph.edge_index,
        graph.edge_weight,
        graph.edge_index_fo,
        graph.edge_weight_fo,
        graph.edge_index_hon_to_fon,
        graph.ho_endpoints,
    )
    assert all(value.device.type == "cpu" for value in tensor_fields)


def test_no_transition_keeps_inputs_and_terminal_projection():
    causal = build_dbgnn_graph(encoded_graph())
    control = build_dbgnn_graph(
        encoded_graph(),
        higher_order_mode="no_transition",
    )

    for name in (
        "x_ho",
        "x_fo",
        "edge_index_fo",
        "edge_weight_fo",
        "edge_index_hon_to_fon",
        "ho_endpoints",
    ):
        assert torch.equal(getattr(causal, name), getattr(control, name))
    assert control.edge_index.shape == (2, 0)
    assert control.edge_weight.shape == (0,)
