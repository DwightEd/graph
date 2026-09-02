import numpy as np
import pytest

from experiments.attention_mechanism_audit.graph import (
    REGISTER_NAMES,
    STAGE_NAMES,
    build_graph,
    sparse_contribution_sum,
)


def _graph_artifact() -> dict:
    shape = (2, 3, 2, 2)
    source = np.full(shape, -1, dtype=np.int32)
    head = np.full(shape, -1, dtype=np.int16)
    magnitude = np.zeros(shape, dtype=np.float64)
    contribution = np.zeros(shape, dtype=np.float64)
    remainder_magnitude = np.ones(shape[:-1], dtype=np.float64)
    remainder_contribution = np.ones(shape[:-1], dtype=np.float64)
    cover_size = np.zeros(shape[:-1], dtype=np.int16)

    edges = [
        (0, 0, 0, 0, 0, 0),  # evidence
        (0, 0, 0, 1, 1, 1),  # other prompt, distinct head
        (0, 0, 1, 0, 2, 2),  # predictor self
        (0, 2, 0, 0, 3, 1),  # response history
        (0, 2, 1, 0, 4, 0),  # predictor self
        (1, 2, 0, 0, 0, 1),  # same endpoint, distinct layer/head
        (1, 2, 1, 0, 0, 0),  # adaptive cover keeps two explicit slots
        (1, 2, 1, 1, 3, 1),
    ]
    for layer, token, register, slot, endpoint, edge_head in edges:
        source[layer, token, register, slot] = endpoint
        head[layer, token, register, slot] = edge_head
        magnitude[layer, token, register, slot] = 0.2
        contribution[layer, token, register, slot] = 0.1
        remainder_magnitude[layer, token, register] -= 0.2
        remainder_contribution[layer, token, register] -= 0.1
        cover_size[layer, token, register] += 1
    contribution[0, 0, 0, 0] = -0.1
    remainder_contribution[0, 0, 0] += 0.2
    cover_size[1, 2, 1] = 3  # two saved endpoints plus a real endpoint-free tail
    role_mass = np.zeros((2, 3, 3, 2, 4))
    role_contribution = np.zeros_like(role_mass)
    role_mass[:, :, 0, :, 0] = magnitude.sum(-1) + remainder_magnitude
    role_contribution[:, :, 0, :, 0] = contribution.sum(-1) + remainder_contribution

    node_shape = (2, 3, 2, 4)
    return {
        "response_start": 3,
        "evidence_mask": np.array([True, False, False]),
        "trace": {
            "register_route_source_index": source,
            "register_route_head_index": head,
            "register_route_magnitude": magnitude,
            "register_route_contribution": contribution,
            "register_route_root_contribution": contribution.copy(),
            "register_route_carrier_contribution": np.zeros_like(contribution),
            "register_route_gate_contribution": np.zeros_like(contribution),
            "register_route_remainder_magnitude": remainder_magnitude,
            "register_route_remainder_contribution": remainder_contribution,
            "register_route_remainder_root_contribution": remainder_contribution.copy(),
            "register_route_remainder_carrier_contribution": np.zeros_like(
                remainder_contribution
            ),
            "register_route_remainder_gate_contribution": np.zeros_like(
                remainder_contribution
            ),
            "register_route_cover_size": cover_size,
            "register_role_mass": role_mass,
            "register_role_contribution": role_contribution,
            "register_role_root_contribution": role_contribution.copy(),
            "register_role_carrier_contribution": np.zeros_like(role_contribution),
            "register_role_gate_contribution": np.zeros_like(role_contribution),
            "register_norm": np.arange(np.prod(node_shape), dtype=float).reshape(
                node_shape
            ),
            "register_conservation_error": np.zeros((2, 3, 2)),
            "register_attention_edge_error": np.zeros((2, 3, 2)),
        },
    }


def test_graph_keeps_true_endpoint_head_register_role_and_tail():
    graph = build_graph(_graph_artifact())
    assert len(graph.source) == 8
    assert set(graph.register) == set(REGISTER_NAMES)
    assert set(graph.role) == {
        "evidence",
        "other_prompt",
        "response_history",
        "predictor_self",
    }
    assert np.all(graph.source <= graph.target)
    assert (graph.contribution < 0).any()
    assert set(graph.head) == {0, 1, 2}
    assert (graph.layer == 1).any()

    history = graph.role == "response_history"
    np.testing.assert_array_equal(graph.source[history], [3, 3])
    np.testing.assert_array_equal(graph.target[history], [4, 4])
    assert len(graph.row_target) == 2 * 3 * 2
    assert graph.cover_size.reshape(2, 3, 2)[1, 2, 1] == 3
    assert graph.remainder_magnitude.reshape(2, 3, 2)[1, 2, 1] > 0
    assert not hasattr(graph, "row_source")
    assert not hasattr(graph, "row_head")
    assert graph.register_role_mass.shape == (2, 3, 3, 2, 4)
    assert graph.register_role_contribution.shape == (2, 3, 3, 2, 4)
    np.testing.assert_allclose(
        graph.contribution,
        graph.root_contribution + graph.carrier_contribution + graph.gate_contribution,
    )


def test_register_nodes_and_vertical_edges_form_an_ordered_dag():
    graph = build_graph(_graph_artifact())
    assert len(graph.node_id) == 2 * 3 * 2 * 4
    assert set(graph.node_stage) == set(STAGE_NAMES)
    assert set(graph.node_register) == set(REGISTER_NAMES)
    within = 2 * 3 * 2 * 3
    assert len(graph.vertical_source) == within + 1 * 3 * 2
    assert np.all(graph.vertical_source < graph.vertical_target)
    assert set(graph.node_stage[graph.vertical_source[:within]]) == set(STAGE_NAMES[:3])
    assert set(graph.node_stage[graph.vertical_target[:within]]) == {STAGE_NAMES[-1]}
    np.testing.assert_array_equal(
        graph.node_norm,
        _graph_artifact()["trace"]["register_norm"].ravel(),
    )
    assert graph.register_conservation_error.shape == (2, 3, 2)
    assert graph.register_attention_edge_error.shape == (2, 3, 2)
    assert set(graph.node_stage[graph.route_target_node]) == {"attention_write"}
    internal = graph.route_source_node >= 0
    assert internal.any()
    assert (graph.route_source_node[~internal] == -1).all()
    assert set(graph.node_stage[graph.route_source_node[internal]]) == {"input_state"}
    np.testing.assert_array_equal(
        graph.node_target[graph.route_source_node[internal]], graph.source[internal]
    )


def test_sparse_explicit_edges_and_endpoint_free_tail_conserve_contribution():
    total = sparse_contribution_sum(_graph_artifact())
    assert total.shape == (2, 3, 2)
    np.testing.assert_allclose(total, 1.0)


def test_graph_rejects_future_edges_and_malformed_register_fields():
    artifact = _graph_artifact()
    artifact["trace"]["register_route_source_index"][0, 0, 0, 0] = 3
    with pytest.raises(ValueError, match="future"):
        build_graph(artifact)

    artifact = _graph_artifact()
    artifact["trace"]["register_norm"] = np.zeros((2, 3, 2, 3))
    with pytest.raises(ValueError, match="register norms"):
        build_graph(artifact)

    artifact = _graph_artifact()
    artifact["trace"]["register_route_remainder_magnitude"][0, 0, 0] = -1
    with pytest.raises(ValueError, match="remainder magnitudes"):
        build_graph(artifact)

    artifact = _graph_artifact()
    artifact["trace"]["register_role_contribution"][0, 0, 0, 0, 0] += 0.5
    artifact["trace"]["register_role_root_contribution"][0, 0, 0, 0, 0] += 0.5
    with pytest.raises(ValueError, match="sparse contribution"):
        build_graph(artifact)
