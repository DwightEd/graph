import numpy as np
import pytest

from experiments.attention_mechanism_audit.graph import (
    build_graph,
    route_contraction,
    route_mass_contraction,
)


def _graph_artifact() -> dict:
    source = np.full((1, 3, 2, 2, 1), -1, dtype=np.int32)
    source[0, :, :, 0, 0] = [[0, 1], [0, 1], [0, 1]]
    source[0, 2, :, 1, 0] = 3
    magnitude = np.zeros_like(source, dtype=np.float64)
    magnitude[0, :, :, 0, 0] = [[0.8, 0.8], [0.8, 0.6], [0.8, 0.8]]
    magnitude[0, 2, :, 1, 0] = 0.7
    remainder = np.zeros((1, 3, 2, 2))
    remainder[0, :, :, 0] = [[0.2, 0.2], [0.2, 0.4], [0.2, 0.2]]
    remainder[0, 2, :, 1] = 0.3
    cover_size = np.zeros((1, 3, 2, 2), dtype=np.int16)
    cover_size[0, :, :, 0] = 1
    cover_size[0, 1, 1, 0] = 2
    cover_size[0, 2, :, 1] = 1
    role_mass = np.zeros((1, 3, 2, 4))
    role_mass[..., 0] = 1
    role_mass[0, 2, :, 2] = 1
    return {
        "response_start": 3,
        "evidence_mask": np.array([True, True, False]),
        "trace": {
            "route_source_index": source,
            "route_source_magnitude": magnitude,
            "route_source_remainder": remainder,
            "route_source_cover_size": cover_size,
            "edge_role_mass": role_mass,
        },
    }


def test_graph_preserves_endpoints_heads_roles_and_omitted_tail():
    graph = build_graph(_graph_artifact())
    assert len(graph.source) == 8
    np.testing.assert_array_equal(graph.target, [2, 2, 3, 3, 4, 4, 4, 4])
    np.testing.assert_array_equal(graph.head, [0, 1, 0, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(
        graph.role,
        [
            "evidence",
            "evidence",
            "evidence",
            "evidence",
            "evidence",
            "response_history",
            "evidence",
            "response_history",
        ],
    )
    assert np.all(graph.source <= graph.target)
    assert graph.cover_size.reshape(3, 2, 2)[1, 1, 0] == 2
    np.testing.assert_array_equal(
        graph.row_role,
        ["evidence", "response_history"] * 6,
    )
    np.testing.assert_allclose(graph.magnitude.sum() + graph.remainder.sum(), 8.0)


def test_graph_rejects_future_role_mismatch_and_nonconserved_tail():
    artifact = _graph_artifact()
    artifact["trace"]["route_source_index"][0, 0, 0, 0, 0] = 3
    with pytest.raises(ValueError, match="non-causal"):
        build_graph(artifact)

    artifact = _graph_artifact()
    artifact["trace"]["route_source_index"][0, 0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="non-evidence"):
        build_graph(artifact)

    artifact = _graph_artifact()
    artifact["trace"]["route_source_remainder"][0, 0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="conserve"):
        build_graph(artifact)


def test_route_contraction_uses_joint_routes_then_mass_weights_layers():
    support = np.ones((2, 3, 4))
    support[0, :, 0] = [8, 8, 2]
    support[1, :, 0] = [2, 2, 8]
    mass = np.zeros((2, 3, 2, 4))
    mass[0, :, :, 0] = [1.0, 0.5]
    mass[1, :, :, 0] = [0.25, 0.25]
    artifact = {
        "trace": {
            "edge_role_effective_routes": support,
            "edge_role_mass": mass,
        }
    }
    contraction = route_contraction(artifact, "evidence")
    np.testing.assert_allclose(contraction[:2], 0)
    np.testing.assert_allclose(contraction[2], 0.5 * np.log(4))


def test_support_and_mass_contraction_keep_disappearance_separate():
    support = np.zeros((1, 4, 4))
    support[0, 2:, 2] = [4, 4]
    mass = np.zeros((1, 4, 2, 4))
    mass[0, 2, :, 2] = 1
    artifact = {
        "trace": {
            "edge_role_effective_routes": support,
            "edge_role_mass": mass,
        }
    }
    np.testing.assert_allclose(
        route_contraction(artifact, "response_history"), [0, 0, 0, 0]
    )
    mass[0, 3, :, 2] = 0
    support[0, 3, 2] = 0
    assert route_contraction(artifact, "response_history")[3] == 0
    assert route_mass_contraction(artifact, "response_history")[3] > 20
