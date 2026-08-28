from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.attention_mechanism_audit.functional_flow import functional_flow
from experiments.grounded_route.graph import TokenEdges, TokenGraph


def make_graph(*, heads=1, weights=None, diagonal=None, unresolved=None):
    weights = [0.4] * heads if weights is None else weights
    source = torch.zeros(heads, dtype=torch.long)
    target = torch.ones(heads, dtype=torch.long)
    edge_head = torch.arange(heads, dtype=torch.long)
    if diagonal is None:
        diagonal = [[([0.0] * heads)], [([1.0] * heads)]]
    diagonal = torch.tensor(diagonal, dtype=torch.float32)
    if unresolved is None:
        unresolved = torch.zeros_like(diagonal)
        unresolved[0, 0] = 1.0 - torch.as_tensor(weights)
    return TokenGraph(
        sample_id="sample",
        source_id="source",
        task_type="QA",
        response_start=1,
        token_count=3,
        response_count=2,
        layer_count=1,
        head_count=heads,
        attention_floor=0.0,
        edges=TokenEdges(
            source=source,
            target=target,
            layer=torch.zeros(heads, dtype=torch.long),
            head=edge_head,
            weight=torch.tensor(weights, dtype=torch.float32),
        ),
        diagonal=diagonal,
        unresolved=unresolved,
        token_ids=torch.tensor([10, 20, 30]),
    ).check().canonicalize()


def make_capture(*, heads=1, prompt_values=None, target_gradients=None):
    values = torch.zeros((1, 3, heads, 1), dtype=torch.float32)
    values[0, 0, :, 0] = torch.as_tensor(
        [2.0] * heads if prompt_values is None else prompt_values
    )
    gradients = torch.zeros((1, 2, heads, 1), dtype=torch.float32)
    gradients[0, 0, :, 0] = 100.0  # last-prompt predictor: token zero only
    gradients[0, 1, :, 0] = torch.as_tensor(
        [3.0] * heads if target_gradients is None else target_gradients
    )
    return SimpleNamespace(
        token_ids=torch.tensor([10, 20, 30]),
        predictor_indices=torch.tensor([0, 1]),
        target_ids=torch.tensor([20, 30]),
        value_states=values,
        o_proj_input_gradients=gradients,
        q_to_kv=torch.arange(heads),
        head_count=heads,
        kv_head_count=heads,
        head_dim=1,
    )


def test_single_edge_matches_explicit_phi_and_alignment():
    output = functional_flow(
        make_graph(), np.asarray([0]), make_capture()
    )
    # Cached query zero predicts response token one and must use gradient row
    # one (3), not the unavailable token-zero gradient row (100).
    assert np.isnan(output["functional_signed_role"][0]).all()
    assert np.isnan(output["functional_absolute_layer_role"][0]).all()
    assert np.isnan(output["functional_total_absolute"][0]).all()
    assert output["functional_available"].tolist() == [False, True]
    np.testing.assert_allclose(
        output["functional_signed_role"][1, 0, 0, 0],
        0.4 * 3.0 * 2.0,
    )
    np.testing.assert_allclose(output["functional_entropy_observed"][1, 0, 0], 0.0)
    np.testing.assert_allclose(output["functional_hhi_observed"][1, 0, 0], 1.0)
    assert output["functional_cached_query_index"].tolist() == [-1, 0]
    assert output["functional_predictor_position"].tolist() == [0, 1]


def test_opposite_heads_cancel_without_erasing_absolute_energy():
    graph = make_graph(heads=2, weights=[1.0, 1.0])
    capture = make_capture(
        heads=2,
        prompt_values=[1.0, 1.0],
        target_gradients=[2.0, -2.0],
    )
    output = functional_flow(graph, np.asarray([0]), capture)
    signed = output["functional_signed_layer_role"][1, 0, 0]
    energy = output["functional_absolute_layer_role"][1, 0, 0]
    np.testing.assert_allclose(signed, 0.0, atol=1e-7)
    np.testing.assert_allclose(energy, 4.0, atol=1e-7)
    np.testing.assert_allclose(output["functional_cancellation"][1, 0], 1.0)


def test_exact_diagonal_is_a_history_contribution():
    graph = make_graph(
        weights=[0.4],
        diagonal=[[[0.6]], [[1.0]]],
        unresolved=torch.zeros((2, 1, 1)),
    )
    capture = make_capture()
    capture.value_states[0, 1, 0, 0] = 4.0
    output = functional_flow(graph, np.asarray([0]), capture)
    np.testing.assert_allclose(
        output["functional_signed_role"][1, 0, 0, 4], 0.6 * 3.0 * 4.0
    )
    np.testing.assert_allclose(
        output["functional_known_attention_coverage"][1, 0, 0], 1.0
    )


def test_grouped_query_heads_use_recorded_kv_mapping():
    graph = make_graph(heads=2, weights=[0.5, 0.5])
    capture = make_capture(
        heads=2,
        prompt_values=[3.0, 99.0],
        target_gradients=[2.0, 4.0],
    )
    # Retain a single KV head and map both query heads to it.
    capture.value_states = capture.value_states[:, :, :1]
    capture.q_to_kv = torch.tensor([0, 0])
    capture.kv_head_count = 1
    output = functional_flow(graph, np.asarray([0]), capture)
    np.testing.assert_allclose(
        output["functional_signed_role"][1, 0, :, 0], [3.0, 6.0]
    )


def test_probe_mean_precedes_absolute_energy_and_signed_se_is_reported():
    capture = make_capture()
    probes = capture.o_proj_input_gradients[None].repeat(2, 1, 1, 1, 1)
    probes[0, 0, 1, 0, 0] = 1.0
    probes[1, 0, 1, 0, 0] = 5.0
    capture.o_proj_input_gradient_probes = probes
    capture.o_proj_input_gradients = probes.mean(dim=0)

    output = functional_flow(make_graph(), np.asarray([0]), capture)

    # phi = .4 * 2 * mean([1,5]) = 2.4; abs is applied after
    # averaging signed diagonal-Jacobian probes.
    np.testing.assert_allclose(
        output["functional_absolute_layer_role"][1, 0, 0], 2.4
    )
    # Probe signed role contributions are [.8, 4.0], whose sample standard
    # error is 1.6.
    np.testing.assert_allclose(
        output["functional_signed_layer_role_estimator_se"][1, 0, 0],
        1.6,
        atol=1e-6,
    )
