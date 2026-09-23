"""Native source/head alignment and boundary invariants for reuse kernels."""

import numpy as np
import pytest

from experiments.native_support.transport_graph import build_graph


def native_arrays(count=5, layers=1, heads=2, sources=2):
    shape = (count, layers, heads)
    effect = np.zeros((*shape, sources + 4, 2), dtype=np.float32)
    effect[..., 0, 0] = 1.
    effect[..., 1, 1] = 1.
    attention = np.zeros((*shape, sources + 4), dtype=np.float32)
    attention[..., :sources] = .7 / sources
    attention[..., sources:] = [.15, .1, .05, 0.]
    history = np.zeros((*shape, count), dtype=np.float32)
    for token in range(1, count):
        history[token, ..., :token] = .25 / token
    return {
        "group_effect": effect,
        "group_attention": attention,
        "ffn_state": np.ones((count, layers, 2), dtype=np.float32),
        "history_attention": history,
    }


def test_constant_signatures_preserve_actual_reuse_and_query_alignment():
    arrays = native_arrays()
    graph = build_graph(arrays, 2)

    np.testing.assert_array_equal(graph["edge_weight"], graph["reuse_weight"])
    np.testing.assert_array_equal(graph["head_bandwidth"], np.ones((1, 2)))
    assert graph["edge_weight"][2, 1] == pytest.approx(.25 / 2)
    assert graph["edge_weight"][2, 0] == 0.
    np.testing.assert_array_equal(np.triu(graph["edge_weight"]), 0.)
    np.testing.assert_allclose(
        graph["degree"], graph["edge_weight"].sum(0) + graph["edge_weight"].sum(1),
    )


def test_self_attention_creates_no_reuse_edges():
    arrays = native_arrays()
    arrays["history_attention"].fill(0.)
    for token in range(1, 5):
        arrays["history_attention"][token, ..., token - 1] = 1.
    graph = build_graph(arrays, 2)

    np.testing.assert_array_equal(graph["edge_weight"], 0.)
    np.testing.assert_array_equal(graph["reuse_weight"], 0.)


def test_one_row_head_swap_changes_affinity_despite_equal_head_mean():
    arrays = native_arrays()
    arrays["group_effect"][:, :, 1, :2] *= -1.
    baseline = build_graph(arrays, 2)
    arrays["group_effect"][3] = arrays["group_effect"][3, :, ::-1].copy()
    swapped = build_graph(arrays, 2)

    np.testing.assert_array_equal(arrays["group_effect"][3].mean(axis=1), 0.)
    assert swapped["edge_weight"][3, 1] < baseline["edge_weight"][3, 1]


def test_kernel_is_weighted_by_its_own_heads_history_attention():
    arrays = native_arrays()
    arrays["group_effect"][3, 0, 0] *= -1.
    arrays["history_attention"].fill(0.)
    arrays["history_attention"][3, 0, 0, 0] = 1.
    changed_head = build_graph(arrays, 2)
    arrays["history_attention"][3, 0] = arrays["history_attention"][3, 0, ::-1].copy()
    stable_head = build_graph(arrays, 2)

    np.testing.assert_array_equal(changed_head["reuse_weight"], stable_head["reuse_weight"])
    assert changed_head["edge_weight"][3, 1] == pytest.approx(.5 * np.exp(-1))
    assert stable_head["edge_weight"][3, 1] == pytest.approx(.5)


def test_consistent_source_permutation_preserves_all_edges():
    arrays = native_arrays()
    arrays["group_effect"][2:, ..., 0, 1] = .3
    arrays["group_attention"][3:, ..., :2] = [.5, .2]
    baseline = build_graph(arrays, 2)
    for name in ("group_effect", "group_attention"):
        values = arrays[name]
        source_axis = -2 if name == "group_effect" else -1
        order = [1, 0, 2, 3, 4, 5]
        arrays[name] = np.take(values, order, axis=source_axis)
    permuted = build_graph(arrays, 2)

    np.testing.assert_allclose(permuted["edge_weight"], baseline["edge_weight"], rtol=1e-6)
    np.testing.assert_allclose(permuted["head_bandwidth"], baseline["head_bandwidth"], rtol=1e-6)


@pytest.mark.parametrize("field", ["group_effect", "ffn_state"])
def test_sign_reversal_changes_similarity_without_defining_signed_risk(field):
    arrays = native_arrays()
    baseline = build_graph(arrays, 2)
    arrays[field][3] *= -1.
    changed = build_graph(arrays, 2)

    assert changed["edge_weight"][3, 1] < baseline["edge_weight"][3, 1]
    assert set(changed) == {"edge_weight", "reuse_weight", "degree", "head_bandwidth"}
    assert np.all(changed["edge_weight"] >= 0.)


def test_global_ffn_sign_change_is_not_a_risk_penalty():
    arrays = native_arrays()
    arrays["ffn_state"][2:] = [1., -.3]
    baseline = build_graph(arrays, 2)
    arrays["ffn_state"] *= -1.
    reversed_sign = build_graph(arrays, 2)

    for name in baseline:
        np.testing.assert_array_equal(reversed_sign[name], baseline[name])


def test_source_focus_switch_breaks_reuse_without_prompt_to_local_switch():
    arrays = native_arrays(count=5)
    arrays["group_effect"].fill(0.)
    arrays["group_effect"][:3, ..., 0, 0] = 1.
    arrays["group_effect"][3:, ..., 1, 0] = 1.
    arrays["group_effect"][1, ..., 0, 1] = .01
    arrays["group_effect"][4, ..., 1, 1] = .01
    arrays["group_attention"][:3, ..., :2] = [.65, .05]
    arrays["group_attention"][3:, ..., :2] = [.05, .65]
    graph = build_graph(arrays, 2)

    np.testing.assert_allclose(arrays["group_attention"][..., :2].sum(-1), .7)
    assert graph["edge_weight"][3, 2] < 1e-6 * graph["reuse_weight"][3, 2]
    assert graph["edge_weight"][2, 1] > .1 * graph["reuse_weight"][2, 1]


def test_missing_future_flags_do_not_create_boundary_edges_or_states():
    arrays = native_arrays()
    baseline = build_graph(arrays, 2)
    arrays["future_observed"] = np.array([True, True, True, False, False])
    arrays["future_attention_mean"] = np.full((5, 1, 2), np.nan)
    arrays["choice_effect"] = np.full((5, 1, 2, 6, 4), 1e10)
    masked = build_graph(arrays, 2)

    for name in baseline:
        np.testing.assert_array_equal(masked[name], baseline[name])
    assert baseline["edge_weight"][-1].sum() > 0.


@pytest.mark.parametrize("count", [1, 5])
def test_head_batches_and_layer_identity_preserve_results(count):
    arrays = native_arrays(count=count, layers=2, heads=3)
    arrays["group_effect"][::2, 1, 2, 0, 1] = .4
    single = build_graph(arrays, 2, head_batch=1)
    batched = build_graph(arrays, 2, head_batch=4)

    for name in single:
        np.testing.assert_allclose(batched[name], single[name], rtol=1e-6, atol=1e-8)
