"""Scientific invariants for the Gaussian budget field, without natural labels."""

import numpy as np
import pytest

from experiments.native_support.routes import route_scores
from experiments.native_support.transport_state import budget_risk, infer_budget


def test_two_node_solution_matches_closed_form_mean_and_variance():
    budget = np.array([[[[3., 1., 2.]]], [[[0., 6., 3.]]]])
    edges = np.array([[0., 0.], [2., 0.]])
    result = infer_budget(budget, edges, strength=.5)

    # One unit-weight edge: precision [[2,-1],[-1,2]], inverse [[2,1],[1,2]]/3.
    expected = np.stack(((2 * budget[0] + budget[1]) / 3,
                         (budget[0] + 2 * budget[1]) / 3))
    np.testing.assert_allclose(result["inferred_budget"], expected)
    np.testing.assert_allclose(result["posterior_variance"], [2 / 3, 2 / 3])
    np.testing.assert_allclose(result["retention_weight"], [.5, .5])


def test_zero_strength_preserves_raw_budgets_and_routing():
    random = np.random.default_rng(4)
    budget = random.uniform(size=(5, 2, 3, 4))
    edges = np.tril(random.uniform(size=(5, 5)), k=-1)
    result = infer_budget(budget, edges, strength=0.)

    np.testing.assert_array_equal(result["inferred_budget"], budget)
    np.testing.assert_allclose(budget_risk(result["inferred_budget"], 2), budget_risk(budget, 2),
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(result["posterior_variance"], np.ones(5))
    np.testing.assert_array_equal(result["retention_weight"], np.zeros(5))


def test_constant_state_does_not_gain_risk_from_persistence():
    random = np.random.default_rng(5)
    state = random.uniform(size=(1, 2, 3, 4))
    budget = np.repeat(state, 7, axis=0)
    edges = np.tril(random.uniform(size=(7, 7)), k=-1).astype(np.float32)
    result = infer_budget(budget, edges, strength=20.)

    np.testing.assert_allclose(result["inferred_budget"], budget)
    np.testing.assert_allclose(budget_risk(result["inferred_budget"], 2), budget_risk(budget, 2))


def test_source_and_history_persistence_are_symmetric():
    budget = np.array([[[[5., 1., 2.]]], [[[1., 3., 2.]]], [[[2., 4., 1.]]]])
    edges = np.array([[0., 0., 0.], [.8, 0., 0.], [.2, .4, 0.]])
    positive = infer_budget(budget, edges)["inferred_budget"]
    swapped = infer_budget(budget[..., [1, 0, 2]], edges)["inferred_budget"]

    np.testing.assert_allclose(swapped, positive[..., [1, 0, 2]])
    np.testing.assert_allclose(budget_risk(swapped, 1), -budget_risk(positive, 1))


def test_supplied_zero_edges_isolate_a_reanchor_segment():
    budget = np.array([[[[1., 4., 1.]]], [[[2., 3., 1.]]],
                       [[[5., 1., 1.]]], [[[4., 2., 1.]]]])
    edges = np.zeros((4, 4))
    edges[1, 0] = .7
    edges[3, 2] = .8
    original = infer_budget(budget, edges)["inferred_budget"]
    changed = budget.copy()
    changed[:2] *= 100
    updated = infer_budget(changed, edges)["inferred_budget"]

    np.testing.assert_array_equal(original[2:], updated[2:])
    assert not np.allclose(original[:2], updated[:2])


def test_inference_preserves_nonnegative_budgets_and_coordinate_ranges():
    random = np.random.default_rng(6)
    budget = random.uniform(size=(8, 2, 4, 5))
    budget[0] = 0.
    edges = np.tril(random.uniform(size=(8, 8)), k=-1)
    result = infer_budget(budget, edges, strength=12.)
    inferred = result["inferred_budget"]

    assert inferred.shape == budget.shape
    assert np.all(inferred >= 0.)
    assert np.all(inferred >= budget.min(axis=0) - 1e-12)
    assert np.all(inferred <= budget.max(axis=0) + 1e-12)
    np.testing.assert_allclose(inferred.sum(axis=0), budget.sum(axis=0))
    assert np.all((result["posterior_variance"] > 0.) & (result["posterior_variance"] <= 1.))


def test_last_token_with_no_reuse_is_an_observation_not_zero_emission():
    budget = np.array([[[[2., 1., 1.]]], [[[4., 2., 1.]]], [[[9., 1., 1.]]]])
    edges = np.zeros((3, 3))
    edges[1, 0] = .6
    result = infer_budget(budget, edges)

    np.testing.assert_array_equal(result["inferred_budget"][-1], budget[-1])
    assert result["posterior_variance"][-1] == 1.
    assert result["retention_weight"][-1] == 0.


@pytest.mark.parametrize("answer_keys", [0, 1, 3])
def test_budget_readout_matches_historical_self_and_special_roles(answer_keys):
    prompt = 3
    count = prompt + answer_keys
    magnitude = np.arange(1., 4 * count + 1.).reshape(2, 2, count)
    attention = np.full_like(magnitude, 1 / count)
    groups = np.array([0, 0, 2] + [1] * answer_keys)
    if answer_keys:
        groups[-1] = 2  # Evidence route counts answer special/self as history.
    evidence = np.array([True, False, False])
    expected = route_scores(attention, magnitude, groups, prompt, evidence)["routing_imbalance"]

    source = magnitude[..., 0]
    history = magnitude[..., prompt:].sum(axis=-1)
    remaining = magnitude[..., :prompt].sum(axis=-1) - source
    budget = np.stack((source, history, remaining), axis=-1)[None]
    np.testing.assert_allclose(budget_risk(budget, 1), [expected])


def test_first_query_self_is_excluded_even_if_prompt_evidence_contains_it():
    magnitude = np.array([[[1., 3., 8.]]])
    evidence = np.array([True, False, True])
    expected = route_scores(magnitude / 12, magnitude, np.zeros(3, dtype=int), 3, evidence)
    budget = np.array([[[[1., 0., 11.]]]])

    assert budget_risk(budget, 1).item() == pytest.approx(expected["routing_imbalance"])
    assert budget_risk(budget, 1).item() == pytest.approx(-1 / 12)
