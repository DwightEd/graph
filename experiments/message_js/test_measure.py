import numpy as np
from scipy import sparse

from experiments.unsupervised_token_graph.channels import ChannelGraph
from experiments.unsupervised_token_graph.information import SourceFlow
from experiments.message_js.measure import direct_relay, js_rows, propagate, branch_alpha, branch_relative_js


def test_root_paths_match_independent_old_implementation():
    generator = np.random.default_rng(19)
    prompt, count = 7, 11
    attention = generator.uniform(size=(count, prompt + count - 1))
    queries = prompt - 1 + np.arange(count)
    attention[np.arange(attention.shape[1])[None] > queries[:, None]] = 0
    attention /= attention.sum(-1, keepdims=True)
    channel = ChannelGraph(0, 0, queries, sparse.csr_matrix(attention), prompt)
    old = SourceFlow().run(channel)
    np.testing.assert_allclose(direct_relay(attention, prompt), old['source_mismatch_bits'], atol=1e-12)


def test_disjoint_js_and_empty_branch():
    left = np.array([[1., 0.], [0., 0.]])
    right = np.array([[0., 1.], [0., 1.]])
    result = js_rows(left, right, np.array([.5, .5]))
    assert result[0] == 1
    assert np.isnan(result[1])


def test_relative_js_removes_branch_imbalance_ceiling():
    alpha = np.array([.5, .01, .99])
    left = np.tile([1., 0.], (3, 1))
    right = np.tile([0., 1.], (3, 1))
    divergence = js_rows(left, right, alpha)
    np.testing.assert_allclose(branch_relative_js(divergence, alpha), 1.)
    assert divergence[1] < .1


def test_branch_mass_recursion_uses_query_state_parent():
    weights = np.array([[.5, .5, 0., 0.], [.1, .2, .7, 0.], [.1, .1, .6, .2]])
    # Last row relays .6 through row1, whose prompt reach is .3.
    alpha = branch_alpha(weights, 2)
    np.testing.assert_allclose(alpha, [1., 1., .2 / (.2 + .6 * .3)])


def test_token_risk_parent_is_previous_prediction_not_query_row():
    weights = np.array([[1., 0., 0.], [.25, .75, 0.], [.5, 0., .5]])
    result = propagate(np.array([1., 0., 0.]), weights, prompt=1)
    np.testing.assert_allclose(result, [1., .75, .375])


def test_propagation_preserves_constant_state_and_causality():
    generator = np.random.default_rng(9)
    count, prompt = 15, 5
    weights = generator.random((count, prompt + count - 1))
    query = prompt - 1 + np.arange(count)
    weights[np.arange(weights.shape[1])[None] > query[:, None]] = 0
    weights /= weights.sum(-1, keepdims=True)
    for shuffled in (False, True):
        result = propagate(np.full(count, .37), weights, prompt, shuffled)
        np.testing.assert_allclose(result, .37, atol=1e-12)
        changed = np.full(count, .37)
        changed[-1] = .99
        np.testing.assert_allclose(propagate(changed, weights, prompt, shuffled)[:-1], result[:-1])


def test_reference_cdf_preserves_ties_source_weights_and_missingness():
    from experiments.message_js.score import fit_cdf, percentile
    # Two sources: three zero tokens vs one token equal to one; source weights .5/.5.
    fitted = fit_cdf(np.array([0., 0., 0., 1., np.nan]), np.array([1/6, 1/6, 1/6, .5, 7.]))
    result = percentile(np.array([-1., 0., .5, 1., 2., np.nan]), fitted)
    np.testing.assert_allclose(result[:-1], [0., .25, .5, .75, 1.])
    assert np.isnan(result[-1])
