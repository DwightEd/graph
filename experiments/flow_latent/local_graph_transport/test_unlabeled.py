"""Reference separation, graph controls and distinct-token scoring contracts."""
import numpy as np

from .unlabeled import (chain_weights, equal_source_weights, fit_reference,
                        local_graph_edges, local_weights, reference_rank,
                        rewire_weights, score_answer)


def test_equal_source_weighting_and_duplicate_token_invariance():
    values = np.array([0., 10., 10., 10.])
    sources = np.array([0, 1, 1, 1])
    weights = equal_source_weights(sources)
    np.testing.assert_allclose([weights[sources == i].sum() for i in (0, 1)], [.5, .5])
    reference = fit_reference(values, values, values, sources)
    np.testing.assert_allclose(reference_rank(reference, 'source_local', [0., 10.]), [.25, .75])
    short = fit_reference(np.array([0., 10.]), np.array([0., 10.]),
                          np.array([0., 10.]), np.array([0, 1]))
    for name in reference:
        np.testing.assert_allclose(reference[name], short[name], rtol=0, atol=2e-16)


def test_weighted_mid_cdf_preserves_ties_monotonicity_and_frozen_reference():
    values = np.array([-3., 0., 0., 2.])
    reference = fit_reference(values, values, values, np.zeros(4))
    archived = {name: value.copy() for name, value in reference.items()}
    query = np.array([-100., -3., -1., 0., 0., 1., 2., 100.])
    rank = reference_rank(reference, 'source_full', query)
    np.testing.assert_allclose(rank, [0., .125, .25, .5, .5, .75, .875, 1.])
    assert (np.diff(rank) >= 0).all()
    for name in reference:
        np.testing.assert_array_equal(reference[name], archived[name])


def test_head_mean_uses_strict_past_posttoken_grid_and_ignores_padding():
    attention = np.full((4, 2, 3), 100.)
    attention[1, :, 0] = [0.1, 0.3]
    attention[2, :, :2] = [[.1, .2], [.3, .4]]
    weights = local_weights(attention)
    np.testing.assert_array_equal(weights[0], [0., 0., 0.])
    np.testing.assert_allclose(weights[1], [.2, 0., 0.])
    np.testing.assert_allclose(weights[2], [.2, .3, 0.])
    sources, targets, raw = local_graph_edges(weights)
    assert (sources < targets).all()
    assert len(raw) == 6


def test_chain_preserves_each_receiver_mass_and_only_uses_predecessor():
    weights = np.tri(10, 8, k=-1) * np.arange(1., 9.)
    chain = chain_weights(weights)
    np.testing.assert_allclose(chain.sum(axis=1), weights.sum(axis=1))
    np.testing.assert_array_equal(chain[:, 1:], 0.)
    sources, targets, raw = local_graph_edges(chain)
    np.testing.assert_array_equal(targets[raw > 0] - sources[raw > 0], 1)


def test_rewire_is_reproducible_preserves_mass_lag_groups_and_changes_endpoints():
    weights = np.tri(10, 8, k=-1) * np.arange(1., 9.)
    rewired = rewire_weights(weights, seed=42)
    np.testing.assert_array_equal(rewired, rewire_weights(weights, seed=42))
    np.testing.assert_array_equal(rewired[:, :2], weights[:, :2])
    np.testing.assert_allclose(rewired.sum(axis=1), weights.sum(axis=1))
    for target in range(len(weights)):
        for first, stop in ((2, 4), (4, 8)):
            slots = np.arange(first, min(stop, weights.shape[1], target))
            np.testing.assert_array_equal(np.sort(rewired[target, slots]),
                                          np.sort(weights[target, slots]))
    assert (rewired != weights).any()
    np.testing.assert_array_equal(rewired[np.tri(10, 8, k=-1) == 0], 0.)


def test_score_families_are_fixed_per_token_and_bounded_with_numeric_canaries():
    reference_values = np.linspace(-1., 1., 21)
    reference = fit_reference(reference_values, reference_values, reference_values,
                              np.zeros(21))
    local = np.array([-.8, .8, -.6, .6, -.4, .4, -.2, .2, 0., .9])
    full = local + .1
    route = -local
    attention = np.broadcast_to(np.linspace(.01, .08, 8), (10, 2, 8)).copy()
    result = score_answer(reference, local, full, route, attention)
    assert len(result['scores']) == 8
    source = .5 * (result['ranks']['source_local'] + result['ranks']['source_full'])
    np.testing.assert_array_equal(result['scores']['source_unary'], source)
    np.testing.assert_array_equal(result['scores']['source_route_unary'],
                                  .75 * source + .25 * result['ranks']['raw_route'])
    for name, score in result['scores'].items():
        assert score.shape == (10,)
        assert np.isfinite(score).all()
        assert score.max() - score.min() > .1
        unary = result['scores']['source_route_unary' if name.startswith('source_route')
                                 else 'source_unary']
        assert np.max(np.abs(score - unary)) <= .5 + 1e-12
    assert result['diagnostics']['offline']
    assert result['diagnostics']['labels_used'] is False
    for name in ('source', 'source_route'):
        for graph in ('native', 'chain', 'rewired'):
            assert result['diagnostics'][name][graph]['converged']
            assert result['diagnostics'][name][graph]['max_gradient'] <= 1e-8


def test_zero_graph_is_identity_without_span_broadcast():
    values = np.arange(5.)
    reference = fit_reference(values, values, values, np.zeros(5))
    result = score_answer(reference, values, values, values, np.zeros((5, 2, 8)))
    for family in ('source', 'source_route'):
        for graph in ('native', 'chain', 'rewired'):
            np.testing.assert_array_equal(result['scores'][f'{family}_{graph}_huber'],
                                          result['scores'][f'{family}_unary'])
    assert result['diagnostics']['rewired_weight_fraction'] == 0.
