"""Scientific invariants of the token contrast, independent of natural labels."""

import numpy as np

from .readout import contrasts, log_odds, pack_contrasts
from .span_metrics import character_counts


def test_odds_and_saturation():
    probability = np.array([.1, .5, .9])
    np.testing.assert_allclose(log_odds(np.log(probability)), np.log(probability / (1 - probability)))
    assert np.isfinite(log_odds(np.array([0., -1e-12, -100.]))).all()


def test_no_neighbour_score_dependence():
    present = np.log(np.array([.2, .5, .9]))
    absent = np.log(np.array([.1, .4, .95]))
    before = contrasts(present, absent, present, absent, np.zeros(3))
    after_present = present.copy()
    after_present[1] = np.log(.99)
    after = contrasts(after_present, absent, present, absent, np.zeros(3))
    for name in before:
        np.testing.assert_array_equal(before[name][[0, 2]], after[name][[0, 2]])
    assert before['odds_full'][0] < 0 < before['odds_full'][2]


def test_pack_reconstruction():
    metadata = dict(context=['full_unit', 'local_unit'], observations=[
        'with_full_logp', 'with_local_logp', 'full_deviation', 'local_deviation', 'route'])
    pack = dict(context=np.array([[.3, -.2]]), observations=np.array([[-2., -1., .1, -.1, .4]]))
    actual = pack_contrasts(pack, metadata)
    expected = contrasts(np.array([-2.]), np.array([-1.6]), np.array([-1.]), np.array([-1.3]), np.array([.4]))
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name])


def test_span_boundary_and_duplicate_token_offsets():
    # Two byte tokens may cover the same character. Count that character once,
    # and retain the exact gold boundary inside a larger alarm token.
    counts = character_counts('abcdef', [{'start': 1, 'end': 4}],
                              [(0, 2), (0, 2), (2, 3), (3, 6)], [True, True, True, False])
    np.testing.assert_array_equal(counts, [2, 1, 1, 1, 1, 0])


def test_history_graph_keeps_previous_token_and_excludes_self_future():
    from .readout import history_weights

    roots = np.zeros((4, 6))  # Three prompt roots and three input answer tokens.
    roots[1, 3] = -2.
    roots[3, 3:6] = [1., -3., 4.]
    actual = history_weights(roots, 3)
    np.testing.assert_array_equal(actual[1], [-2, 0, 0, 0])
    np.testing.assert_array_equal(actual[3], [1, -3, 4, 0])
    assert not np.any(np.triu(actual))


def test_matched_shuffle_preserves_signed_strata_and_causality():
    from .readout import matched_history

    rng = np.random.default_rng(10)
    weights = np.tril(rng.normal(size=(30, 30)), k=-1)
    tokens = np.arange(30) % 3
    changed = matched_history(weights, tokens, 17)
    assert not np.any(np.triu(changed))
    assert not np.array_equal(weights, changed)
    for target in range(1, 30):
        group = 2 * np.floor(np.log2(target - np.arange(target))).astype(int)
        group += tokens[:target] == tokens[target]
        for value in np.unique(group):
            indices = np.flatnonzero(group == value)
            np.testing.assert_array_equal(np.sort(weights[target, indices]), np.sort(changed[target, indices]))


def test_signed_context_uses_attributes_not_target_or_future():
    from .readout import signed_context

    weights = np.zeros((4, 4))
    weights[2, :2] = [2., -4.]
    attributes = np.arange(8).reshape(4, 2).astype(float)
    before = signed_context(weights, attributes)
    attributes[2:] = 1000
    after = signed_context(weights, attributes)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(before[2], [0, 1, 2, 3])


def test_expected_graph_does_not_cancel_opposite_sign_channels():
    from .readout import signed_context

    weights = np.zeros((4, 4))
    weights[3, :2] = [2., -2.]  # Both parents in lag {2,3}, same token-ID status.
    attributes = np.array([[1.], [5.], [20.], [90.]])
    actual = signed_context(weights, attributes, expected=True, token_ids=np.zeros(4, int))
    np.testing.assert_allclose(actual[3], [3., 3.])


def test_mass_context_retains_weak_edge_magnitude():
    from .readout import signed_context

    weights = np.array([[0., 0.], [2., 0.]])
    attributes = np.array([[3.], [5.]])
    small = weights * 1e-8
    np.testing.assert_allclose(signed_context(weights, attributes), signed_context(small, attributes))
    np.testing.assert_allclose(signed_context(small, attributes, preserve_mass=True),
                              1e-8 * signed_context(weights, attributes, preserve_mass=True))


def test_lineage_solves_triangular_equation_without_changing_root_signs_in_place():
    from .readout import attribution_lineage

    # Two prompt roots, four targets and three previous-answer root columns.
    roots = np.array([[-1., 0., 50., 50., 50.], [0., 1., -1., 50., 50.],
                      [0., 0., 0., -2., 50.], [0., 0., 0., 0., 0.]])
    original = roots.copy()
    result = attribution_lineage(roots, 2)
    np.testing.assert_array_equal(roots, original)
    np.testing.assert_array_equal(result['L'], np.tril(result['L'], -1))
    np.testing.assert_allclose(result['C'], result['B'] + result['L'] @ result['C'])
    np.testing.assert_allclose(result['C'][:3], [[1., 0.], [.5, .5], [.5, .5]])
    assert not result['resolved'][3]
    np.testing.assert_array_equal(result['C'][3], [0., 0.])


def native_effects(count=4, width=2):
    causal = np.triu(np.ones((count, count), bool), 1)
    valid = np.broadcast_to(causal[:, None, :], (count, width, count)).copy()
    eta = np.broadcast_to(np.where(valid, 0., np.nan), (3, 3, count, width, count)).copy()
    selection = np.tile(np.array([(0, head) for head in range(width)]), (count, 1, 1))
    return eta, selection, np.ones(count, bool), valid


def test_pair_statistic_matches_physical_heads_after_selection_order_changes():
    from .readout import selected_statistics

    eta, heads, aligned, valid = native_effects()
    heads[1] = heads[1, ::-1]
    eta[0, 0, 0, :, 3] = [8., 2.]
    eta[0, 0, 1, :, 3] = [7., 4.]
    result = selected_statistics(eta, heads, aligned, valid)
    assert result['pair'][0, 1] == 4.
    assert result['pair_target'][0, 1] == 3
    np.testing.assert_array_equal(result['pair_head'][0, 1], [0, 0])
    np.testing.assert_array_equal(result['node'][0], [8., 7., 0., 0.])
    assert result['edge'][0, 1, 3] == 7.


def test_no_shared_physical_head_has_zero_pair_statistic_not_fake_slot_matching():
    from .readout import selected_statistics

    eta, heads, aligned, valid = native_effects()
    heads[1, :, 0] = 1
    eta[0, 0, :2, :, 3] = 10.
    result = selected_statistics(eta, heads, aligned, valid)
    assert result['pair'][0, 1] == 0.
    assert not result['pair_eligible'][1]
    assert result['pair_target'][0, 1] == -1


def test_equivalent_and_unrelated_controls_use_their_own_directions():
    from .readout import selected_statistics

    eta, heads, aligned, valid = native_effects()
    eta[0, :, 0, 0, 1] = [1., 10., 0.]
    eta[1, :, 0, 0, 1] = [2., 5., 0.]
    eta[2, :, 0, 0, 1] = [1., 4., 6.]
    result = selected_statistics(eta, heads, aligned, valid)
    np.testing.assert_array_equal(result['edge'][:, 0, 1], [0., 3., 2.])


def test_missing_expected_effect_propagates_nan_instead_of_biased_maximum():
    from .readout import selected_statistics

    eta, heads, aligned, valid = native_effects()
    eta[0, 0, 0, 0, 3] = 10.
    eta[0, 1, 0, 1, 3] = np.nan  # One uncompleted donor comparison under R.
    result = selected_statistics(eta, heads, aligned, valid)
    assert np.isnan(result['edge'][0, 0, 3])
    assert np.isnan(result['node'][0, 0])
    assert np.isnan(result['pair'][0, 1])
    assert not result['node_complete'][0, 0]
    assert not result['complete']
    assert result['node_complete'][1, 0]
    valid[0, 0, 2] = False
    result = selected_statistics(eta, heads, aligned, valid)
    assert np.isnan(result['edge'][:, 0, 2]).all()


def test_alignment_gaps_and_noncausal_entries_are_separately_marked():
    from .readout import selected_statistics

    eta, heads, aligned, valid = native_effects()
    eta[:, :, 3, :, 0] = 999.  # Forbidden backward-in-time state cannot enter a max.
    eta[0, 0, 0, :, 3] = 2.
    aligned[1] = False
    result = selected_statistics(eta, heads, aligned, valid)
    assert np.isnan(result['edge'][:, 0, 1]).all()
    assert not result['edge_eligible'][0, 1]
    assert not result['structural_zero'][0, 1]
    assert result['structural_zero'][3, 0]
    np.testing.assert_array_equal(result['edge'][:, 3, 0], [0., 0., 0.])
    np.testing.assert_array_equal(result['node'][:, -1], [0., 0., 0.])
    assert result['node'][0, 0] == 2.


def test_pair_winner_ties_choose_earliest_target_then_lexical_physical_head():
    from .readout import selected_statistics

    eta, heads, aligned, valid = native_effects(count=5)
    heads[0] = heads[0, ::-1]
    eta[0, 0, :2, :, 2:] = 4.
    result = selected_statistics(eta, heads, aligned, valid)
    assert result['pair_target'][0, 1] == 2
    np.testing.assert_array_equal(result['pair_head'][0, 1], [0, 0])
