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


def test_crossfit_calibration_excludes_held_source_features():
    from .readout import crossfit_graph

    rng = np.random.default_rng(2)
    records = [dict(key=str(i), source_id=str(i), valid=np.ones(6, bool),
        attributes=rng.normal(size=(6, 38)), designs={'node_ridge': rng.normal(size=(6, 3))})
        for i in range(4)]
    _, before = crossfit_graph(records, 'node_ridge')
    records[0]['attributes'] *= 100
    records[0]['designs']['node_ridge'] += 50
    _, after = crossfit_graph(records, 'node_ridge')
    np.testing.assert_array_equal(before['0']['values'], after['0']['values'])
    assert '0' not in after['0']['fit_sources']
    assert '0' not in after['0']['calibration_sources']
