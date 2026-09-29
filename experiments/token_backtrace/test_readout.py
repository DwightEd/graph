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
