import numpy as np
import pytest
from revisit_state import head_statistics, summarize_events, hold_state


def fixture_attention(n=40):
    p = 7
    a = np.zeros((2, n, p+n), dtype=np.float32)
    for t in range(n):
        a[:, t, 1] = .65 if t < 23 else .05
        a[:, t, 5] = .05 if t < 23 else .65
        a[:, t, 0] = .20
        a[:, t, p+t-1] += .1
    return a, np.array([False, True, True, True, True, True, False]), np.zeros(p+n, bool)


def test_old_operator_equivalence():
    from decoding.reading_graph import RevisitSignal
    a, source, special = fixture_attention()
    old = RevisitSignal(a[None], source, special, 16, .95).run()
    shift, mass = head_statistics(a, source, special)
    new = summarize_events(shift[None], mass[None])
    np.testing.assert_allclose(new['revisit'][2:], old['revisit'][2:], atol=1e-10)
    np.testing.assert_array_equal(new['event'], old['event'])


def test_causality_and_movement():
    a, source, special = fixture_attention()
    shift, mass = head_statistics(a, source, special)
    result = summarize_events(shift[None], mass[None])
    assert result['event'][23]
    short_shift, short_mass = head_statistics(a[:, :24, :31], source, special[:31])
    short = summarize_events(short_shift[None], short_mass[None])
    np.testing.assert_allclose(result['revisit'][:24], short['revisit'], equal_nan=True)
    assert not result['event'][:17].any()


def test_sink_change_not_content_event():
    a, source, special = fixture_attention()
    a[:, :, 1] = .6
    a[:, :, 5] = .1
    a[:, 23:, 1] = .3
    a[:, 23:, 5] = .05
    a[:, 23:, 0] = .55
    s, m = head_statistics(a, source, special)
    assert np.max(s) < 1e-6
    assert not summarize_events(s[None], m[None])['event'].any()


def test_invalid_future_rejected():
    a, source, special = fixture_attention()
    a[0, 10, -1] = .1
    with pytest.raises(ValueError, match='future'):
        head_statistics(a, source, special)


def test_state_updates_only_at_event():
    h = np.array([1., 2., 3., 4., 5., 6.])
    events = np.array([False, False, True, False, False, True])
    np.testing.assert_equal(hold_state(h, events), [1., 2., 3., 3., 3., 6.])
    np.testing.assert_equal(hold_state(h[:5], events[:5]), [1., 2., 3., 3., 3.])
