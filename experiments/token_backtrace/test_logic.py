import numpy as np

from .logic import apply_constraints, interval, interval_relation, polarity_claims, duration_claims


def test_open_bounds_and_one_way_entailment():
    assert interval_relation(interval('more than', 4), interval('', 4)) == 'contradiction'
    assert interval_relation(interval('at least', 4), interval('', 4)) == 'unknown'
    assert interval_relation(interval('', 4), interval('at least', 4)) == 'supported'
    assert interval_relation(interval('at most', 4), interval('at least', 4)) == 'unknown'


def test_polarity_scope_conflict_and_role():
    assert polarity_claims('The store is not open.', 'The store is open.')[0]['status'] == 'contradiction'
    assert polarity_claims('The store is not open.', 'The store is not open.')[0]['status'] == 'supported'
    assert polarity_claims('The school is not open.', 'The store is open.') == []
    assert polarity_claims('The store is not open. The store is open.', 'The store is open.') == []
    assert polarity_claims('If it rains the store is not open.', 'The store is open.') == []
    assert polarity_claims('The store is not open on Sundays.', 'The store is open.') == []


def test_duration_controls_and_alignment_abstention():
    source = 'For more than three weeks, the sailor was stranded on a remote island.'
    wrong = 'The sailor was stranded on the remote island for three weeks.'
    correct = 'The sailor was stranded on the remote island for more than three weeks.'
    assert duration_claims(source, wrong)[0]['status'] == 'contradiction'
    assert duration_claims(source, correct)[0]['status'] == 'supported'
    assert duration_claims(source, 'The opera played for three weeks.') == []
    assert duration_claims(source + ' Another boat travelled for two weeks.', wrong) == []


def test_unknown_preserves_native_and_no_averaging():
    native = np.array([.1, .4, .8])
    np.testing.assert_array_equal(apply_constraints(native, np.array([1, 0, -1])), [1., .4, 0.])
