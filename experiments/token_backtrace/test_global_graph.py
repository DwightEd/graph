"""Exact optimization and evidence/coverage contracts, without natural labels."""

from itertools import product

import numpy as np
import pytest

from .global_graph import (graph_energy, reference_scale, reference_threshold,
                           score_events, solve_energy)


def enumerate_energy(source, support, carrier, directed, continuity):
    states = np.asarray(list(product((0, 1), repeat=len(source))))
    unary = .5 + support - source - carrier
    energies = np.asarray([graph_energy(state, unary, directed, continuity) for state in states])
    return states, energies


def event(count=5, **changes):
    result = dict(event_id='relation:alternative', anchor=0, source_conflict=2.,
                  source_support=0., external_conflict=True, external_support=False,
                  source_reference_resolved=True, edit_mask=np.arange(count) == 0,
                  support=np.zeros(count), carrier=np.zeros(count),
                  directed=np.zeros((count, count)), continuity=np.zeros(count),
                  scope=np.ones(count, bool), aligned=np.ones(count, bool),
                  reference_resolved=np.ones(count, bool),
                  measurement_complete=np.ones(count, bool))
    result.update(changes)
    return result


def test_exact_cuts_and_all_min_marginals_match_enumeration():
    rng = np.random.default_rng(17)
    for _ in range(40):
        source, support, carrier = rng.uniform(0, 1.5, (3, 6))
        directed = np.triu(rng.uniform(0, .8, (6, 6)), 1)
        continuity = np.r_[0., rng.uniform(0, .375, 5)]
        answer = solve_energy(source, support, carrier, directed, continuity)
        states, energies = enumerate_energy(source, support, carrier, directed, continuity)
        assert answer['energy'] == pytest.approx(energies.min())
        assert answer['cut_count'] == 7
        for token in range(6):
            minimum_zero = energies[states[:, token] == 0].min()
            minimum_one = energies[states[:, token] == 1].min()
            assert answer['min_marginal'][token] == pytest.approx(minimum_zero - minimum_one)
            assert answer['opposite_labels'][token, token] != answer['labels'][token]


def test_minimal_source_tie_uses_no_epsilon():
    # z0 is genuinely favoured. Neutral z1/z2 must remain absent even though
    # alternative optimum partitions containing them exist.
    answer = solve_energy([1., .5, .5], [0., 0., 0.], [0., 0., 0.],
                          np.zeros((3, 3)), np.zeros(3))
    np.testing.assert_array_equal(answer['labels'], [True, False, False])
    np.testing.assert_array_equal(answer['min_marginal'], [.5, 0., 0.])


def test_connected_ties_choose_intersection_of_all_optimal_source_sets():
    rng = np.random.default_rng(29)
    for _ in range(25):
        source = .25 * rng.integers(0, 5, 5)
        support = .25 * rng.integers(0, 5, 5)
        directed = np.triu(.25 * rng.integers(0, 3, (5, 5)), 1)
        continuity = np.r_[0., .25 * rng.integers(0, 3, 4)]
        solved = solve_energy(source, support, np.zeros(5), directed, continuity)
        states, energies = enumerate_energy(source, support, np.zeros(5), directed, continuity)
        intersection = states[energies == energies.min()].all(axis=0)
        np.testing.assert_array_equal(solved['labels'], intersection)


def test_directed_capacity_has_the_correct_orientation():
    directed = np.array([[0., 2.], [0., 0.]])
    unary = np.zeros(2)
    assert graph_energy([1, 0], unary, directed, np.zeros(2)) == 2.
    assert graph_energy([0, 1], unary, directed, np.zeros(2)) == 0.


def test_two_sides_can_fill_one_neutral_token_without_extending_the_tail():
    answer = solve_energy([2., 0., 0., 0.], np.zeros(4), [0., 0., 2., 0.],
                          np.zeros((4, 4)), [0., .375, .375, .375])
    np.testing.assert_array_equal(answer['labels'], [1, 1, 1, 0])
    assert answer['min_marginal'][1] > 0 > answer['min_marginal'][3]


def test_long_span_needs_independent_carriers_not_one_seed():
    answer = solve_energy(np.r_[1., np.zeros(9)], np.zeros(10), np.zeros(10),
                          np.zeros((10, 10)), np.r_[0., np.full(9, .375)])
    np.testing.assert_array_equal(answer['labels'], np.r_[True, np.zeros(9, bool)])


def test_external_conflict_is_required_even_with_large_repeated_head_effects():
    record = event(external_conflict=False, carrier=np.full(5, 100.),
                   directed=np.triu(np.full((5, 5), 100.), 1),
                   continuity=np.r_[0., np.full(4, 100.)])
    answer = score_events([record], 5)
    assert not answer['alarm'].any()
    assert answer['low_evidence'].all()


def test_scope_and_support_break_continuity_while_direct_survives_graph_threshold():
    record = event(carrier=np.full(5, 2.), scope=np.array([1, 1, 0, 1, 1]),
                   continuity=np.r_[0., np.ones(4)], support=np.array([0, 0, 0, 4, 0.]))
    answer = score_events([record], 5, graph_threshold=100.)
    assert answer['alarm'][0]
    assert not answer['events'][0]['labels'][2]
    assert not answer['events'][0]['labels'][3]
    assert not answer['events'][0]['labels'][4]
    assert not answer['events'][0]['usable'][2]


def test_unknown_scope_does_not_become_truthy():
    record = event(carrier=np.ones(5), scope=np.array([1, 1, -1, 1, 1]))
    answer = score_events([record], 5)
    assert not answer['events'][0]['usable'][2]
    assert not answer['events'][0]['masks']['scope'][2]
    assert answer['events'][0]['masks']['scope_unknown'][2]
    assert not answer['alarm'][2]


def test_outgoing_capacity_is_capped_and_final_token_has_no_carrier_future():
    record = event(carrier=np.full(5, 2.), directed=np.triu(np.full((5, 5), 10.), 1),
                   continuity=np.r_[0., np.ones(4)])
    answer = score_events([record], 5)['events'][0]['capacities']
    assert np.all(answer['directed'].sum(axis=1) <= 1.)
    assert answer['carrier'][-1] == 0.
    assert np.max(answer['continuity']) == .375


def test_missing_measurements_remain_incomplete_and_all_tokens_remain_present():
    record = event(carrier=np.full(5, 2.), measurement_complete=np.array([1, 1, 0, 1, 1]))
    answer = score_events([record], 5)
    assert not answer['complete']
    assert len(answer['risk']) == 5
    assert not answer['measurement_complete'][2]
    assert not answer['events'][0]['usable'][2]
    assert answer['low_evidence'][2]


def test_current_event_cannot_mark_prefork_carriers_as_members():
    record = event(anchor=2, edit_mask=np.array([0, 0, 1, 0, 0]),
                   carrier=np.full(5, 3.), directed=np.triu(np.ones((5, 5)), 1))
    answer = score_events([record], 5)
    assert not answer['alarm'][:2].any()
    assert not answer['events'][0]['usable'][:2].any()
    assert not answer['events'][0]['capacities']['directed'][:2].any()


def test_support_can_make_direct_score_lower_than_the_empty_event_default():
    record = event(external_conflict=False, external_support=True, source_support=2.)
    answer = score_events([record], 5)
    assert answer['direct'][0] == -2.5
    assert answer['direct'][1] == -.5
    assert not answer['alarm'].any()


def test_adjacent_alarms_from_different_events_keep_their_own_spans():
    first = event(count=3, event_id='a', anchor=0, edit_mask=np.array([1, 0, 0]))
    second = event(count=3, event_id='b', anchor=1, edit_mask=np.array([0, 1, 0]))
    answer = score_events([first, second], 3)
    assert answer['events'][0]['spans'] == [(0, 1)]
    assert answer['events'][1]['spans'] == [(1, 2)]
    np.testing.assert_array_equal(answer['alarm'], [1, 1, 0])


def test_no_proposal_has_numeric_low_evidence_outputs():
    answer = score_events([], 4)
    np.testing.assert_array_equal(answer['risk'], np.full(4, -.5))
    assert answer['low_evidence'].all()
    assert not answer['alarm'].any()


def records(count=32, value=1.):
    return [dict(source_id=str(index), record_id='a', value=value) for index in range(count)]


def test_reference_ties_deduplication_source_weights_and_resolution():
    pool = records()
    pool += [dict(source_id='0', record_id='b', value=3.)]
    pool += [pool[0]] * 5
    answer = reference_scale(np.array([1., 2., 4.]), pool)
    # Only source 0 has one of its two records >= 2; five duplicates have no weight.
    np.testing.assert_allclose(answer['tail'], [1., 1.5 / 33, 1 / 33])
    assert answer['score'][2] > .5
    assert answer['duplicate_count'] == 5
    assert answer['source_count'] == 32
    assert answer['record_count'] == 33


def test_unresolved_reference_is_nan_and_numerical_noise_does_not_become_evidence():
    unresolved = reference_scale(np.array([4.]), records(31))
    assert not unresolved['resolved']
    assert np.isnan(unresolved['score']).all()
    tiny = reference_scale(np.array([1e-10]), records(value=0.))
    assert tiny['score'][0] == 0.
    assert tiny['below_numerical_resolution'][0]


def test_reference_quantile_weights_sources_and_clamps_negative_thresholds():
    pool = [dict(source_id='short', record_id='a', value=10.)]
    pool += [dict(source_id='long', record_id=str(i), value=0.) for i in range(100)]
    assert reference_threshold(pool, quantile=.75) == 10.
    assert reference_threshold(records(value=-2.)) == 0.
