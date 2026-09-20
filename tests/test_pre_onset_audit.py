"""Scientific contracts: excluded sinks, timing, multiplicity and denominators."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from experiments.unsupervised_token_graph.channels import ChannelGraph
from experiments.unsupervised_token_graph.span_audit.onset_changes import (
    before_peaks, head_changes, old_endpoint_change)
from experiments.unsupervised_token_graph.span_audit.onset_report import incidence
from experiments.unsupervised_token_graph.span_audit.onset_windows import compare_window
from experiments.charm_structure_audit.score_onsets import describe


def fixture():
    answer = SimpleNamespace(prompt_length=3, response_ids=np.arange(8))
    attention = np.zeros((8, 11))
    attention[:, 0] = .1
    attention[:, 1] = .2
    attention[:, 2] = .1
    return answer, attention


def calculate(attention, special):
    answer, _ = fixture()
    channel = ChannelGraph(2, 30, np.arange(2, 10), csr_matrix(attention), 3)
    return head_changes(channel, answer, special, 3)


def test_all_special_ids_excluded_without_renormalizing_retained_attention():
    _, attention = fixture()
    attention[3:, 0] = .4
    attention[3:, 2] = .3
    special = np.zeros(11, dtype=bool)
    special[[0, 2]] = True
    gain, endpoint, _, _ = calculate(attention, special)
    assert gain[3] == 0
    assert endpoint[3] == -1
    attention[3:, 1] += .1
    gain, endpoint, _, _ = calculate(attention, special)
    assert gain[3] == pytest.approx(.1)
    assert endpoint[3] == 1


def test_aging_endpoint_alone_does_not_make_a_reanchor():
    row = (np.array([1, 5]), np.array([.1, .8]))
    gain, _, _ = old_endpoint_change(row, row, prompt_length=3, query=8, local_window=3)
    assert gain == 0


def test_query_predicts_next_token_and_future_does_not_change_before_or_decision():
    _, attention = fixture()
    special = np.zeros(11, dtype=bool)
    attention[3:, 1] += .2  # q=5 predicts answer t=3, not t=2.
    gain, _, _, _ = calculate(attention, special)
    assert gain[3] == pytest.approx(.2)
    assert before_peaks(gain, 2)[3] == 0
    assert before_peaks(gain, 2)[4] == pytest.approx(.2)
    attention[4:, 1] += .3
    changed, _, _, _ = calculate(attention, special)
    np.testing.assert_equal(changed[:4], gain[:4])


def test_missing_head_rows_propagate_to_joint_window_missingness():
    first = np.array([0., .1, np.nan, .2, 0.])
    second = np.array([0., .8, .9, .1, .1])
    joint = np.maximum(before_peaks(first, 2), before_peaks(second, 2))
    assert np.isnan(joint[3])
    result = compare_window(joint, 3, np.array([2, 4]), 1, .95, .05)
    assert result['status'] == 'missing_attention_rows'
    assert result['event'] is None


def test_joint_maximum_uses_joint_normal_reference_and_holds_out_control():
    # A large error-head value is ordinary when normal windows also have large heads.
    values = np.array([.7, .8, .9, .4, .6])
    result = compare_window(values, 0, np.array([1, 2, 3, 4]), 3, .95, .05)
    assert result['normal_onset'] == 1
    assert result['reference_windows'] == 3
    assert result['threshold'] == .9
    assert result['event'] is False


def test_no_control_does_not_become_absence_and_denominator_keeps_every_onset():
    missing = compare_window(np.array([.5]), 0, np.array([], dtype=int), 20, .95, .05)
    assert missing['event'] is None
    frame = pd.DataFrame([
        dict(source_id='s1', status='measured', event=True, normal_event=False),
        dict(source_id='s2', status='measured', event=False, normal_event=False),
        dict(source_id='s3', **missing),
    ])
    result = incidence(frame)
    assert result['annotated_onsets'] == 3
    assert (result['events'], result['absent'], result['unavailable']) == (1, 1, 1)


def test_score_onset_can_be_missed_then_detected_later_and_short_history_is_missing():
    row = describe(np.array([.1]), np.array([.2, .3, .9, .9]), .8, 8)
    assert row['before_alarm'] is None
    assert not row['onset_alarm'] and row['first4_alarm']
    assert row['first_alarm_offset'] == 2
    assert not row['first4_sustained']


def test_complete_answer_report_retains_missing_early_onset_and_negative_later_onset(tmp_path):
    from experiments.unsupervised_token_graph.span_audit.tests.test_audit import answer_fixture, channel_fixture
    from experiments.unsupervised_token_graph.span_audit.units import Span, span_mask
    from experiments.unsupervised_token_graph.span_audit.onset_run import measure_answer
    from experiments.unsupervised_token_graph.span_audit.onset_report import summarize_onsets

    answer = answer_fixture()
    answer.spans = [Span(0, 2), Span(20, 24)]
    answer.error_mask = span_mask(len(answer.response_ids), answer.spans)
    inputs = SimpleNamespace(channels=lambda *args: [channel_fixture(answer, head=0), channel_fixture(answer, head=1)])
    args = SimpleNamespace(window=2, local_window=3, position_gap=1., minimum_controls=1,
                           quantile=.95, minimum_gain=.05, layers=None, heads=None)
    destination = tmp_path / 'samples' / '1'
    destination.mkdir(parents=True)
    measure_answer(inputs, answer, np.zeros(len(answer.token_ids), bool), args, destination)
    pd.DataFrame([dict(id='1', cached=True), dict(id='2', cached=False)]).to_csv(
        tmp_path / 'population_coverage.csv', index=False)
    summary = summarize_onsets(tmp_path)
    assert summary['annotated_onsets'] == 2 and summary['missing_answers'] == 1
    onsets = pd.read_csv(tmp_path / 'onsets.csv')
    before = onsets[onsets.phase.eq('strict_before')].set_index('onset')
    assert before.loc[0, 'status'] == 'missing_attention_rows'
    assert before.loc[20, 'status'] == 'measured'
    assert before.loc[20, 'event'] == False
    heads = pd.read_csv(tmp_path / 'head_incidence.csv')
    assert set(heads['head']) == {0, 1}
