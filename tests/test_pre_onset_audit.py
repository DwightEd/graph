"""Scientific validity of label-free switches and bidirectional span association."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from experiments.unsupervised_token_graph.channels import ChannelGraph
from experiments.unsupervised_token_graph.span_audit.onset_changes import (
    MEASURES, aggregate_heads, classify, event_starts, measure_head, switch_bounds)
from experiments.unsupervised_token_graph.span_audit.onset_detection import save_detection, scan_channels
from experiments.unsupervised_token_graph.span_audit.onset_run import save_associations
from experiments.unsupervised_token_graph.span_audit.onset_windows import preceding_state, position_rows, span_rows
from experiments.unsupervised_token_graph.span_audit.units import Answer, Span, span_mask
from experiments.charm_structure_audit.score_onsets import describe


def fixture(switch_query=13):
    length, prompt = 40, 3
    attention = np.zeros((length, length))
    for query in range(prompt, length):
        old = .8 if query >= switch_query else .2
        attention[query, 1] = old
        attention[query, query] = 1-old
    channel = ChannelGraph(2, 30, np.arange(length), csr_matrix(attention), prompt)
    text = 'a'*(length-prompt)
    offsets = np.array([[i, i+1] for i in range(len(text))])
    spans = [Span(14, 18)]
    answer = Answer('1', 's1', 'QA', 'fixture', 'test', text, np.arange(length), prompt,
                    offsets, span_mask(len(text), spans), spans, np.full(len(text), np.nan), [])
    args = SimpleNamespace(baseline_steps=3, local_window=10, minimum_shift=.1, window=8)
    return answer, channel, args, np.zeros(length, bool)


def states_for(channel, answer, args, special):
    measures, _ = measure_head(channel, answer.token_ids, answer.prompt_length,
                               special, args.baseline_steps, args.local_window)
    return classify(measures, .1), measures


def test_actual_switch_requires_local_drop_old_rise_and_dominance_reversal():
    answer, channel, args, special = fixture()
    states, measures = states_for(channel, answer, args, special)
    target = 13+1-answer.prompt_length
    assert states[target] == 1
    readings = dict(zip(MEASURES, measures[target]))
    assert readings['shift_low'] == pytest.approx(.6)
    assert states[target-1] == 0
    assert states[target+4] == 0  # Staying global is not a new switch.


@pytest.mark.parametrize('previous,current', [
    ((.4, .6, 0), (.4, .6, 0)),  # Old A→old B redistribution.
    ((.8, .2, 0), (.9, .1, 0)),  # Always global, no local departure.
    ((.1, .2, 0), (.7, .2, 0)),  # Special-token release, no local drop.
    ((.1, .7, 0), (.1, .1, 0)),  # Local→special, no old increase.
])
def test_redistribution_static_global_and_special_sinks_are_not_switches(previous, current):
    assert classify(switch_bounds(previous, current)[None], .1)[0] == 0


def test_ageing_does_not_move_mass_between_sets_in_the_comparison():
    answer, channel, args, special = fixture()
    attention = np.zeros(channel.attention.shape)
    attention[5:, 5] = 1.
    channel.attention = csr_matrix(attention)
    states, _ = states_for(channel, answer, args, special)
    assert not np.any(states == 1)


def test_sparse_bounds_do_not_turn_missing_weights_into_zero_or_restore_them():
    observed = switch_bounds((.1, .5, .4), (.5, .1, .4))
    assert classify(observed[None], .1)[0] == -1
    strong = switch_bounds((.1, .8, .1), (.8, .1, .1))
    assert classify(strong[None], .1)[0] == 1
    rng = np.random.default_rng(4)
    for _ in range(100):
        previous = rng.dirichlet(np.ones(3))*.1
        current = rng.dirichlet(np.ones(3))*.1
        complete = switch_bounds((.1+previous[0], .8+previous[1], 0),
                                 (.8+current[0], .1+current[1], 0))
        assert classify(complete[None], .1)[0] == 1


def test_special_source_excluded_and_future_rows_cannot_change_current_selection():
    answer, channel, args, special = fixture()
    original, _ = states_for(channel, answer, args, special)
    changed = channel.attention.toarray()
    changed[17:, :] = 0
    changed[17:, 0] = 1
    channel.attention = csr_matrix(changed)
    later, _ = states_for(channel, answer, args, special)
    np.testing.assert_equal(original[:15], later[:15])
    special[[0, 1]] = True
    without_source, _ = states_for(channel, answer, args, special)
    assert not np.any(without_source == 1)


def test_one_certified_head_suffices_but_uncertain_heads_prevent_negative_claim():
    states = np.array([[0, -1, -1, 1], [0, 0, 1, -1]])
    np.testing.assert_equal(aggregate_heads(states), [0, -1, 1, 1])


def test_contiguous_head_switches_are_one_episode_without_future_peak_selection():
    np.testing.assert_equal(event_starts(np.array([0, 1, 1, 0, 1, -1, 1])), [0, 1, 0, 0, 1, -1, 1])


def test_strict_pre_onset_excludes_decision_and_unknown_is_not_absence():
    states = np.zeros(20, dtype=int)
    states[10] = 1
    assert preceding_state(states, 10, 4) == 0
    assert preceding_state(states, 11, 4) == 1
    states[3] = -1
    assert preceding_state(states, 5, 4) == -1
    assert preceding_state(states, 2, 4) == -1


def test_future_span_and_end_censoring_are_kept_separate():
    answer, _, args, _ = fixture()
    states = np.zeros(len(answer.response_ids)+1, dtype=int)
    positions = position_rows(answer, states, args.window).set_index('target')
    assert positions.loc[10, 'future_span'] and positions.loc[10, 'next_gap'] == 4
    assert positions.loc[14, 'at_span_onset'] and not positions.loc[14, 'future_span']
    assert not positions.iloc[-1].full_followup


def test_labels_do_not_select_nodes_and_full_outputs_contain_exact_token_sources(tmp_path):
    from experiments.unsupervised_token_graph.span_audit.onset_report import summarize, write_review

    root = tmp_path
    tmp_path = root / 'samples' / '1'
    answer, channel, args, special = fixture()
    arrays, events, sources = scan_channels([channel], answer.token_ids, answer.prompt_length, special, args)
    original = arrays['node_states'].copy()
    text = ['BOS', 'evidence', 'header'] + list(answer.text)
    nodes = save_detection(tmp_path, arrays, events, sources, answer.token_ids, text, answer.prompt_length)
    save_associations(answer, arrays, nodes, tmp_path, args)
    saved = pd.read_csv(tmp_path / 'nodes.csv')
    assert saved.node_text.eq('a').all()
    assert saved['query'].eq(saved.target+answer.prompt_length-1).all()
    assert pd.read_csv(tmp_path / 'source_reads.csv').source_text.eq('evidence').all()
    links = pd.read_csv(tmp_path / 'node_span_links.csv')
    assert len(links) and links.onset.eq(14).all()
    assert summarize(root, .1)['spans'] == 1
    write_review(root)
    assert (root / 'review.html').exists()
    answer.spans = []
    answer.error_mask[:] = False
    changed, _, _ = scan_channels([channel], answer.token_ids, answer.prompt_length, special, args)
    np.testing.assert_equal(changed['node_states'], original)
    assert span_rows(answer, original[1], args.window).empty


def test_score_onset_can_be_missed_then_detected_later():
    row = describe(np.array([.1]), np.array([.2, .3, .9, .9]), .8, 8)
    assert row['before_alarm'] is None
    assert not row['onset_alarm'] and row['first4_alarm']
    assert row['first_alarm_offset'] == 2


def test_entirely_normal_answer_with_no_switch_is_retained_as_control(tmp_path):
    from experiments.unsupervised_token_graph.span_audit.onset_report import summarize, write_review

    answer, channel, args, special = fixture(switch_query=1000)
    answer.spans = []
    answer.error_mask[:] = False
    arrays, events, sources = scan_channels([channel], answer.token_ids, answer.prompt_length, special, args)
    destination = tmp_path / 'samples' / '1'
    nodes = save_detection(destination, arrays, events, sources, answer.token_ids,
                           ['p']*answer.prompt_length+list(answer.text), answer.prompt_length)
    save_associations(answer, arrays, nodes, destination, args)
    summary = summarize(tmp_path, .1)
    assert summary['nodes'] == 0 and summary['spans'] == 0 and summary['answers'] == 1
    forward = pd.read_csv(tmp_path / 'forward_summary.csv')
    assert set(forward.state) == {0} and forward.future_spans.sum() == 0
    write_review(tmp_path)
