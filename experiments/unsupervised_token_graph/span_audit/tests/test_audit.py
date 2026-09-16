"""测试索引、匹配和条件对照；不是自然机制的模拟证明。"""

from dataclasses import replace
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from experiments.unsupervised_token_graph.channels import ChannelGraph
from experiments.unsupervised_token_graph.span_audit.units import Answer, Span, marked_spans, span_mask
from experiments.unsupervised_token_graph.span_audit.matching import match_controls, matched_positions
from experiments.unsupervised_token_graph.span_audit.readings import (
    PredictionRows, endpoint_expectation, lag_groups, reuse_statistics)
from experiments.unsupervised_token_graph.span_audit.measurements import measure_answer, METRICS
from experiments.unsupervised_token_graph.span_audit.statistics import source_effects, source_bootstrap
from experiments.unsupervised_token_graph.span_audit.inputs import AuditInputs
from experiments.unsupervised_token_graph.span_audit.report import summarize
from experiments.unsupervised_token_graph.span_audit.run import main


def answer_fixture():
    text = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz'
    ids = np.r_[1000, 1001, [ord(char) for char in text]]
    offsets = np.array([[index, index + 1] for index in range(len(text))])
    spans = [Span(20, 24)]
    return Answer('1', 'source1', 'QA', 'fixture', 'train', text, ids, 2, offsets,
                  span_mask(len(text), spans), spans, np.full(len(text), np.nan), [])


def channel_fixture(answer, head=0, first_query=2):
    queries = np.arange(first_query, len(answer.token_ids))
    attention = np.zeros((len(queries), len(answer.token_ids)))
    for row, query in enumerate(queries):
        attention[row, 0] = .2
        attention[row, query] = .7
    return ChannelGraph(1, head, queries, csr_matrix(attention), answer.prompt_length)


def test_gold_overlap_is_merged_but_adjacent_annotations_are_not():
    offsets = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 4]])
    annotations = [{'start': 0, 'end': 2}, {'start': 1, 'end': 3}, {'start': 3, 'end': 4}]
    assert marked_spans(offsets, annotations) == [Span(0, 3), Span(3, 4)]


def test_control_has_same_length_and_no_error_and_never_reused():
    answer = answer_fixture()
    answer.spans.append(Span(30, 34))
    answer.error_mask = span_mask(len(answer.response_ids), answer.spans)
    pairs = match_controls(answer, position_limit=.5, window=1)
    assert len(pairs) == 2
    for pair in pairs:
        assert pair.error.length == pair.control.length
        assert not answer.error_mask[pair.control.start:pair.control.end].any()
    assert pairs[0].control != pairs[1].control


def test_matching_does_not_relax_calipers_to_force_a_pair():
    assert match_controls(answer_fixture(), position_limit=0) == []


def test_entropy_is_used_when_available_not_imputed_when_missing():
    answer = answer_fixture()
    missing_pairs = match_controls(answer)
    assert np.isnan(missing_pairs[0].entropy_gap)
    answer.entropy[:] = 0
    answer.entropy[20] = 5
    assert match_controls(answer, entropy_limit=.5) == []


def test_predictor_index_includes_previous_token_self_but_not_current_output():
    answer = answer_fixture()
    rows = PredictionRows(channel_fixture(answer), answer.prompt_length)
    assert rows.read(0) is None
    keys, weights = rows.read(1)
    assert keys.max() == answer.prompt_length
    assert weights[keys == answer.prompt_length] == .7


def test_lag_copy_null_equals_exact_permutation_expectation():
    answer = answer_fixture()
    position = 8
    answer.token_ids[2:10] = [7, 8, 7, 8, 7, 8, 7, 8]
    answer.token_ids[10] = 7
    keys = np.arange(2, 10)
    weights = np.arange(1, 9) / 50
    span = Span(1, 5)
    expected, _, _ = endpoint_expectation(answer, position, keys, weights, span)

    history = np.arange(position)
    copied = answer.response_ids[:position] == answer.response_ids[position]
    strata = 2 * lag_groups(position - history) + copied
    groups = [history[strata == group] for group in np.unique(strata)]
    observed = []
    for choices in product(*[list(permutations(group)) for group in groups]):
        changed = weights.copy()
        for group, permutation in zip(groups, choices):
            changed[group] = weights[list(permutation)]
        observed.append(changed[span.start:span.end].sum())
    assert expected == pytest.approx(np.mean(observed))


def test_large_local_mass_is_not_automatically_an_excess():
    answer = answer_fixture()
    position = 22
    keys = np.array([0, answer.prompt_length + position - 1])
    result = reuse_statistics(answer, position, keys, np.array([.2, .7]), Span(20, 24))
    np.testing.assert_allclose(result[:4], [.7, .7, 0, 0])


def test_dependence_is_not_reset_at_gold_end():
    answer = answer_fixture()
    position = 24
    keys = np.array([answer.prompt_length + 23])
    observed = reuse_statistics(answer, position, keys, np.array([.9]), Span(20, 24))
    assert observed[0] == pytest.approx(.9)


def test_boundary_observations_stop_before_the_next_error():
    answer = answer_fixture()
    pair = match_controls(answer)[0]
    answer.error_mask[26] = True
    positions = matched_positions(answer, pair, 'after', 8)
    assert all(left < 26 for left, _ in positions)


def test_measurements_keep_heads_and_use_common_prediction_rows():
    answer = answer_fixture()
    pairs = match_controls(answer)
    channels = [channel_fixture(answer, head=7), channel_fixture(answer, head=9)]
    identities, values, counts = measure_answer(answer, pairs, iter(channels), 3)
    assert identities.tolist() == [[1, 7], [1, 9]]
    assert values.shape == (2, 1, 2, len(METRICS))
    assert counts.shape == (2, 1, 4)
    excess = METRICS.index('inside_endpoint_excess')
    np.testing.assert_allclose(values[:, :, :, excess], 0)


def test_source_weight_is_fixed_not_one_vote_per_token_or_head():
    effects = np.array([[[2.]], [[4.]], [[9.]]])
    means = source_effects(['A', 'A', 'B'], effects)
    np.testing.assert_allclose(means[:, 0, 0], [3., 9.])
    estimate, _, _, count = source_bootstrap(means, 0)
    assert estimate[0, 0] == 6 and count[0, 0] == 2


def test_one_source_does_not_produce_a_confidence_interval():
    values = np.array([[[.1, np.nan]]])
    mean, low, high, count = source_bootstrap(values, 20)
    assert mean[0, 0] == .1
    assert np.isnan(low).all() and np.isnan(high).all()
    assert count.tolist() == [[1, 0]]


class CharacterTokenizer:
    all_special_ids = []

    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(char) for char in text],
                    offset_mapping=[[index, index + 1] for index in range(len(text))])


def write_dataset(root, layout='dense', task='QA', metadata=True):
    cache = root / 'cache'
    dataset = root / 'dataset'
    cache.mkdir(parents=True)
    dataset.mkdir()
    answer = answer_fixture()
    channel = channel_fixture(answer)
    dense = channel.attention.toarray()[None, None]
    fields = dict(token_ids=answer.token_ids, response_idx=answer.prompt_length)
    if metadata:
        fields.update(id='1', source_id='source1', split='train', task=task,
                      generator='fixture', response=answer.text, offsets=answer.offsets)
    if layout == 'dense':
        fields.update(attention=dense, query_positions=channel.queries)
    elif layout == 'canonical':
        diagonal = np.zeros((1, 1, len(answer.token_ids)))
        values, columns, pointer = [], [], [0]
        for index, query in enumerate(channel.queries):
            diagonal[0, 0, query] = dense[0, 0, index, query]
            keys = np.flatnonzero(dense[0, 0, index])
            keys = keys[keys != query]
            columns.extend(keys.tolist())
            values.extend(dense[0, 0, index, keys].tolist())
            pointer.append(len(values))
        fields.update(attention_diagonal=diagonal, response_row_ptr=pointer,
                      response_column_indices=columns, response_values=values)
    else:
        fields.update(adjacency=dense[0, 0], layer=1, head=3)
    fields['labels'] = np.array([{'poisoned': True}], dtype=object)
    np.savez(cache / 'attention_1.npz', **fields)
    gold = dict(id='1', source_id='source1', split='train', response=answer.text,
                model='fixture', labels=[dict(start=20, end=24)])
    (dataset / 'response.jsonl').write_text(json.dumps(gold) + '\n')
    (dataset / 'source_info.json').write_text(json.dumps([dict(source_id='source1', task_type=task)]))
    return cache, dataset


@pytest.mark.parametrize('layout', ['dense', 'canonical', 'head'])
def test_actual_parent_adapters_all_formats(tmp_path, layout):
    cache, dataset = write_dataset(tmp_path, layout)
    inputs = AuditInputs(cache, dataset)
    answer = inputs.load_answer('1')
    pairs = match_controls(answer)
    channels, values, _ = measure_answer(answer, pairs, inputs.channels(answer), 3)
    assert answer.source_id == 'source1'
    assert channels.shape == (1, 2)
    assert values.shape[1] == 1


def test_bare_cache_gets_metadata_and_verified_offsets_without_loading_npz_labels(tmp_path):
    cache, dataset = write_dataset(tmp_path, 'canonical', metadata=False)
    inputs = AuditInputs(cache, dataset, tokenizer='fixture')
    inputs.binding.tokenizers['fixture'] = CharacterTokenizer()
    answer = inputs.load_answer('1')
    assert answer.task == 'QA' and answer.source_id == 'source1'
    assert answer.offsets.shape == (52, 2)
    assert list(inputs.selected_ids('train', [])) == ['1']


@pytest.mark.parametrize('task', ['QA', 'Summary', 'Data2txt'])
def test_cli_saves_readable_pairs_and_real_effects_with_resume(tmp_path, task):
    cache, dataset = write_dataset(tmp_path, task=task)
    output = tmp_path / 'out'
    args = ['--cache', str(cache), '--dataset', str(dataset), '--output', str(output),
            '--window', '3', '--bootstrap', '2']
    main(args)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['matched_pairs'] == 1
    assert summary['gold_spans'] == 1
    assert summary['entropy_available_answers'] == 0
    saved = (output / 'samples/1.npz').read_bytes()
    main(args + ['--resume'])
    assert (output / 'samples/1.npz').read_bytes() == saved
    pairs = json.loads((output / 'matched_pairs.json').read_text())
    assert pairs[0]['pairs'][0]['error']['text'] == 'uvwx'
    assert (output / 'paired_effects.csv').is_file()


def test_no_match_is_reported_not_manufactured(tmp_path):
    cache, dataset = write_dataset(tmp_path)
    output = tmp_path / 'out'
    main(['--cache', str(cache), '--dataset', str(dataset), '--output', str(output),
          '--position-gap', '0', '--bootstrap', '0'])
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['gold_spans'] == 1 and summary['matched_pairs'] == 0
    assert not (output / 'paired_effects.csv').exists()


def test_default_old_training_entry_does_not_start_an_optimizer():
    from experiments.unsupervised_token_graph.offline_span.run import main as retired
    with pytest.raises(SystemExit, match='training is retired'):
        retired(['--phase', 'all'])


def test_changed_annotation_does_not_silently_resume_old_pairs(tmp_path):
    cache, dataset = write_dataset(tmp_path)
    output = tmp_path / 'out'
    args = ['--cache', str(cache), '--dataset', str(dataset), '--output', str(output)]
    main(args)
    annotation = dataset / 'response.jsonl'
    annotation.write_text(annotation.read_text() + '\n')
    with pytest.raises(ValueError, match='same experiment settings'):
        main(args + ['--resume'])


def test_measuring_no_pairs_does_not_consume_attention_channels():
    answer = answer_fixture()
    identities, values, counts = measure_answer(answer, [], [], 8)
    assert identities.shape == (0, 2)
    assert values.shape == (0, 0, 2, len(METRICS))
    assert counts.shape == (0, 0, 4)
