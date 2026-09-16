"""接口/学习/区间推断的实际软件测试；这些构造例不是自然检测成绩。"""

from dataclasses import replace
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
import torch

from experiments.unsupervised_token_graph.offline_span.data import (
    load_samples, load_observations, load_graph, save_graph)
from experiments.unsupervised_token_graph.offline_span.graph import build_token_graph
from experiments.unsupervised_token_graph.offline_span.regions import build_views, propose_spans
from experiments.unsupervised_token_graph.offline_span.controls import build_contrastive_pairs
from experiments.unsupervised_token_graph.offline_span.model import EvidenceSpanScorer
from experiments.unsupervised_token_graph.offline_span.learning import contrastive_loss, partition_sources
from experiments.unsupervised_token_graph.offline_span.detection import interval_inference, fit_unlabeled_reference
from experiments.unsupervised_token_graph.offline_span.evaluation import match_spans, evaluate_saved_predictions
from experiments.unsupervised_token_graph.offline_span.run import main


torch.set_num_threads(1)


def make_attention(seed=3, prompt=8, response=8):
    random = np.random.default_rng(seed)
    count = prompt + response
    attention = np.zeros((2, 2, response, count), np.float32)
    for layer in range(2):
        for head in range(2):
            for index, query in enumerate(range(prompt, count)):
                attention[layer, head, index, :query + 1] = random.uniform(.1, 1, query + 1)
                attention[layer, head, index] *= .85 / attention[layer, head, index].sum()
    return attention


def cache_fields(sample_id='1', split='train', seed=3):
    attention = make_attention(seed)
    response = 'abcdefgh'
    return dict(attention=attention, token_ids=np.arange(16) % 8 + 10,
                response_idx=8, query_positions=np.arange(8, 16),
                id=sample_id, source_id='source_' + sample_id, split=split, task='QA', generator='fixture',
                response=response, offsets=np.array([[i, i + 1] for i in range(8)]),
                source_groups=np.repeat([0, 1], 4), entropy=np.linspace(.5, 2, 8),
                labels=np.array([{'NEVER_READ': True}], dtype=object))


def save_cache(path, layout='dense', **kwargs):
    fields = cache_fields(**kwargs)
    attention = fields.pop('attention')
    if layout == 'dense':
        fields['attention'] = attention
    elif layout == 'data':
        fields['data'] = attention
    elif layout == 'canonical':
        fields.pop('query_positions')
        diagonal = np.zeros((2, 2, 16), np.float32)
        columns, values, pointers = [], [], [0]
        for layer in range(2):
            for head in range(2):
                for row, query in enumerate(range(8, 16)):
                    diagonal[layer, head, query] = attention[layer, head, row, query]
                    keys = np.flatnonzero(attention[layer, head, row])
                    keys = keys[keys != query]
                    columns.extend(keys.tolist())
                    values.extend(attention[layer, head, row, keys].tolist())
                    pointers.append(len(values))
        fields.update(attention_diagonal=diagonal, response_row_ptr=np.asarray(pointers),
                      response_column_indices=np.asarray(columns), response_values=np.asarray(values))
    else:
        layer, head = layout
        fields.update(adjacency=attention[layer:layer + 1, head:head + 1][0, 0], layer=layer, head=head)
        fields.pop('query_positions')
    np.savez_compressed(path, **fields)


def read_graph(path, **kwargs):
    samples, index = load_samples(path)
    observations = load_observations(samples[0], index, **kwargs)
    return build_token_graph(samples[0], observations, edges_per_partition=0, context_width=4)


@pytest.mark.parametrize('layout', ['dense', 'data', 'canonical'])
def test_supported_formats_match(tmp_path, layout):
    first, second = tmp_path / 'first.npz', tmp_path / 'second.npz'
    save_cache(first, 'dense')
    save_cache(second, layout)
    dense, other = read_graph(first), read_graph(second)
    np.testing.assert_array_equal(dense.channels, other.channels)
    np.testing.assert_array_equal(dense.edges, other.edges)
    np.testing.assert_allclose(dense.weights, other.weights, atol=1e-7)
    np.testing.assert_allclose(dense.masses, other.masses, atol=1e-6)
    assert other.coverage.tolist() == [False] + [True] * 7


def test_per_head_files_join_into_one_answer(tmp_path):
    folder = tmp_path / 'channels'
    folder.mkdir()
    for layer in range(2):
        for head in range(2):
            save_cache(folder / f'l{layer}h{head}.npz', (layer, head))
    samples, index = load_samples(folder)
    assert len(samples) == 1 and len(samples[0].cache_files) == 4
    graph = build_token_graph(samples[0], load_observations(samples[0], index), edges_per_partition=0)
    reference = tmp_path / 'dense.npz'
    save_cache(reference)
    other = read_graph(reference)
    np.testing.assert_array_equal(graph.edges, other.edges)
    np.testing.assert_allclose(graph.weights, other.weights)


def test_no_npz_label_member_is_loaded(tmp_path, monkeypatch):
    path = tmp_path / 'a.npz'
    save_cache(path, 'canonical')
    original = np.lib.npyio.NpzFile.__getitem__
    def guarded(archive, key):
        if key in ('labels', 'hallucination_labels'):
            raise AssertionError('labels were read')
        return original(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', guarded)
    graph = read_graph(path)
    assert graph.sample.response_id == '1'


def test_six_fields_use_existing_index_without_new_metadata(tmp_path):
    full = tmp_path / 'full.npz'
    save_cache(full, 'canonical')
    with np.load(full) as archive:
        names = ('token_ids', 'response_idx', 'attention_diagonal', 'response_row_ptr',
                 'response_column_indices', 'response_values')
        fields = {name: archive[name] for name in names}
        token_ids = archive['token_ids'].tolist()
        offsets = archive['offsets'].tolist()
    full.unlink()
    np.savez(tmp_path / 'attention_1.npz', **fields)
    record = dict(id='1', source_id='source_1', split='train', task='QA', generator='fixture',
                  token_ids=token_ids, response_idx=8, offsets=offsets)
    (tmp_path / 'inputs.jsonl').write_text(json.dumps(record))
    graph = read_graph(tmp_path)
    assert graph.sample.source_id == 'source_1'
    assert graph.sample.offsets.shape == (8, 2)


def test_hidden_mode_is_explicit_and_requires_all_positions(tmp_path):
    path = tmp_path / 'a.npz'
    save_cache(path)
    with pytest.raises(ValueError, match='hidden mode'):
        read_graph(path, feature_mode='hidden')
    fields = cache_fields()
    fields['hidden_states'] = np.arange(2 * 16 * 5).reshape(2, 16, 5).astype(np.float32)
    np.savez(path, **fields)
    graph = read_graph(path, feature_mode='hidden', hidden_layer=1)
    assert graph.feature_mode == 'hidden'
    np.testing.assert_array_equal(graph.node_features, fields['hidden_states'][1])


def test_missing_mass_is_not_locally_renormalized(tmp_path):
    path = tmp_path / 'a.npz'
    save_cache(path)
    samples, index = load_samples(path)
    observations = load_observations(samples[0], index)
    graph = build_token_graph(samples[0], observations, edges_per_partition=1, context_width=4)
    assert np.nanmin(graph.masses[:, :, 2]) > .14
    assert np.nanmax(graph.masses[:, :, 3]) > .1
    for channel in range(4):
        for query in range(8, 16):
            selected = (graph.edges[:, 0] == channel) & (graph.edges[:, 2] == query)
            assert selected.sum() <= 2
            assert graph.weights[selected].sum() < .85


def test_future_context_changes_view_not_native_edges(tmp_path):
    path = tmp_path / 'a.npz'
    save_cache(path)
    graph = read_graph(path)
    before = graph.edges.copy()
    future, _ = build_views(graph, 3, 4)
    prefix, _ = build_views(graph, 3, 0)
    assert any(len(span.later_nodes) for span in future)
    assert all(not len(span.later_nodes) for span in prefix)
    assert all(np.all(span.later_nodes >= graph.sample.prompt_length + span.end) for span in future)
    np.testing.assert_array_equal(before, graph.edges)
    assert all(span.start != 0 for span in future)


def test_graph_roundtrip_and_no_gold_in_output(tmp_path):
    path = tmp_path / 'input.npz'
    save_cache(path)
    graph = read_graph(path)
    saved = tmp_path / 'graph.npz'
    save_graph(graph, saved)
    restored = load_graph(saved)
    np.testing.assert_array_equal(graph.edges, restored.edges)
    with np.load(saved) as archive:
        assert 'labels' not in archive.files


def test_controls_leave_physical_graph_unchanged(tmp_path):
    path = tmp_path / 'a.npz'
    save_cache(path)
    graph = read_graph(path)
    spans, index = build_views(graph, 3, 2)
    edges, weights = graph.edges.copy(), graph.weights.copy()
    pairs = build_contrastive_pairs(graph, spans, index, maximum=20)
    assert pairs
    assert any(pair.control_kind == 'evidence' for pair in pairs)
    for pair in pairs:
        np.testing.assert_array_equal(pair.observed.response_nodes, pair.reconnected.response_nodes)
    np.testing.assert_array_equal(edges, graph.edges)
    np.testing.assert_array_equal(weights, graph.weights)


def test_model_backward_updates_relational_parameters(tmp_path):
    path = tmp_path / 'a.npz'
    save_cache(path)
    graph = read_graph(path)
    spans, index = build_views(graph, 3, 2)
    pairs = build_contrastive_pairs(graph, spans, index, 8)
    model = EvidenceSpanScorer(sorted(set(graph.sample.token_ids)), graph.channels, hidden_size=8)
    encoded = model.encode_graph(graph)
    observed = model.score_encoded(encoded, [pair.observed for pair in pairs])
    changed = model.score_encoded(encoded, [pair.reconnected for pair in pairs])
    loss = contrastive_loss(observed, changed)
    loss.backward()
    assert model.head_embedding.weight.grad.abs().sum() > 0
    assert model.comparison[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(loss)


def test_storage_edge_order_invariant(tmp_path):
    path = tmp_path / 'a.npz'
    save_cache(path)
    graph = read_graph(path)
    spans, _ = build_views(graph, 2, 2)
    model = EvidenceSpanScorer(sorted(set(graph.sample.token_ids)), graph.channels, hidden_size=8).eval()
    with torch.no_grad():
        first = model(graph, spans)
        permuted = replace(graph, edges=graph.edges[::-1].copy(), weights=graph.weights[::-1].copy())
        second = model(permuted, spans)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def brute_force(length, bounds, potentials):
    choices, energies = [], []
    for bits in range(1 << len(bounds)):
        selected = [index for index in range(len(bounds)) if bits >> index & 1]
        intervals = sorted(bounds[selected].tolist())
        if any(a[1] >= b[0] for a, b in zip(intervals, intervals[1:])):
            continue
        choices.append(selected)
        energies.append(sum(potentials[selected]))
    probabilities = np.exp(energies - logsumexp(energies))
    marginal = np.zeros(length)
    for choice, probability in zip(choices, probabilities):
        for index in choice:
            start, end = bounds[index]
            marginal[start:end] += probability
    return marginal, choices[int(np.argmax(energies))]


@pytest.mark.parametrize('seed', range(5))
def test_interval_dp_exactly_matches_enumeration(seed):
    bounds = np.asarray([(a, b) for a in range(4) for b in range(a + 1, min(4, a + 2) + 1)])
    potential = np.random.default_rng(seed).normal(size=len(bounds))
    expected, selected = brute_force(4, bounds, potential)
    actual, chosen, _ = interval_inference(4, bounds, potential)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert chosen.tolist() == selected


def test_empty_map_short_errors_and_extreme_potentials():
    bounds = np.array([[0, 1], [0, 3], [2, 3]])
    scores, chosen, _ = interval_inference(3, bounds, [-1000., -1000., -1000.])
    assert not len(chosen)
    assert np.isfinite(scores).all()
    scores, chosen, _ = interval_inference(3, bounds, [1000., -1000., 900.])
    assert chosen.tolist() == [0, 2]
    np.testing.assert_allclose(scores, [1, 0, 1])


def test_source_split_never_uses_test_and_rejects_overlap():
    records = [dict(source_id=str(i), task='QA', split='train') for i in range(10)]
    records += [dict(source_id='future', task='QA', split='test')]
    split = partition_sources(records)
    assert 'future' not in split and set(split.values()) == {'fit', 'calibration'}
    with pytest.raises(ValueError, match='same source'):
        partition_sources(records + [dict(source_id='0', task='QA', split='test')])


def test_span_matching_is_one_to_one():
    assert len(match_spans(np.array([[0, 4]]), np.array([[0, 2], [2, 4]]))) == 1
    assert len(match_spans(np.array([[0, 2], [2, 4]]), np.array([[0, 4]]))) == 1


def make_dataset(root):
    for split in ('train', 'test'):
        (root / split).mkdir(parents=True)
    labels = []
    for index in range(12):
        split = 'train' if index < 8 else 'test'
        save_cache(root / split / f'attention_{index}.npz', 'canonical', sample_id=str(index), split=split, seed=index)
        labels.append(dict(id=str(index), source_id=f'source_{index}', split=split,
                           response='abcdefgh', model='fixture',
                           labels=[dict(start=2, end=5)] if index % 2 else []))
    annotations = root / 'response.jsonl'
    annotations.write_text('\n'.join(json.dumps(row) for row in labels))
    return annotations


def arguments(root, output):
    return ['--train-cache', str(root / 'train'), '--test-cache', str(root / 'test'), '--output', str(output),
            '--epochs', '1', '--hidden-size', '8', '--context-width', '4', '--max-length', '3',
            '--pair-budget', '8', '--future-budget', '2', '--threads', '1', '--bootstrap', '2',
            '--annotations', str(root / 'response.jsonl')]


def test_end_to_end_all_resume_and_label_isolation(tmp_path, monkeypatch):
    root = tmp_path / 'cache'
    annotations = make_dataset(root)
    output = tmp_path / 'output'
    args = arguments(root, output)
    original_open = Path.open
    def without_labels(path, *args, **kwargs):
        if path.resolve() == annotations.resolve():
            raise AssertionError('annotation file read before evaluation')
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', without_labels)
        for phase in ('prepare', 'fit', 'score'):
            main(args + ['--phase', phase])
    main(args + ['--phase', 'evaluate'])
    report = json.loads((output / 'predictions/evaluation.json').read_text())
    assert report['groups']['ALL']['answers'] == 4
    metric = report['groups']['ALL']['views']['all_error']['offline_span']
    assert metric['eligible_tokens'] == 32 and metric['evaluated_tokens'] == 28
    assert metric['eligible_positives'] == 6
    assert (output / 'model/checkpoint.pt').exists()
    before = {path.name: path.read_bytes() for path in (output / 'predictions/samples').glob('*.npz')}
    main(args + ['--resume'])
    after = {path.name: path.read_bytes() for path in (output / 'predictions/samples').glob('*.npz')}
    assert before == after


def test_test_label_change_does_not_change_saved_scores(tmp_path):
    root = tmp_path / 'cache'
    annotations = make_dataset(root)
    output = tmp_path / 'output'
    main(arguments(root, output))
    files = list((output / 'predictions/samples').glob('*.npz'))
    before = [file.read_bytes() for file in files]
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    for row in rows:
        if row['split'] == 'test':
            row['labels'] = [dict(start=1, end=2)]
    annotations.write_text('\n'.join(json.dumps(row) for row in rows))
    evaluate_saved_predictions(output / 'predictions', annotations, bootstrap=0)
    assert before == [file.read_bytes() for file in files]


def test_compact_legacy_rows_start_before_first_response(tmp_path):
    fields = cache_fields()
    fields.pop('query_positions')
    attention = np.zeros((2, 2, 8, 16), np.float32)
    for row, query in enumerate(range(7, 15)):
        attention[:, :, row, :query + 1] = .8 / (query + 1)
    fields['attention'] = attention
    path = tmp_path / 'compact.npz'
    np.savez(path, **fields)
    graph = read_graph(path)
    assert graph.coverage.all()
    assert graph.edges[:, 2].min() == 7
    assert any(span.start == 0 for span in build_views(graph, 2, 0)[0])


def test_training_checkpoint_equals_uninterrupted(tmp_path):
    from experiments.unsupervised_token_graph.offline_span.learning import train_scorer
    path = tmp_path / 'input.npz'
    save_cache(path)
    graph = read_graph(path)
    saved = tmp_path / 'graph.npz'
    save_graph(graph, saved)
    rows = [dict(file='graph.npz', source_id='source_1')]
    vocabulary = sorted(set(graph.sample.token_ids))
    torch.manual_seed(9)
    first = EvidenceSpanScorer(vocabulary, graph.channels, hidden_size=8)
    initial = {name: value.detach().clone() for name, value in first.state_dict().items()}
    train_scorer(first, tmp_path, rows, epochs=2, max_length=2, pair_budget=4)
    partial = EvidenceSpanScorer(vocabulary, graph.channels, hidden_size=8)
    partial.load_state_dict(initial)
    checkpoint = tmp_path / 'checkpoint.pt'
    train_scorer(partial, tmp_path, rows, epochs=1, max_length=2, pair_budget=4, checkpoint=checkpoint)
    resumed = EvidenceSpanScorer(vocabulary, graph.channels, hidden_size=8)
    train_scorer(resumed, tmp_path, rows, epochs=2, max_length=2, pair_budget=4, checkpoint=checkpoint)
    for name in initial:
        torch.testing.assert_close(first.state_dict()[name], resumed.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize('relation', ['none', 'permuted'])
def test_relation_ablation_has_real_forward_and_gradient(tmp_path, relation):
    path = tmp_path / 'a.npz'
    save_cache(path)
    graph = read_graph(path)
    spans, _ = build_views(graph, 2, 2)
    model = EvidenceSpanScorer(sorted(set(graph.sample.token_ids)), graph.channels, hidden_size=8, relation=relation)
    scores = model(graph, spans)
    scores.square().sum().backward()
    assert torch.isfinite(scores).all()


def test_eval_recovers_offsets_with_exact_tokenizer(tmp_path):
    from experiments.unsupervised_token_graph.evaluation_data import EvaluationBinding
    class Tokenizer:
        all_special_ids = []
        def __call__(self, text, **kwargs):
            return dict(input_ids=[ord(char) for char in text], offset_mapping=[[i, i + 1] for i in range(len(text))])
    binding = EvaluationBinding({}, 'fixture')
    binding.tokenizers['fixture'] = Tokenizer()
    record = dict(id='1', source_id='s', split='test', file='samples/1.npz')
    annotation = dict(id='1', source_id='s', split='test', response='abc', model='fixture')
    arrays = dict(token_ids=np.array([10, 97, 98, 99]), prompt_length=1)
    _, offsets = binding.bind(record, annotation, arrays, {})
    np.testing.assert_array_equal(offsets, [[0, 1], [1, 2], [2, 3]])


def test_layer_order_changes_encoder_but_future_nodes_do_not_rewrite_past(tmp_path):
    path = tmp_path / 'a.npz'
    save_cache(path)
    graph = read_graph(path)
    model = EvidenceSpanScorer(sorted(set(graph.sample.token_ids)), graph.channels, hidden_size=8).eval()
    changed_ids = graph.sample.token_ids.copy()
    changed_ids[14:] = changed_ids[14:][::-1]
    changed_graph = replace(graph, sample=replace(graph.sample, token_ids=changed_ids))
    with torch.no_grad():
        first = model.encode_graph(graph)[1]
        second = model.encode_graph(changed_graph)[1]
    torch.testing.assert_close(first[:14], second[:14], atol=0, rtol=0)


def test_all_three_tasks_end_to_end(tmp_path):
    root = tmp_path / 'cache'
    (root / 'train').mkdir(parents=True)
    (root / 'test').mkdir()
    rows = []
    for task_index, task in enumerate(['QA', 'Summary', 'Data2txt']):
        for index in range(6):
            sample_id = str(task_index * 100 + index)
            split = 'train' if index < 4 else 'test'
            fields = cache_fields(sample_id, split, index)
            fields['task'] = task
            np.savez(root / split / f'attention_{sample_id}.npz', **fields)
            rows.append(dict(id=sample_id, source_id='source_' + sample_id, split=split,
                             response='abcdefgh', labels=[] if index % 2 == 0 else [dict(start=2, end=4)], model='fixture'))
    (root / 'response.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    output = tmp_path / 'out'
    main(arguments(root, output))
    report = json.loads((output / 'predictions/evaluation.json').read_text())
    assert set(report['groups']) == {'ALL', 'QA|fixture', 'Summary|fixture', 'Data2txt|fixture'}
    assert report['groups']['ALL']['answers'] == 6
