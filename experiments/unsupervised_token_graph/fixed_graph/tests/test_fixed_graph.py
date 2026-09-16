"""用真实reader验证格式、图算子、无标签参照和冻结评价；不模拟自然效果。"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from experiments.unsupervised_token_graph.offline_span.data import load_samples
from experiments.unsupervised_token_graph.fixed_graph.inputs import extract_sample, sample_channels
from experiments.unsupervised_token_graph.fixed_graph.operators import (
    FEATURES, VARIANTS, channel_arrays, describe_row, graph_blocks, shuffle_history, typed_operators)
from experiments.unsupervised_token_graph.fixed_graph.reference import (
    binary_spans, fit_reference, novelty_distance, smooth_scores, source_quantiles, split_sources)
from experiments.unsupervised_token_graph.fixed_graph.pipeline import list_samples
from experiments.unsupervised_token_graph.fixed_graph.run import main, arguments


SETTINGS = dict(local_window=4, dimensions=32, seed=17)


def write_cache(path, identity='1', task='QA', split='train', layout='canonical', bare=False, seed=1):
    prompt, response = 8, 12
    count = prompt + response
    tokens = np.arange(count, dtype=int) % 13 + 10
    text = 'abcdefghijkl'
    fields = dict(token_ids=tokens, response_idx=prompt,
                  labels=np.array([{'should_never_be_read': True}], dtype=object))
    if not bare:
        fields.update(id=identity, source_id='source_' + identity, task=task, split=split,
                      generator='fixture', response=text,
                      offsets=np.array([[i, i + 1] for i in range(response)]))
    random = np.random.default_rng(seed)
    matrices = np.zeros((2, 2, response, count), np.float32)
    for layer in range(2):
        for head in range(2):
            for row, query in enumerate(range(prompt, count)):
                value = random.uniform(.1, 1., size=query + 1)
                matrices[layer, head, row, :query + 1] = .9 * value / value.sum()
    if layout in ('attention', 'data'):
        fields.update({layout: matrices}, query_positions=np.arange(prompt, count))
    elif isinstance(layout, tuple):
        layer, head = layout
        fields.update(adjacency=matrices[layer, head], layer=layer, head=head)
    else:
        diagonal = np.zeros((2, 2, count), np.float32)
        columns, values, pointer = [], [], [0]
        for layer in range(2):
            for head in range(2):
                for row, query in enumerate(range(prompt, count)):
                    diagonal[layer, head, query] = matrices[layer, head, row, query]
                    columns.extend(range(query))
                    values.extend(matrices[layer, head, row, :query])
                    pointer.append(len(columns))
        fields.update(attention_diagonal=diagonal, response_row_ptr=pointer,
                      response_column_indices=columns, response_values=values)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)


def extract(path):
    samples, index = load_samples(path)
    channels = sample_channels(samples[0], index, None, None)
    return extract_sample(samples[0], channels, SETTINGS)


@pytest.mark.parametrize('layout', ['attention', 'data', 'canonical'])
def test_actual_cache_formats_have_identical_embeddings(tmp_path, layout):
    first, second = tmp_path / 'reference.npz', tmp_path / 'other.npz'
    write_cache(first, layout='attention')
    write_cache(second, layout=layout)
    a, b = extract(first), extract(second)
    for name in VARIANTS:
        np.testing.assert_allclose(a[0][name], b[0][name], atol=1e-7)
    np.testing.assert_array_equal(a[1], [False] + [True] * 11)
    assert a[2].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_per_head_export_groups_and_order_independence(tmp_path):
    folder = tmp_path / 'head_files'
    for layer in range(2):
        for head in range(2):
            write_cache(folder / f'l{layer}h{head}.npz', layout=(layer, head))
    original = tmp_path / 'dense.npz'
    write_cache(original, layout='attention')
    a, b = extract(folder), extract(original)
    for name in VARIANTS:
        np.testing.assert_allclose(a[0][name], b[0][name], atol=1e-7)


def test_label_members_are_never_loaded(tmp_path, monkeypatch):
    path = tmp_path / '1.npz'
    write_cache(path)
    original = np.lib.npyio.NpzFile.__getitem__
    def guarded(archive, key):
        if key in ('labels', 'hallucination_labels'):
            raise AssertionError('label member was accessed')
        return original(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', guarded)
    extract(path)


def test_features_distinguish_dispersion_with_same_partition_mass():
    ids = np.arange(14)
    weights_a = np.array([.2, .2, .2, .2, .2])
    weights_b = np.array([.8, .2])
    a = describe_row(np.array([0, 1, 2, 3, 9]), weights_a, ids, 8, 2, 4)
    b = describe_row(np.array([0, 9]), weights_b, ids, 8, 2, 4)
    np.testing.assert_allclose(a[:3], b[:3])
    assert a[3] > b[3] and a[5] > b[5]
    assert a[-1] < b[-1]


def test_first_token_coverage_uses_prediction_not_emitted_position(tmp_path):
    path = tmp_path / '1.npz'
    write_cache(path)
    samples, index = load_samples(path)
    channel = next(sample_channels(samples[0], index, None, None))
    features, covered, _, history = channel_arrays(channel, samples[0], 4)
    assert not covered[0]
    assert history[1, 0] > 0
    assert history.diagonal().sum() == 0
    assert np.isfinite(features[covered]).all()


def test_two_hop_contains_information_absent_from_one_hop():
    features = np.zeros((4, 1))
    features[0] = 1
    local = sparse.csr_matrix(([1., 1.], ([1, 2], [0, 1])), shape=(4, 4))
    remote = sparse.csr_matrix((4, 4))
    blocks = graph_blocks(features, local, remote)
    assert blocks[1][2, 0] == 0
    assert blocks[4][2, 0] == 1


def test_relation_order_cannot_be_exchanged():
    features = np.eye(4)
    local = sparse.csr_matrix(([1.], ([3], [2])), shape=(4, 4))
    remote = sparse.csr_matrix(([1.], ([2], [0])), shape=(4, 4))
    blocks = graph_blocks(features, local, remote)
    assert blocks[5][3, 0] == 1
    assert blocks[6][3, 0] == 0


def test_later_readers_add_context_without_reversing_native_edges():
    features = np.eye(4)
    local = sparse.csr_matrix(([.7, .8], ([2, 3], [1, 1])), shape=(4, 4))
    before = local.toarray().copy()
    later = graph_blocks(features, local, sparse.csr_matrix((4, 4)))[7]
    np.testing.assert_allclose(later[1], [0, 0, .7 / 1.5, .8 / 1.5])
    np.testing.assert_array_equal(local.toarray(), before)


def row_groups(matrix, target, tokens, covered, local_window):
    begin, end = matrix.indptr[target:target + 2]
    sources, weights = matrix.indices[begin:end], matrix.data[begin:end]
    lag = target - sources
    bands = np.ceil(np.log2(lag)).astype(int)
    strata = 8 * bands + 4 * (lag <= local_window) + 2 * (tokens[sources] == tokens[target]) + covered[sources]
    return {int(group): sorted(weights[strata == group]) for group in np.unique(strata)}


def test_shuffle_preserves_nuisance_statistics_and_changes_endpoints():
    random = np.random.default_rng(7)
    rows, columns, weights = [], [], []
    for target in range(2, 32):
        candidates = random.choice(target, min(5, target), replace=False)
        rows.extend([target] * len(candidates))
        columns.extend(candidates)
        weights.extend(random.uniform(.01, .1, len(candidates)))
    history = sparse.csr_matrix((weights, (rows, columns)), shape=(32, 32))
    tokens = np.arange(32) % 5
    coverage = np.r_[False, np.ones(31, bool)]
    original = history.toarray().copy()
    changed, moved = shuffle_history(history, tokens, coverage, 7, np.random.default_rng(3))
    for target in range(32):
        assert row_groups(history, target, tokens, coverage, 7) == row_groups(changed, target, tokens, coverage, 7)
    assert moved > 0 and not np.array_equal(changed.toarray(), original)
    np.testing.assert_array_equal(history.toarray(), original)
    _, _, dropped_a = typed_operators(history, coverage, 7)
    _, _, dropped_b = typed_operators(changed, coverage, 7)
    assert dropped_a == pytest.approx(dropped_b)


def test_missing_attributes_are_not_promoted_to_normalized_messages():
    history = sparse.csr_matrix(([.7, .1], ([2, 2], [0, 1])), shape=(3, 3))
    local, remote, dropped = typed_operators(history, np.array([False, True, True]), 4)
    assert local.sum() == pytest.approx(.1)
    assert dropped == pytest.approx(.7)


def test_knn_distance_increases_for_distant_query():
    reference = fit_reference(np.array([[0., 0.], [1., 1.], [2., 2.]]), 2)
    near = novelty_distance(np.array([[1., 1.]]), reference)[0]
    far = novelty_distance(np.array([[8., 8.]]), reference)[0]
    assert far > near
    assert len(reference['bank']) == 3


def test_source_quantiles_give_equal_source_weight():
    first = source_quantiles(np.array([0., 10.]), ['a', 'b'], [.5, .75])
    repeated = source_quantiles(np.r_[np.zeros(100), 10.], ['a'] * 100 + ['b'], [.5, .75])
    np.testing.assert_allclose(first, repeated)


def test_partition_ignores_test_and_keeps_source_together():
    rows = [dict(source_id=str(i), split='train', task='QA') for i in range(10)]
    rows.extend([dict(source_id='outside', split='test', task='QA')])
    roles = split_sources(rows, 17)
    assert 'outside' not in roles and len(roles) == 10
    with pytest.raises(ValueError, match='overlapping'):
        split_sources(rows + [dict(source_id='1', split='test', task='QA')], 17)


def test_smoothing_and_spans_allow_missing_and_long_runs():
    scores = smooth_scores(np.array([np.nan, 2., 4., 6.]), 1)
    assert np.isnan(scores[0])
    np.testing.assert_allclose(scores[1:], [3., 4., 5.])
    spans = binary_spans(np.r_[True, False, np.ones(40, bool), False])
    np.testing.assert_array_equal(spans, [[0, 1], [2, 42]])


def make_population(root, bare=False):
    annotations, sources = [], []
    for task_index, task in enumerate(('QA', 'Summary', 'Data2txt')):
        for index in range(8):
            identity = str(task_index * 100 + index)
            split = 'train' if index < 6 else 'test'
            path = root / split / f'attention_{identity}.npz'
            write_cache(path, identity, task, split, bare=bare, seed=index + task_index)
            annotations.append(dict(id=identity, source_id='source_' + identity, split=split,
                                    model='fixture', response='abcdefghijkl',
                                    labels=[] if index % 2 == 0 else [dict(start=3, end=6)]))
            sources.append(dict(source_id='source_' + identity, task_type=task))
    dataset = root / 'dataset'
    dataset.mkdir()
    (dataset / 'response.jsonl').write_text('\n'.join(map(json.dumps, annotations)))
    (dataset / 'source_info.json').write_text(json.dumps(sources))
    return dataset


def command(root, dataset, output):
    return ['--train-cache', str(root / 'train'), '--test-cache', str(root / 'test'),
            '--dataset', str(dataset), '--output', str(output), '--dimensions', '16',
            '--bank-size', '32', '--tokens-per-source', '4', '--neighbors', '2',
            '--local-window', '4', '--bootstrap', '2', '--threads', '1']


def test_bare_cache_metadata_without_loading_offsets_or_labels(tmp_path):
    dataset = make_population(tmp_path, bare=True)
    args = arguments(command(tmp_path, dataset, tmp_path / 'out'))
    samples, _ = list_samples(args)
    assert len(samples) == 24
    assert all(sample.source_id and sample.task != 'unknown' for sample in samples)
    assert all(len(sample.offsets) == 0 for sample in samples)


def test_all_tasks_all_variants_end_to_end_and_resume(tmp_path, monkeypatch):
    dataset = make_population(tmp_path)
    output = tmp_path / 'out'
    args = command(tmp_path, dataset, output)
    source_files = list((tmp_path / 'train').glob('*.npz')) + list((tmp_path / 'test').glob('*.npz'))
    before = [path.read_bytes() for path in source_files]
    original = Path.open
    def block_annotations(path, *a, **kw):
        if path == dataset / 'response.jsonl':
            raise AssertionError('scoring opened annotations despite embedded metadata')
        return original(path, *a, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', block_annotations)
        for phase in ('prepare', 'fit', 'score'):
            main(args + ['--phase', phase, '--resume'])
    main(args + ['--phase', 'evaluate'])
    evaluation = json.loads((output / 'predictions/evaluation.json').read_text())
    assert set(evaluation['groups']) == {'ALL', 'QA|fixture', 'Summary|fixture', 'Data2txt|fixture'}
    metrics = evaluation['groups']['ALL']['views']['all_error']
    assert metrics['graph']['evaluated_tokens'] == 66
    assert metrics['graph']['eligible_tokens'] == 72
    frozen = {path.name: path.read_bytes() for path in (output / 'predictions').glob('*.npz')}
    main(args + ['--resume'])
    assert frozen == {path.name: path.read_bytes() for path in (output / 'predictions').glob('*.npz')}
    assert before == [path.read_bytes() for path in source_files]


def test_label_change_leaves_features_references_and_scores_identical(tmp_path):
    dataset = make_population(tmp_path)
    args = command(tmp_path, dataset, tmp_path / 'out')
    main(args)
    root = tmp_path / 'out'
    paths = list(root.rglob('*.npz'))
    before = [path.read_bytes() for path in paths]
    labels = dataset / 'response.jsonl'
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    for row in rows:
        row['labels'] = []
    labels.write_text('\n'.join(map(json.dumps, rows)))
    main(args + ['--phase', 'evaluate'])
    assert before == [path.read_bytes() for path in paths]


def test_different_parameters_cannot_resume_old_features(tmp_path):
    dataset = make_population(tmp_path)
    args = command(tmp_path, dataset, tmp_path / 'out')
    main(args + ['--phase', 'prepare'])
    with pytest.raises(ValueError, match='identical'):
        main(args + ['--phase', 'prepare', '--resume', '--dimensions', '24'])


def test_changed_attention_cannot_silently_resume(tmp_path):
    dataset = make_population(tmp_path)
    args = command(tmp_path, dataset, tmp_path / 'out')
    main(args + ['--phase', 'prepare'])
    path = tmp_path / 'train/attention_0.npz'
    write_cache(path, identity='0', seed=555)
    with pytest.raises(ValueError, match='roster changed'):
        main(args + ['--phase', 'prepare', '--resume'])


def test_uniform_shuffle_is_reported_as_zero_actual_change():
    history = sparse.csr_matrix(np.tril(np.ones((12, 12)), -1) / 12)
    tokens = np.arange(12)
    covered = np.ones(12, bool)
    shuffled, changed = shuffle_history(history, tokens, covered, 4, np.random.default_rng(1))
    np.testing.assert_array_equal(shuffled.toarray(), history.toarray())
    assert changed == 0


def bank_arrays(output):
    return {str(path.relative_to(output)): {name: values.copy() for name, values in np.load(path).items()}
            for path in (output / 'reference').glob('*/bank.npz')}


def test_bare_metadata_label_change_cannot_change_unsupervised_reference(tmp_path):
    dataset = make_population(tmp_path, bare=True)
    first, second = tmp_path / 'first', tmp_path / 'second'
    for phase in ('prepare', 'fit', 'score'):
        main(command(tmp_path, dataset, first) + ['--phase', phase, '--resume'])
    annotations = dataset / 'response.jsonl'
    rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    for row in rows:
        row['labels'] = [dict(start=0, end=12)]
    annotations.write_text('\n'.join(map(json.dumps, rows)))
    for phase in ('prepare', 'fit', 'score'):
        main(command(tmp_path, dataset, second) + ['--phase', phase, '--resume'])
    a, b = bank_arrays(first), bank_arrays(second)
    for path in a:
        for field in a[path]:
            np.testing.assert_array_equal(a[path][field], b[path][field])
    for path in (first / 'predictions').glob('*.npz'):
        with np.load(path) as a, np.load(second / 'predictions' / path.name) as b:
            for method in VARIANTS:
                np.testing.assert_array_equal(a[method], b[method])


def test_changed_test_observations_do_not_change_reference_bank(tmp_path):
    dataset = make_population(tmp_path)
    first, second = tmp_path / 'first', tmp_path / 'second'
    for phase in ('prepare', 'fit'):
        main(command(tmp_path, dataset, first) + ['--phase', phase, '--resume'])
    write_cache(tmp_path / 'test/attention_6.npz', identity='6', split='test', seed=100)
    for phase in ('prepare', 'fit'):
        main(command(tmp_path, dataset, second) + ['--phase', phase, '--resume'])
    a, b = bank_arrays(first), bank_arrays(second)
    for path in a:
        for field in a[path]:
            np.testing.assert_array_equal(a[path][field], b[path][field])
