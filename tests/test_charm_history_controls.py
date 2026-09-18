"""Boundary-aligned historical controls and actual original-network loss training."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from experiments.charm_structure_audit.history_math import SCORE_NAMES, score_controls, score_table
from experiments.charm_structure_audit.continuity_history import (
    analyze_history, calibrated_thresholds, main, ranking, read_calibration,
)
from experiments.charm_structure_audit.continuity import growth_rows, transition_tables, run_continuity
from experiments.charm_structure_audit.continuity_weights import loss_weights, prepare_weights, SCHEMES
from experiments.charm_structure_audit.continuity_train import (
    fit_model, load_input, node_logits, run_training, score_record,
)
from experiments.charm_structure_audit.data import read_json, original_parts, write_json
from experiments.charm_structure_audit.model import CHARM


torch.set_num_threads(1)


def network():
    torch.manual_seed(5)
    return CHARM(4, 4, dict(hidden_dim=8, gnn_layers=2, residual_mp=True), edge_chunk=16).eval()


def test_history_never_reads_current_future_or_labels():
    original = np.array([.1, .2, .3, .4, .5, .6])
    altered = original.copy()
    altered[3:] = [.99, .02, .85]
    first, second = score_controls(original, 2), score_controls(altered, 2)
    for name in ('previous', 'past_mean', 'past_ewma', 'prefix_mean', 'sampled_past'):
        np.testing.assert_allclose(first[name][:4], second[name][:4], equal_nan=True)
    assert first['current'][3] != second['current'][3]
    assert first['causal_ewma'][3] != second['causal_ewma'][3]


def test_first_token_is_missing_only_for_history_views():
    scores = score_controls([.2, .8, .9])
    for name in SCORE_NAMES:
        assert np.isnan(scores[name][0]) == (name not in ('current', 'causal_ewma'))
    np.testing.assert_allclose(scores['past_mean'][1:], [.2, .5])
    np.testing.assert_allclose(scores['current'][1:],
                               scores['past_mean'][1:] + scores['current_increment'][1:])


def test_inclusive_ewma_is_not_past_only():
    scores = score_controls([.1, .9, .9], beta=.5)
    np.testing.assert_allclose(scores['causal_ewma'], [.1, .5, .7])
    np.testing.assert_allclose(scores['past_ewma'][1:], [.1, .5])


def test_one_spike_cannot_create_permanently_rising_smoothing():
    scores = score_controls([.1, .9, .1, .1, .1, .1])
    assert np.all(np.diff(scores['causal_ewma'][1:]) < 0)
    sustained = score_controls([.1, .9, .9, .9, .9])
    assert np.all(np.diff(sustained['causal_ewma']) > 0)


def test_no_cross_answer_history_and_label_change_is_irrelevant():
    table = pd.DataFrame(dict(id=['a', 'a', 'b', 'b'], token=[0, 1, 0, 1],
                              source_id=['s', 's', 't', 't'], score=[.9, .8, .1, .2], gold=[1, 1, 0, 0]))
    a = score_table(table)
    changed = table.copy()
    changed['gold'] = 1 - changed.gold
    b = score_table(changed)
    np.testing.assert_allclose(a[list(SCORE_NAMES)], b[list(SCORE_NAMES)], equal_nan=True)
    assert np.isnan(a.loc[2, 'past_mean'])
    assert a.loc[3, 'past_mean'] == .1


def test_short_prefix_sample_agrees_with_all_past():
    scores = score_controls(np.linspace(.1, .9, 8), window=10)
    np.testing.assert_allclose(scores['sampled_past'], scores['prefix_mean'], equal_nan=True)
    np.testing.assert_allclose(scores['past_mean'], scores['prefix_mean'], equal_nan=True)


def test_empty_and_single_class_rankings_are_not_fake_auc():
    assert ranking([0, 0], [.1, .2])['auroc'] is None
    assert ranking([1, 1], [.1, .2])['auroc'] is None
    assert ranking([], [])['tokens'] == 0
    assert all(len(v) == 0 for v in score_controls([]).values())


def fixture_table():
    labels = np.r_[np.zeros(8), np.ones(6), np.zeros(10)].astype(int)
    scores = np.r_[np.repeat(.15, 8), [.45, .55, .65, .75, .8, .9], np.repeat(.15, 10)]
    table = pd.DataFrame(dict(id='test', source_id='test_source', token=np.arange(24),
                              text='x', score=scores, gold=labels, predicted=scores > .7))
    spans = pd.DataFrame([dict(id='test', source_id='test_source', start=8, end=14)])
    pairs = [dict(tier='cluster', id='test', source_id='test_source', error_start=8, normal_start=1, length=6)]
    return table, spans, pairs


def test_reports_have_common_support_and_do_not_reset_at_exits(tmp_path):
    table, spans, pairs = fixture_table()
    output = analyze_history(table, spans, .7, pairs, tmp_path, bootstrap=5)
    metrics = output['metrics'].query('population == "all"')
    assert set(metrics.tokens) == {23}
    assert set(metrics.positives) == {6}
    scores = pd.read_csv(tmp_path / 'tokens.csv.gz')
    assert scores.loc[14, 'past_mean'] > scores.loc[14, 'current']
    assert np.array_equal(scores.score, table.score)
    exits = output['exits']
    assert set(exits.exit_token) == {14}
    assert exits[exits.method != 'current'].alarm.isna().all()
    assert output['matched_pairs'].query('region == "all"').tokens_each.eq(6).all()


def test_calibration_not_test_fits_transformed_thresholds(tmp_path):
    table, spans, pairs = fixture_table()
    calibration = table.assign(id='cal', source_id='cal_source')
    first = calibrated_thresholds(calibration, .7, 10, .5, 17, .05)
    assert first['current'] == .7
    assert all(value is not None for value in first.values())
    changed = table.copy()
    changed.gold = 1 - changed.gold
    second = calibrated_thresholds(calibration, .7, 10, .5, 17, .05)
    assert first == second


def test_calibration_loader_rejects_test_sources(tmp_path):
    table, _, _ = fixture_table()
    (tmp_path / 'calibration').mkdir()
    table.to_csv(tmp_path / 'calibration/tokens.csv', index=False)
    write_json(tmp_path / 'training.json', dict(partitions=dict(calibration=['test'])))
    with pytest.raises(ValueError, match='held-out'):
        read_calibration(tmp_path, table)


def test_pointwise_order_does_not_encode_label_adjacency():
    model = network().double().eval()
    other = copy.deepcopy(model)
    values = torch.rand(16, 4, dtype=torch.double)
    labels = torch.tensor([0.] * 4 + [1.] * 8 + [0.] * 4, dtype=torch.double)
    permutation = torch.randperm(len(labels))
    original, altered = node_logits(model, values), node_logits(other, values[permutation])
    torch.testing.assert_close(original[permutation], altered)
    left = F.binary_cross_entropy_with_logits(original, labels, reduction='sum')
    right = F.binary_cross_entropy_with_logits(altered, labels[permutation], reduction='sum')
    left.backward()
    right.backward()
    for a, b in zip(model.parameters(), other.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-10)
    assert all(p.grad is None for layer in model.mp_layers for p in layer.msg_mlp.parameters())


@pytest.mark.parametrize('scheme', SCHEMES)
def test_weight_mass_and_labels_unchanged(scheme):
    labels = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0], bool)
    spans = [[1, 3], [4, 8]]
    before = labels.copy()
    weights = loss_weights(labels, spans, scheme, mean_span_length=3, seed=7)
    np.testing.assert_array_equal(labels, before)
    np.testing.assert_allclose(weights[labels].sum(), 6)
    np.testing.assert_allclose(weights[~labels], 1)


def test_onset_and_random_same_multiset_and_span_equal_different_objective():
    labels = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0], bool)
    spans = [[1, 3], [4, 8]]
    onset = loss_weights(labels, spans, 'onset_half', 3, 7)
    random = loss_weights(labels, spans, 'random_onset_half', 3, 7)
    equal = loss_weights(labels, spans, 'span_equal', 3, 7)
    for start, end in spans:
        np.testing.assert_allclose(np.sort(onset[start:end]), np.sort(random[start:end]))
        assert equal[start:end].sum() == pytest.approx(3)


def test_previous_label_only_has_conditional_auc_half():
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1])
    table = pd.DataFrame(dict(id='a', source_id='s', token=np.arange(len(labels)), gold=labels,
                              score=np.r_[0, labels[:-1]]))
    _, conditional, _, _ = transition_tables(table, .5, 0)
    np.testing.assert_allclose(conditional.auroc, .5)


def test_growth_compares_normal_partner_not_only_error_curve():
    table = pd.DataFrame(dict(id='a', source_id='s', token=np.arange(8), gold=[0]*4+[1]*4,
                              score=[.2, .3, .4, .5, .2, .3, .4, .5]))
    pair = dict(id='a', source_id='s', error_start=4, normal_start=0, length=4, tier='cluster')
    result = growth_rows(table, [pair])
    assert result.gap_growth.iloc[0] == 0
    assert result.auc_growth.iloc[0] == 0


def prepared_fixture(root):
    prepared, model_root = root / 'prepared', root / 'root'
    recipe = dict(seed=0, epochs=2, batch_size=2, learning_rate=.001,
                  hidden_dim=8, gnn_layers=2, fpr=.05, partitions={})
    records = []
    model = network()
    for number, part in enumerate(('fit', 'select', 'calibration', 'test')):
        table, spans, pairs = fixture_table()
        identity, source = part, part + '_source'
        split = 'test' if part == 'test' else 'train'
        random = np.random.default_rng(number)
        nodes = random.random((26, 4)).astype(np.float32)
        nodes[2:, 0] += table.gold.to_numpy() * .3
        edge_index = np.stack((np.arange(25), np.arange(1, 26)))
        graph = dict(x=nodes, edge_index=edge_index, edge_attr=np.ones((25, 4), np.float32) / 4,
                     edge_mark=np.zeros((25, 2), np.float32), prompt_length=np.array(2), layers=2, heads=2)
        record = dict(id=identity, source_id=source, task='QA', generator='fixture', split=split,
                      response_tokens=24, positives=6)
        file = prepared / 'graphs' / split / (identity + '.npz')
        file.parent.mkdir(parents=True, exist_ok=True)
        onset = np.arange(24) == 8
        np.savez(file, **graph, gold=table.gold.to_numpy().astype(bool), onset=onset,
                 spans=np.array([[8, 14]]), offsets=np.array([[i, i + 1] for i in range(24)]),
                 response=np.array('x' * 24), token_ids=np.arange(26), record_json=json.dumps(record))
        records.append(record)
        recipe['partitions'][part] = [identity]
    write_json(prepared / 'index.json', records)
    for name in ('node_only', 'charm_in'):
        folder = model_root / name
        write_json(folder / 'training.json', recipe)
        write_json(folder / 'threshold.json', dict(value=.7))
        table, spans, pairs = fixture_table()
        (folder / 'test').mkdir()
        table.to_csv(folder / 'test/tokens.csv', index=False)
        spans.to_csv(folder / 'test/spans.csv', index=False)
    write_json(model_root / 'charm_in/test/cluster_audit/pairs.json', pairs)
    return prepared, model_root, recipe, pairs


def arguments(prepared, root):
    return SimpleNamespace(prepared=str(prepared), root=str(root), models=['node_only'],
                           device='cpu', edge_chunk=16, epochs=2, bootstrap=0,
                           continuity_seeds=[0], continuity_schemes=list(SCHEMES), continuity_stage='train')


def test_actual_saved_score_cli_and_existing_observe_dispatch(tmp_path):
    prepared, root, recipe, pairs = prepared_fixture(tmp_path)
    output = tmp_path / 'history'
    main(['--root', str(root), '--output', str(output), '--bootstrap', '0'])
    assert (output / 'node_only/metrics.csv').is_file()
    args = arguments(prepared, root)
    args.continuity_stage = 'observe'
    target = tmp_path / 'observe'
    target.mkdir()
    run_continuity(args, target, pairs)
    assert (target / 'node_only/history_controls/transitions.csv').is_file()
    assert (target / 'continuity_review.tar.gz').is_file()


def test_node_loader_and_model_match_original_no_graph(tmp_path):
    prepared, root, recipe, _ = prepared_fixture(tmp_path)
    record = read_json(prepared / 'index.json')[0]
    values, _ = load_input(record, prepared, 'node_only')
    graph, _ = load_input(record, prepared, 'charm_in')
    model = network()
    torch.testing.assert_close(score_record(model, values, 'node_only'), model(graph, ablation='no_graph')[2:])


def test_all_weight_schemes_train_and_evaluate_without_modifying_inputs(tmp_path):
    prepared, root, recipe, pairs = prepared_fixture(tmp_path)
    before = {path: path.read_bytes() for path in prepared.rglob('*.npz')}
    args = arguments(prepared, root)
    output = tmp_path / 'trained'
    output.mkdir()
    run_training(args, output, pairs)
    assert (output / 'onset_vs_random.csv').is_file()
    assert (output / 'growth_deltas.csv').is_file()
    for scheme in SCHEMES:
        run = output / 'node_only' / scheme / 'seed_0'
        assert (run / 'epoch_state.pt').is_file()
        assert (run / 'calibration/tokens.csv').is_file()
        protocol = read_json(run / 'history_controls/protocol.json')
        assert all(value is not None for value in protocol['thresholds'].values())
        assert len(read_json(run / 'history.json')) == 2
    assert all(path.read_bytes() == content for path, content in before.items())
    checkpoint = output / 'node_only/token/seed_0/checkpoint.pt'
    first = checkpoint.read_bytes()
    run_training(args, output, pairs)
    assert checkpoint.read_bytes() == first


def test_epoch_resume_matches_uninterrupted_dropout_training(tmp_path, monkeypatch):
    from experiments.charm_structure_audit import continuity_train as training
    prepared, root, recipe, _ = prepared_fixture(tmp_path)
    args = arguments(prepared, root)
    parts = original_parts(prepared, recipe)
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    full.mkdir()
    resumed.mkdir()
    expected = fit_model(args, parts, 'node_only', 'token', recipe, full)
    original_save = training.save_epoch

    def interrupted(path, model, optimizer, scheduler, history):
        original_save(path, model, optimizer, scheduler, history)
        raise RuntimeError('intentional interruption after complete epoch')

    with monkeypatch.context() as patch:
        patch.setattr(training, 'save_epoch', interrupted)
        with pytest.raises(RuntimeError, match='intentional interruption'):
            fit_model(args, parts, 'node_only', 'token', recipe, resumed)
    actual = fit_model(args, parts, 'node_only', 'token', recipe, resumed)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[name], atol=0, rtol=0)
    assert read_json(full / 'history.json') == read_json(resumed / 'history.json')
