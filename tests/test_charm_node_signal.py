"""Regression tests for separating node inputs, model decisions and prior audits."""

from pathlib import Path
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
import torch

from test_charm_structure_audit import fixture_files, graph_sample, model
from experiments.charm_structure_audit.main import main
from experiments.charm_structure_audit.data import token_frame, save_scores, read_json
from experiments.charm_structure_audit.compare_models import (
    aligned_models, comparison_tokens, decision_counts, relative_regions, matched_tokens)
from experiments.charm_structure_audit.node_signal import (
    pointwise_logits, pointwise_scores, changed_attributes, source_vectors,
    vector_interval, attribute_statistics, pair_indices)
from experiments.charm_structure_audit.review_results import copy_completed


def two_models(tmp_path):
    root, prepared = fixture_files(tmp_path)
    graph, sample = graph_sample('test')
    net = model()
    scores = torch.sigmoid(net(graph, ablation='no_graph')[2:]).detach().numpy()
    directory = root/'node_only'
    directory.mkdir()
    hp = dict(hidden_dim=8, gnn_layers=2, residual_mp=True)
    torch.save(dict(model_state=net.state_dict(), hp=hp), directory/'checkpoint.pt')
    (directory/'test').mkdir()
    token_frame(sample, scores, .5).to_csv(directory/'test/tokens.csv', index=False)
    pd.read_csv(root/'charm_in/test/spans.csv').to_csv(directory/'test/spans.csv', index=False)
    (directory/'threshold.json').write_text(json.dumps(dict(value=.5)))
    return root, prepared


def test_pointwise_model_is_exact_no_graph_forward():
    graph, _ = graph_sample()
    rng = np.random.default_rng(5)
    graph['x'] = rng.random(graph['x'].shape).astype(np.float32)
    net = model()
    expected = net(graph, ablation='no_graph')
    actual = pointwise_logits(net, torch.as_tensor(graph['x']))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(p.grad is None for layer in net.mp_layers for p in layer.msg_mlp.parameters())


def test_node_prediction_does_not_depend_on_other_rows():
    net = model()
    values = np.random.default_rng(4).random((9, 4)).astype(np.float32)
    changed = values.copy()
    changed[1:] = 0
    assert pointwise_scores(net, values)[0] == pytest.approx(pointwise_scores(net, changed)[0], abs=1e-7)


def test_full_swap_interchanges_scores_and_preserves_input():
    left = np.random.default_rng(5).random((8, 4)).astype(np.float32)
    right = np.random.default_rng(7).random((8, 4)).astype(np.float32)
    original = left.copy()
    a, b = changed_attributes(left, right, dict(columns=[0, 1, 2, 3], operation='swap'))
    np.testing.assert_array_equal(a, right)
    np.testing.assert_array_equal(b, left)
    np.testing.assert_array_equal(left, original)
    net = model()
    np.testing.assert_allclose(pointwise_scores(net, a), pointwise_scores(net, right))


def test_single_channel_swap_keeps_every_other_head():
    left = np.ones((3, 4), np.float32)
    right = np.full((3, 4), 2., np.float32)
    a, b = changed_attributes(left, right, dict(columns=[2], operation='swap'))
    np.testing.assert_array_equal(a[:, [0, 1, 3]], left[:, [0, 1, 3]])
    np.testing.assert_array_equal(a[:, 2], right[:, 2])
    np.testing.assert_array_equal(b[:, 2], left[:, 2])


def test_half_definitions_and_overlapping_views_have_denominators():
    masks = relative_regions(np.arange(5), np.full(5, 5))
    assert masks['front_half'].sum() == 2
    assert masks['back_half'].sum() == 3
    np.testing.assert_array_equal(masks['front_half'] | masks['back_half'], True)
    assert sum(masks[k].sum() for k in ('early_third', 'middle_third', 'late_third')) == 5


def test_sources_not_tokens_control_attribute_average():
    meta = pd.DataFrame(dict(id=['a']*100+['b'], source_id=['s1']*100+['s2'], pair_error_start=[2]*101))
    vectors = np.r_[np.ones((100, 4)), np.zeros((1, 4))]
    grouped = source_vectors(vectors, meta)
    np.testing.assert_allclose(grouped.mean(axis=0), .5)
    _, low, high = vector_interval(grouped[:1], 100)
    assert np.isnan(low).all() and np.isnan(high).all()


def test_raw_feature_direction_is_not_optimized_on_labels():
    paired = pd.DataFrame(
        dict(id=['a', 'a'], source_id=['s', 's'], pair_error_start=[1, 1], pair_offset=[0, 1], pair_length=[2, 2],
             node_alarm=[True, False], full_alarm=[True, True]))
    frame, coverage = attribute_statistics(np.zeros((2, 4)), np.ones((2, 4)), paired, 2, 0)
    all_rows = frame[(frame.region == 'all') & (frame.cohort == 'all')]
    assert (all_rows.high_value_pair_win == 0).all()
    assert (all_rows.source_delta == -1).all()
    assert set(coverage.cohort) >= {'node_hit', 'node_miss'}


def test_matched_windows_reject_reused_coordinates(tmp_path):
    root, _ = two_models(tmp_path)
    tables, _, _ = aligned_models(root)
    table = comparison_tokens(tables)
    pairs = read_json(root/'charm_in/test/cluster_audit/pairs.json')
    with pytest.raises(ValueError, match='reuse'):
        matched_tokens(table, pairs+pairs, 'cluster')


def test_decision_groups_cover_both_error_and_normal():
    table = pd.DataFrame(dict(gold=[1]*4+[0]*4, offset=[0,1,2,3]*2, span_length=[4]*8,
        decision_group=['both_correct', 'node_only_correct', 'full_only_correct', 'both_wrong']*2))
    counts = decision_counts(table)
    whole = counts[counts.region == 'all']
    assert len(whole) == 8
    assert (whole.tokens == 1).all()
    assert (whole.denominator == 4).all()


def test_compact_copy_omits_models_and_big_score_arrays(tmp_path):
    source = tmp_path/'audit_heads'
    (source/'scores').mkdir(parents=True)
    (source/'scores/a.npz').write_bytes(b'large simulated score data')
    (source/'checkpoint.pt').write_bytes(b'no model export')
    (source/'channel_effects.csv').write_text('name,auroc_delta\na,-.01\n')
    files = copy_completed(source, tmp_path/'copy')
    assert [p.name for p in files] == ['channel_effects.csv']


def test_all_new_modes_keep_original_inputs_and_no_edge_read(tmp_path, monkeypatch):
    root, prepared = two_models(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    original = np.lib.npyio.NpzFile.__getitem__
    def restricted(self, key):
        assert key not in ('edge_attr', 'edge_index', 'edge_mark', 'embedding')
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', restricted)
    common = ['--root', str(root), '--prepared', str(prepared), '--device', 'cpu', '--bootstrap', '0']
    for mode in ('compare', 'node', 'review'):
        main(common+['--mode', mode, '--output', str(tmp_path/mode), '--llm-layers', '0'])
    for path, expected in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == expected
    assert (tmp_path/'compare/figures/node_vs_full_position_recall.png').exists()
    assert (tmp_path/'node/node_channel_effects.csv').exists()
    effects = pd.read_csv(tmp_path/'node/node_channel_effects.csv')
    swap = effects[(effects.unit == 'all_swap') & (effects.region == 'all')]
    np.testing.assert_allclose(swap.changed_auc, 1-swap.base_auc, atol=1e-7)
    with tarfile.open(tmp_path/'review/review_bundle.tar.gz') as tar:
        assert not any(n.endswith(('.npz', '.pt')) for n in tar.getnames())
        assert 'review/REPORT_zh.md' in tar.getnames()


def test_wrong_node_checkpoint_stops_before_attribution(tmp_path):
    root, prepared = two_models(tmp_path)
    path = root/'node_only/test/tokens.csv'
    table = pd.read_csv(path, keep_default_na=False)
    table['score'] = .1
    table['predicted'] = 0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match='node_only pointwise replay mismatch'):
        main(['--mode', 'node', '--root', str(root), '--prepared', str(prepared), '--device', 'cpu',
              '--output', str(tmp_path/'bad'), '--bootstrap', '0'])


def test_coordinate_mapping_uses_ids_not_row_order(tmp_path):
    root, _ = two_models(tmp_path)
    tables, _, _ = aligned_models(root)
    paired = matched_tokens(comparison_tokens(tables), read_json(root/'charm_in/test/cluster_audit/pairs.json'), 'cluster')
    a, b = pair_indices(tables['node_only'], paired)
    np.testing.assert_array_equal(a, [12, 13, 14, 15])
    np.testing.assert_array_equal(b, [8, 9, 10, 11])


def test_review_reads_finished_audits_without_model_execution(tmp_path, monkeypatch):
    root, prepared = two_models(tmp_path)
    common = ['--root', str(root), '--prepared', str(prepared), '--device', 'cpu', '--bootstrap', '0']
    main(common+['--mode', 'routes'])
    main(common+['--mode', 'heads', '--llm-layers', '0'])
    from experiments.charm_structure_audit import model as model_module
    def forbidden(*args, **kwargs):
        raise AssertionError('review must not load a checkpoint')
    monkeypatch.setattr(model_module, 'load_checkpoint', forbidden)
    main(common+['--mode', 'review'])
    report = (root/'audit_review/REPORT_zh.md').read_text()
    assert '逐层逐头路由关联：已找到文件' in report
    assert '完整CHARM通道敏感性：已找到文件' in report
    assert '节点模型自身的通道依赖：未找到' in report
    assert (root/'audit_review/route_coverage_0.csv').exists()


def test_old_config_keys_still_match_defaults(tmp_path):
    root, prepared = two_models(tmp_path)
    command = ['--mode', 'report', '--root', str(root), '--prepared', str(prepared), '--models', 'charm_in', '--bootstrap', '0']
    main(command)
    config = read_json(root/'audit_report/config.json')
    assert 'node_operations' not in config and 'audit_inputs' not in config
    assert 'channel_unit' not in config
    main(command)
