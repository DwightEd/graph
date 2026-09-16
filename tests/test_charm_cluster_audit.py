"""Synthetic invariants and actual CLI; no natural-data performance claims."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.charm_structure_audit.cluster_audit.features import (
    head_profile, internal_edges, marginal_windows, prepare_features,
    representation_geometry, structure_at, window_average,
)
from experiments.charm_structure_audit.cluster_audit.io import (
    load_capture, load_graph, population, read_predictions, verify_identity,
)
from experiments.charm_structure_audit.cluster_audit.matching import (
    TIERS, candidates, match_answer, merged_spans,
)
from experiments.charm_structure_audit.cluster_audit.reporting import (
    evaluate_pair, source_mean, summarize_pairs, win,
)
from experiments.charm_structure_audit.cluster_audit.__main__ import DEFAULTS, main


def fixture(identity="a", source_id="s1", two_errors=False):
    prompt, count, channels = 2, 32, 4
    words = [f"word{i % 4}" for i in range(count)]
    response = " ".join(words)
    offsets, cursor = [], 0
    for word in words:
        offsets.append([cursor, cursor + len(word)])
        cursor += len(word) + 1
    edges, values = [], []
    for token in range(count):
        edges.append([0, prompt + token])
        values.append([.2] * channels)
        if token % 4:
            edges.append([prompt + token - 1, prompt + token])
            values.append([.2] * channels)
    edge_index = np.asarray(edges).T
    spans = np.asarray([[12, 16], [24, 28]] if two_errors else [[12, 16]])
    gold = np.zeros(count, bool)
    onset = np.zeros(count, bool)
    for start, end in spans:
        gold[start:end] = True
        onset[start] = True
    score = np.full(count, .1)
    score[gold] = .9
    score[8:12] = .9
    sample = dict(id=identity, source_id=source_id, task="QA", generator="fixture", split="test",
                  score=score, gold=gold, onset=onset, spans=spans, offsets=np.asarray(offsets), response=response)
    graph = dict(x=np.full((prompt + count, channels), .1, np.float32),
                 edge_index=edge_index, edge_attr=np.asarray(values, np.float32),
                 edge_mark=np.stack((edge_index[0] < prompt, edge_index[0] >= prompt), axis=1).astype(np.float32),
                 prompt_length=np.asarray(prompt), layers=np.asarray(2), heads=np.asarray(2),
                 token_ids=np.r_[90, 91, np.tile(np.arange(4), 8)])
    return sample, graph


def identity_pair(**changes):
    row = dict(id="a", source_id="s1", tier="cluster", error_start=12, normal_start=8, length=4)
    row.update(changes)
    return row


def save_fixture(tmp_path, two_sources=False):
    root = tmp_path / "charm_in" / "test"
    prepared = tmp_path / "prepared"
    (root / "samples").mkdir(parents=True)
    (prepared / "graphs" / "test").mkdir(parents=True)
    paths = []
    for identity, source in ([('a', 's1'), ('b', 's2')] if two_sources else [('a', 's1')]):
        sample, graph = fixture(identity, source)
        graph_path = prepared / "graphs" / "test" / (identity + ".npz")
        record = {k: sample[k] for k in ("id", "source_id", "task", "generator", "split")}
        record["graph"] = str(graph_path)
        labels = {k: sample[k] for k in ("gold", "onset", "spans", "offsets", "response")}
        np.savez_compressed(graph_path, **graph, **labels, record_json=np.asarray(json.dumps(record)))
        path = root / "samples" / (identity + ".npz")
        np.savez_compressed(path, **labels, score=sample["score"], embedding=graph['x'][2:],
                            record_json=np.asarray(json.dumps(record)))
        paths.append("/old/absolute/" + path.name)
    settings = dict(variant="charm_in", seed=0, threshold=dict(value=.7, rule="score > threshold"))
    (root / "prediction_settings.json").write_text(json.dumps(settings))
    (root / "predictions.json").write_text(json.dumps(paths))
    return root, prepared


def test_window_average_all_axes():
    values = np.arange(48).reshape(6, 2, 4)
    result = window_average(values, 3)
    np.testing.assert_allclose(result, [values[i:i + 3].mean(axis=0) for i in range(4)])


def test_chain_structure_known_values():
    sample, graph = fixture()
    data = prepare_features(graph)
    values = structure_at(data, 12, 4)
    assert len(internal_edges(data, 12, 4)) == 3
    np.testing.assert_allclose(values[:5], [.5, 1., 3/7, 1., 1/3], atol=1e-7)


def test_per_head_entropy_kept_distribution():
    _, graph = fixture()
    data = prepare_features(graph)
    np.testing.assert_allclose(data['entropy'][12], 0.)
    np.testing.assert_allclose(data['entropy'][13], 1.)
    assert marginal_windows(data, 4).shape == (29, 3, 4)


def test_all_channels_kept_in_profile():
    _, graph = fixture()
    data = prepare_features(graph)
    profile = head_profile(data, 12, 4)
    assert profile.shape == (5, 4)
    np.testing.assert_allclose(profile[:, 0], [.1, .35, .75, .625, .375], atol=1e-7)


def test_matching_never_reads_score_or_embeddings():
    sample, graph = fixture()
    data = prepare_features(graph)
    pairs1, status1, skipped1 = match_answer(sample, data, DEFAULTS)
    altered = dict(sample, score=np.full(32, np.nan), embedding=np.full((32, 8), np.nan))
    pairs2, status2, skipped2 = match_answer(altered, data, DEFAULTS)
    assert pairs1 == pairs2 and status1 == status2 and skipped1 == skipped2
    assert len(pairs1) == 3
    assert all(row['normal_start'] == 8 for row in pairs1)


def test_controls_normal_equal_length_nonoverlap():
    sample, graph = fixture(two_errors=True)
    pairs, _, _ = match_answer(sample, prepare_features(graph), DEFAULTS)
    for tier in TIERS:
        used = set()
        for pair in [r for r in pairs if r['tier'] == tier]:
            left, length = pair['normal_start'], pair['length']
            assert not sample['gold'][left:left+length].any()
            ids = set(range(left, left+length))
            assert not ids & used
            used |= ids
            assert sample['gold'][:left].any() == sample['gold'][:pair['error_start']].any()


def test_no_structural_match_is_not_replaced_by_easy_normal():
    sample, graph = fixture()
    rr = graph['edge_index'][0] >= 2
    query = graph['edge_index'][1] - 2
    # Only the actual error keeps internal RR edges.
    keep = ~rr | ((query >= 12) & (query < 16))
    for key in ('edge_attr', 'edge_mark'):
        graph[key] = graph[key][keep]
    graph['edge_index'] = graph['edge_index'][:, keep]
    pairs, status, _ = match_answer(sample, prepare_features(graph), DEFAULTS)
    assert any(row['tier'] == 'context' for row in pairs)
    assert not any(row['tier'] == 'cluster' for row in pairs)
    assert next(r for r in status if r['tier'] == 'cluster')['reason'] == 'no_candidate'


def test_singletons_do_not_count_as_internal_clusters():
    sample, graph = fixture()
    sample.update(spans=np.array([[12, 13]]), gold=np.arange(32) == 12)
    pairs, _, skipped = match_answer(sample, prepare_features(graph), DEFAULTS)
    assert pairs == []
    assert skipped[0]['reason'] == 'singleton_has_no_internal_cluster'


def test_adjacent_annotations_not_merged():
    assert merged_spans([[1, 3], [2, 4], [4, 6]]) == [[1, 4], [4, 6]]


def test_high_global_auc_can_be_only_clustering():
    from sklearn.metrics import roc_auc_score
    sample, _ = fixture()
    assert roc_auc_score(sample['gold'], sample['score']) > .9
    row = evaluate_pair(sample['score'], identity_pair(), .7)
    assert row['within_pair_token_auroc'] == .5
    assert row['normal_cluster_token_fpr'] == 1.
    assert row['span_mean_win'] == .5


def test_token_auc_and_fixed_alarm_are_not_same():
    scores = np.full(32, .1)
    scores[12:16] = [.5, .6, .7, .8]
    scores[8:12] = [.1, .2, .3, .4]
    row = evaluate_pair(scores, identity_pair(), .7)
    assert row['within_pair_token_auroc'] == 1.
    assert row['error_token_recall'] == .25
    assert row['error_start_hit'] == 0.


def test_partial_control_does_not_borrow_unmatched_baselines():
    sample, _ = fixture()
    prefix = np.full(32, np.nan)
    prefix[[12, 8]] = [.9, .2]
    result, _ = summarize_pairs([identity_pair()], {'a': dict(score=sample['score'], score_prefix=prefix)}, .7, 0)
    control = result['controls']['score_prefix']
    assert control['complete_pairs'] == 0
    assert control['baseline_on_same_pairs']['pairs'] == 0
    assert control['start_only_pairs'] == 1
    assert control['start_only_win_delta']['mean'] == .5


def test_common_shift_not_invented_as_discrimination_gain():
    sample, _ = fixture()
    scores = {'a': dict(score=sample['score'], score_shift=sample['score'] - .3)}
    result, _ = summarize_pairs([identity_pair()], scores, .7, 0)
    delta = result['controls']['score_shift']['delta_control_minus_base']['pair_macro']
    assert delta['within_pair_token_auroc'] == 0.
    assert delta['mean_score_margin'] == 0.
    assert delta['normal_cluster_token_fpr'] == -1.
    assert delta['error_token_recall'] == -1.


def test_independent_model_uses_own_threshold():
    sample, _ = fixture()
    scores = {'a': dict(score=sample['score'], model_node=np.full(32, .6))}
    result, _ = summarize_pairs([identity_pair()], scores, .7, 0, thresholds={'model_node': .5})
    delta = result['controls']['model_node']['delta_control_minus_base']['pair_macro']
    assert delta['error_token_recall'] == 0.


def test_source_not_token_is_bootstrap_unit():
    rows = [dict(source_id='a', value=1.)] * 100 + [dict(source_id='b', value=0.)]
    result = source_mean(rows, 'value', 20)
    assert result['mean'] == .5 and result['sources'] == 2


def test_correct_and_error_both_cluster_in_geometry():
    x = np.r_[np.tile([[1., 0.]], (4, 1)), np.tile([[0., 1.]], (4, 1))]
    measured = representation_geometry(x, 0, 4, 4)
    assert measured['error_within_cosine'] == measured['normal_within_cosine'] == 1.
    assert measured['cross_span_cosine'] == 0.


def test_relocated_prediction_manifest(tmp_path):
    root, _ = save_fixture(tmp_path)
    samples, settings = read_predictions(root)
    assert samples[0]['id'] == 'a' and settings['variant'] == 'charm_in'


def test_graph_identity_mismatch_rejected(tmp_path):
    root, prepared = save_fixture(tmp_path)
    samples, _ = read_predictions(root)
    samples[0]['source_id'] = 'wrong'
    with pytest.raises(ValueError, match='Identity mismatch'):
        load_graph(samples[0], prepared)


def test_capture_mismatch_rejected(tmp_path):
    root, prepared = save_fixture(tmp_path)
    samples, _ = read_predictions(root)
    _, graph_path = load_graph(samples[0], prepared)
    directory = tmp_path / 'captures'
    directory.mkdir()
    stamp = [graph_path.stat().st_size, graph_path.stat().st_mtime_ns]
    np.savez(directory / 'a.npz', graph_stamp=stamp, replay_score=np.zeros(32))
    with pytest.raises(ValueError, match='does not replay'):
        load_capture(samples[0], graph_path, directory)


def test_population_first_error_denominators():
    one, _ = fixture(two_errors=True)
    pop = population([one])
    assert pop['error_tokens'] == 8 and pop['first_error_tokens'] == 1
    assert pop['first_error_fraction_error'] == .125


def test_actual_cli_inputs_unchanged_and_no_torch(tmp_path):
    root, prepared = save_fixture(tmp_path, two_sources=True)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    main(['--root', str(root), '--prepared', str(prepared), '--bootstrap', '3'])
    for path, (content, timestamp) in before.items():
        assert Path(path).read_bytes() == content
        assert Path(path).stat().st_mtime_ns == timestamp
    report = json.loads((root / 'cluster_audit' / 'report.json').read_text())
    assert report['tiers']['cluster']['pairs'] == 2
    assert report['tiers']['cluster']['pair_macro']['within_pair_token_auroc'] == .5
    assert (root / 'cluster_audit' / 'pairs.html').exists()
    assert (root / 'cluster_audit' / 'tokens.csv.gz').exists()


def test_python_module_entrypoint(tmp_path):
    root, prepared = save_fixture(tmp_path)
    code = "import runpy, sys; sys.argv = ['cluster_audit'] + sys.argv[1:]; runpy.run_module('experiments.charm_structure_audit.cluster_audit', run_name='__main__'); assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, '-c', code,
                               '--root', str(root), '--prepared', str(prepared), '--bootstrap', '0'],
                              capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert 'Saved matched cluster audit:' in completed.stdout


def test_matching_per_head_marginals_is_additional_not_replacement():
    sample, graph = fixture()
    graph['x'][2:14] = .9
    pairs, status, _ = match_answer(sample, prepare_features(graph), DEFAULTS)
    assert any(r['tier'] == 'cluster' for r in pairs)
    assert not any(r['tier'] == 'cluster_heads' for r in pairs)


def test_verified_capture_controls_reused_on_same_pairs(tmp_path):
    root, prepared = save_fixture(tmp_path)
    samples, _ = read_predictions(root)
    _, graph_path = load_graph(samples[0], prepared)
    directory = root / 'deep_audit' / 'captures'
    directory.mkdir(parents=True)
    stamp = [graph_path.stat().st_size, graph_path.stat().st_mtime_ns]
    np.savez(directory / 'a.npz', graph_stamp=stamp, replay_score=samples[0]['score'],
             score_no_messages=np.full(32, .1), feature_projected=np.ones((32, 2)))
    main(['--root', str(root), '--prepared', str(prepared), '--bootstrap', '0'])
    report = json.loads((root / 'cluster_audit' / 'report.json').read_text())
    control = report['tiers']['cluster']['controls']['capture_score_no_messages']
    assert control['complete_pairs'] == control['total_pairs'] == 1
    assert control['delta_control_minus_base']['pair_macro']['within_pair_token_auroc'] == 0.
    assert report['common_error_cohort']['error_spans'] == 1
    geometry = (root / 'cluster_audit' / 'representation_geometry.csv').read_text()
    assert 'projected' in geometry


def test_empty_cluster_match_cli_reports_missing_not_half(tmp_path):
    root, prepared = save_fixture(tmp_path)
    path = prepared / 'graphs' / 'test' / 'a.npz'
    with np.load(path) as saved:
        arrays = {k: saved[k] for k in saved.files}
    keep = arrays['edge_index'][0] < 2
    arrays['edge_index'] = arrays['edge_index'][:, keep]
    arrays['edge_attr'] = arrays['edge_attr'][keep]
    arrays['edge_mark'] = arrays['edge_mark'][keep]
    np.savez(path, **arrays)
    main(['--root', str(root), '--prepared', str(prepared), '--bootstrap', '0'])
    report = json.loads((root / 'cluster_audit' / 'report.json').read_text())
    assert report['tiers']['cluster']['pairs'] == 0
    assert report['tiers']['cluster']['pair_macro']['within_pair_token_auroc'] is None
