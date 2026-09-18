"""Formula equivalence and hypothesis-discriminating tests, not just file existence."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import solve
from sklearn.metrics import roc_auc_score

from experiments.charm_structure_audit.lda_math import (
    moments, coefficients, covariance_terms, common_and_contrast, head_features,
    past_mean, balance_context, fit_linear_prediction)
from experiments.charm_structure_audit.lda_report import pair_indices, conditional_ranking
from experiments.charm_structure_audit.mixture_math import prepare_coordinates, estimate_components, raw_parameters
from experiments.charm_structure_audit.main import main


def correlated_data(seed=42, count=3000):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, count)
    shared = rng.normal(0, 2, count)
    values = rng.normal(0, .1, (count, 4))
    values[:, 0] += shared + labels * .7
    values[:, 1] += shared
    return values, labels


def test_full_matches_existing_mixture_supervised_lda():
    values, labels = correlated_data(count=1000)
    transformed, transform = prepare_coordinates(values, .001)
    reference = raw_parameters(estimate_components(transformed, labels), transform)
    result = coefficients(moments(values, labels, .001))
    np.testing.assert_allclose(result['weight'], reference['coefficient'], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result['intercept'], reference['intercept'], rtol=1e-9, atol=1e-10)


def test_covariance_coefficient_decomposition_is_exact():
    values, labels = correlated_data()
    stats = moments(values, labels)
    weight = coefficients(stats)['weight']
    terms = covariance_terms(stats, weight, heads=2)
    np.testing.assert_allclose(sum(terms.values()), weight, rtol=1e-8, atol=1e-10)
    assert weight[0] > 0 and weight[1] < 0
    assert abs(terms['same_layer'][1]) > abs(terms['direct'][1])


def test_same_layer_noise_cancellation_beats_independent_heads():
    train, y = correlated_data(seed=1)
    test, truth = correlated_data(seed=2)
    stats = moments(train, y)
    auc = {name: roc_auc_score(truth, test @ coefficients(stats, name, 2)['weight'])
           for name in ('diagonal', 'layer_block', 'full')}
    assert auc['full'] > .98 and auc['layer_block'] > .98
    assert auc['diagonal'] < .7


def test_cross_layer_noise_requires_cross_layer_covariance():
    train, y = correlated_data(seed=3)
    test, truth = correlated_data(seed=4)
    order = [0, 2, 1, 3]
    train, test = train[:, order], test[:, order]
    stats = moments(train, y)
    full = roc_auc_score(truth, test @ coefficients(stats, 'full', 2)['weight'])
    block = roc_auc_score(truth, test @ coefficients(stats, 'layer_block', 2)['weight'])
    assert full > .98 and block < .7


def test_head_identity_control_preserves_common_amount():
    normal = np.tile([.9, .1, .2, .2], (40, 1))
    error = np.tile([.1, .9, .2, .2], (40, 1))
    np.testing.assert_allclose(head_features(normal, 2, 'layer_mean'), head_features(error, 2, 'layer_mean'))
    np.testing.assert_allclose(np.sort(normal.reshape(-1, 2, 2), axis=2), np.sort(error.reshape(-1, 2, 2), axis=2))
    assert not np.allclose(normal, error)


def test_common_and_contrast_reconstruct_actual_score():
    values, labels = correlated_data(count=200)
    weight = coefficients(moments(values, labels))['weight']
    common, contrast = common_and_contrast(weight, 2)
    np.testing.assert_allclose(values @ common + values @ contrast, values @ weight)
    np.testing.assert_allclose(contrast.reshape(-1, 2).sum(axis=1), 0, atol=1e-12)


def test_past_mean_cannot_read_current_future_or_other_answer():
    ids = np.array(['a', 'a', 'a', 'b', 'b'])
    values = np.array([1., 3., 9., 100., 7.])
    result = past_mean(values, ids, 2)
    np.testing.assert_allclose(result, [np.nan, 1, 2, np.nan, 100], equal_nan=True)
    values[2] = -999
    np.testing.assert_allclose(past_mean(values, ids, 2), result, equal_nan=True)


def test_balancing_cancels_a_pure_context_label_shortcut():
    meta = pd.DataFrame(dict(position_bin=[0]*100 + [1]*100, surface='word', past_run_bin=0,
                            gold=[0]*90 + [1]*10 + [0]*10 + [1]*90))
    weights, coverage = balance_context(meta)
    values = np.c_[meta.position_bin, np.ones(len(meta))]
    baseline = moments(values, meta.gold.to_numpy())
    balanced = moments(values, meta.gold.to_numpy(), weights=weights)
    assert baseline['means'][1, 0] > baseline['means'][0, 0]
    np.testing.assert_allclose(balanced['means'][1], balanced['means'][0])
    assert all(row['eligible'] for row in coverage)


def test_prompt_regression_is_fit_only_and_can_leave_independent_signal():
    rng = np.random.default_rng(6)
    prompt = rng.normal(size=(500, 3))
    labels = rng.integers(0, 2, 500)
    values = np.c_[prompt[:, 0], labels + rng.normal(0, .1, 500)]
    regression = fit_linear_prediction(prompt, values)
    residual = values - (prompt @ regression['weight'] + regression['intercept'])
    assert residual[:, 0].std() < .01
    assert roc_auc_score(labels, residual[:, 1]) > .99


def fixture(root):
    from test_charm_mixture import fixture as mixture_fixture
    result, prepared = mixture_fixture(root)
    for path in (prepared / 'graphs').rglob('*.npz'):
        with np.load(path) as saved:
            arrays = {k: saved[k] for k in saved.files}
        n = len(arrays['gold'])
        target = np.arange(2, n + 2)
        arrays['edge_index'] = np.stack((np.zeros(n, int), target))
        arrays['edge_attr'] = abs(arrays['x'][2:]) * .02
        np.savez(path, **arrays)
    return result, prepared


@pytest.mark.parametrize('prompt', [False, True])
def test_actual_main_preserves_originals_and_report_needs_no_fit(tmp_path, prompt):
    result, prepared = fixture(tmp_path)
    output = tmp_path / 'out'
    originals = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    args = ['--mode', 'lda_audit', '--root', str(result), '--prepared', str(prepared),
            '--output', str(output), '--bootstrap', '3']
    if prompt:
        args += ['--lda-prompt']
    main(args)
    table = pd.read_csv(output / 'metrics.csv')
    assert set(table.model) >= {'full', 'diagonal', 'layer_block', 'history_mean', 'current_increment'}
    if prompt:
        assert 'self_after_prompt_regression' in set(table.model)
    for path, state in originals.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == state
    main(args + ['--lda-stage', 'report'])
    assert (output / 'lda_review.tar.gz').exists()


def test_node_run_does_not_materialize_edges(tmp_path, monkeypatch):
    from experiments.charm_structure_audit.lda_data import read_split
    result, prepared = fixture(tmp_path)
    record = json.loads((prepared / 'index.json').read_text())[0]
    load = np.load
    class Arrays:
        def __init__(self, array): self.array = array
        def __enter__(self): return self
        def __exit__(self, *args): self.array.close()
        def __getitem__(self, key):
            assert key not in ('edge_index', 'edge_attr', 'embedding')
            return self.array[key]
    monkeypatch.setattr(np, 'load', lambda *a, **k: Arrays(load(*a, **k)))
    values, table, prompts, geometry = read_split([record], prepared)
    assert prompts is None and values.shape == (60, 4)


def test_test_labels_do_not_change_fit_coefficients(tmp_path):
    result, prepared = fixture(tmp_path)
    args = ['--mode', 'lda_audit', '--root', str(result), '--prepared', str(prepared), '--bootstrap', '0']
    a, b = tmp_path / 'first', tmp_path / 'second'
    main(args + ['--output', str(a)])
    for path in (prepared / 'graphs/test').glob('*.npz'):
        with np.load(path) as saved:
            values = {k: saved[k] for k in saved.files}
        values['gold'] = ~values['gold']
        flags = values['gold']
        starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
        ends = np.flatnonzero(flags & ~np.r_[flags[1:], False]) + 1
        values['spans'] = np.stack((starts, ends), axis=1)
        np.savez(path, **values)
    main(args + ['--output', str(b)])
    with np.load(a / 'parameters/full.npz') as left, np.load(b / 'parameters/full.npz') as right:
        np.testing.assert_array_equal(left['weight'], right['weight'])
        np.testing.assert_array_equal(left['intercept'], right['intercept'])


def test_real_pair_contributions_reconstruct_score_gap(tmp_path):
    from experiments.charm_structure_audit.lda_report import save_channel_rules, matched_scores
    values, labels = correlated_data(count=40)
    labels[:4] = [1, 1, 0, 0]
    stats = moments(values, labels)
    fitted = coefficients(stats)
    terms = covariance_terms(stats, fitted['weight'], 2)
    table = pd.DataFrame(dict(id='a', source_id='s', token=np.arange(40), gold=labels))
    pair = dict(id='a', source_id='s', error_start=0, normal_start=2, length=2)
    save_channel_rules(stats, fitted, terms, (values, table, None, (2, 2)), [pair], tmp_path)
    rows = pd.read_csv(tmp_path / 'pair_head_contributions.csv.gz')
    contribution = rows[rows.region == 'all'].score_contribution.sum()
    expected = (values[:2].mean(0) - values[2:4].mean(0)) @ fitted['weight']
    assert contribution == pytest.approx(expected)


def test_prompt_edges_are_summed_per_original_head():
    from experiments.charm_structure_audit.lda_data import prompt_features
    saved = dict(prompt_length=2, gold=np.zeros(3), x=np.ones((5, 4)) * .1,
                 edge_index=np.array([[0, 1, 2, 1], [2, 3, 3, 4]]),
                 edge_attr=np.arange(16).reshape(4, 4) * .01)
    prompt, retained = prompt_features(saved)
    np.testing.assert_allclose(prompt[0], saved['edge_attr'][0])
    np.testing.assert_allclose(prompt[1], saved['edge_attr'][1])
    np.testing.assert_allclose(retained[1], (saved['edge_attr'][1] + saved['edge_attr'][2] + .1).mean())


def test_scores_stage_runs_on_existing_format_without_raw_nodes(tmp_path):
    from experiments.charm_structure_audit.lda_report import report_saved_scores
    result = tmp_path / 'root'
    mixture = result / 'audit_mixture'
    mixture.mkdir(parents=True)
    table = pd.DataFrame(dict(id=['a']*8, source_id=['s']*8, token=np.arange(8), seed=2,
        position=(np.arange(8)+.5)/8, gold=[0, 0, 1, 1, 1, 0, 0, 0],
        surface='word', text='word', supervised_lda=[-1, -1, 2, 2, 2, -1, -1, -1]))
    table.to_csv(mixture / 'test_tokens.csv.gz', index=False)
    (mixture / 'frozen.json').write_text(json.dumps(dict(seed_by_unlabelled_select_density=2)))
    main(['--mode', 'lda_audit', '--lda-stage', 'scores', '--root', str(result), '--bootstrap', '0'])
    scores = pd.read_csv(result / 'audit_lda_scores/metrics.csv')
    assert set(scores.model) == {'full', 'history_mean', 'current_increment'}
    assert set(scores[scores.population == 'common_all'].tokens) == {7}


def test_full_cli_with_locked_pairs_emits_exact_head_summary(tmp_path):
    result, prepared = fixture(tmp_path)
    with np.load(prepared / 'graphs/test/test0.npz') as saved:
        labels = saved['gold'].astype(bool)
    error_start = int(np.flatnonzero(labels[:-1] & labels[1:])[0])
    normal_start = int(np.flatnonzero(~labels[:-1] & ~labels[1:])[0])
    folder = result / 'charm_in/test/cluster_audit'
    folder.mkdir(parents=True)
    pair = dict(id='test0', source_id='test0source', error_start=error_start,
                normal_start=normal_start, length=2, tier='cluster')
    (folder / 'pairs.json').write_text(json.dumps([pair]))
    main(['--mode', 'lda_audit', '--root', str(result), '--prepared', str(prepared), '--bootstrap', '3'])
    output = result / 'audit_lda_nodes'
    channels = pd.read_csv(output / 'channel_summary.csv')
    paired = pd.read_csv(output / 'matched_pairs.csv')
    expected = paired[(paired.model == 'full') & (paired.region == 'all')].gap.iloc[0]
    assert channels[channels.region == 'all'].contribution.sum() == pytest.approx(expected)
    assert channels.low.isna().all()
