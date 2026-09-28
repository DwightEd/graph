"""Regression truth and in-fold probe predictions must not enter fitted readouts."""
import joblib
import numpy as np

from experiments.decision_risk_flow.evaluate import fit_readouts, calibrate


def test_readout_uses_only_source_oof_rows_and_dev_selection(tmp_path):
    rng = np.random.default_rng(91)
    features = rng.normal(size=(5, 20, 2))
    names = ['confidence_0', 'probe_rank8']
    folds = np.arange(20) % 5
    fitting = np.arange(20) < 10
    development = (np.arange(20) >= 10) & (np.arange(20) < 16)
    target = np.arange(20) % 2
    weights = np.ones(20)
    (tmp_path / 'models').mkdir()
    fit_readouts(features, names, target, fitting, development, weights, folds, tmp_path)
    baseline = joblib.load(tmp_path / 'models/readouts.joblib')['probe']
    expected = features[folds[:10], np.arange(10)]
    np.testing.assert_allclose(baseline['scaler'].mean_, expected.mean(0))
    # Poison every probe row that was fitted on its own source, and all
    # regression labels/features. Neither can change the selected readout.
    changed = features.copy()
    for row in range(10):
        changed[np.arange(5) != folds[row], row] = 10000
    changed[:, 16:] = -10000
    target[16:] = 1 - target[16:]
    fit_readouts(changed, names, target, fitting, development, weights, folds, tmp_path)
    poisoned = joblib.load(tmp_path / 'models/readouts.joblib')['probe']
    np.testing.assert_array_equal(baseline['model'].coef_, poisoned['model'].coef_)
    np.testing.assert_array_equal(baseline['scaler'].mean_, poisoned['scaler'].mean_)


def test_thresholds_use_dev_normal_tokens_and_normal_answer_max_only():
    records, regions = [], {}
    for index, task in enumerate(('QA', 'Summary', 'Data2txt')):
        for offset, role in enumerate(('dev', 'dev', 'regression')):
            key = str(index * 3 + offset)
            records.append(dict(key=key, role=role, task=task))
            regions[key] = slice(int(key) * 4, int(key) * 4 + 4)
    target = np.tile([0, 0, 0, 0, 0, 1, 0, 1, -1, -1, -1, -1], 3)
    scores = np.tile([.1, .2, .3, .4, .2, .99, .5, .99, 100, 100, 100, 100], 3)
    result = calibrate(records, regions, target, np.ones(36, dtype=bool), dict(new_method=scores))
    for task in ('QA', 'Summary', 'Data2txt'):
        assert result['token_5pct'][task]['new_method'] == .5
        assert result['answer_5pct'][task]['new_method'] == .4
        assert result['calibration_counts'][task]['normal_answers'] == 1
