"""Check equations, label separation, real CLI and counterexamples."""

from pathlib import Path
import copy
import json
import hashlib

import numpy as np
import pytest
from scipy.special import logsumexp, expit
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from experiments.charm_structure_audit.mixture_math import (
    prepare_coordinates, transform_values, estimate_components, raw_parameters,
    fit_two, log_densities, canonical_order)
from experiments.charm_structure_audit.main import main


def data(seed=3, count=1000, width=4):
    rng = np.random.default_rng(seed)
    labels = rng.random(count) < .3
    values = rng.normal(size=(count, width))
    values[:, 0] += 5 * labels
    return values, labels


def test_fast_mstep_equals_full_weighted_covariance():
    values, _ = data(count=200)
    coordinates, transform = prepare_coordinates(values, ridge=.003)
    response = np.random.default_rng(2).uniform(.01, .99, len(values))
    state = estimate_components(coordinates, response)
    raw = raw_parameters(state, transform)
    responsibilities = np.stack((1 - response, response), axis=1)
    means = responsibilities.T @ values / responsibilities.sum(axis=0)[:, None]
    covariance = np.zeros((values.shape[1], values.shape[1]))
    for k in range(2):
        residual = values - means[k]
        covariance += (residual * responsibilities[:, k, None]).T @ residual / len(values)
    covariance += .003 * np.diag(transform['scale'] ** 2)
    np.testing.assert_allclose(raw['means'], means, atol=1e-12)
    np.testing.assert_allclose(raw['covariance'], covariance, atol=1e-12)
    expected = np.linalg.solve(covariance, means[1] - means[0])
    np.testing.assert_allclose(raw['coefficient'], expected, atol=1e-12)


def test_linear_odds_equal_full_gaussian_densities():
    values, _ = data(count=200)
    coordinates, transform = prepare_coordinates(values)
    response = np.random.default_rng(4).uniform(.1, .9, len(values))
    state = estimate_components(coordinates, response)
    raw = raw_parameters(state, transform)
    component_logs = np.stack([np.log(raw['priors'][k]) +
        multivariate_normal.logpdf(values, raw['means'][k], raw['covariance']) for k in range(2)])
    odds = values @ raw['coefficient'] + raw['intercept']
    np.testing.assert_allclose(odds, component_logs[1] - component_logs[0], atol=1e-12)
    _, fast_density = log_densities(coordinates, state, transform)
    np.testing.assert_allclose(fast_density, logsumexp(component_logs, axis=0), atol=1e-12)


def test_whitening_is_invertible_and_keeps_every_channel():
    values, _ = data(count=200)
    coordinates, transform = prepare_coordinates(values)
    recovered = (coordinates @ transform['factor'].T) * transform['scale'] + transform['center']
    np.testing.assert_allclose(recovered, values, atol=1e-12)
    np.testing.assert_allclose(transform_values(values, transform), coordinates)
    assert coordinates.shape == values.shape


def test_label_switch_preserves_density_and_reverses_odds():
    values, _ = data(count=300)
    coordinates, transform = prepare_coordinates(values)
    response = expit(coordinates[:, 0])
    state = estimate_components(coordinates, response)
    flipped = estimate_components(coordinates, 1 - response)
    np.testing.assert_allclose(state['coefficient'], -flipped['coefficient'])
    np.testing.assert_allclose(log_densities(coordinates, state, transform)[1],
                               log_densities(coordinates, flipped, transform)[1])
    a = canonical_order(state, transform)
    b = canonical_order(flipped, transform)
    np.testing.assert_allclose(a['coefficient'], b['coefficient'])


def test_one_gaussian_is_nested_at_identical_components():
    values, _ = data(count=200)
    coordinates, transform = prepare_coordinates(values)
    state = estimate_components(coordinates, np.full(len(values), .5))
    single, two = log_densities(coordinates, state, transform)
    np.testing.assert_allclose(single, two, atol=1e-12)
    np.testing.assert_allclose(state['coefficient'], 0, atol=1e-12)


def test_known_tied_gaussian_separation_is_recovered_without_labels():
    values, labels = data(count=1500)
    coordinates, transform = prepare_coordinates(values)
    initial = KMeans(n_clusters=2, n_init=1, random_state=2).fit_predict(values)
    state, history = fit_two(coordinates, transform, initial)
    odds = coordinates @ state['coefficient'] + state['intercept']
    assert roc_auc_score(labels, odds) > .98
    assert history[-1]['log_density'] > history[0]['log_density']


def test_linear_separation_does_not_require_two_gaussians():
    rng = np.random.default_rng(7)
    values = rng.uniform(-1, 1, (3000, 4))
    labels = values[:, 0] > 0
    assert roc_auc_score(labels, values[:, 0]) == 1
    # Uniform box data have bounded support and are not Gaussian class conditionals.
    assert values.min() >= -1 and values.max() <= 1


def test_two_real_clusters_need_not_be_truth_clusters():
    values, latent = data(count=3000)
    truth = values[:, 1] > 0
    coordinates, transform = prepare_coordinates(values)
    initial = KMeans(n_clusters=2, n_init=1, random_state=1).fit_predict(values)
    state, _ = fit_two(coordinates, transform, initial)
    odds = coordinates @ state['coefficient'] + state['intercept']
    assert roc_auc_score(latent, odds) > .98
    assert .45 < roc_auc_score(truth, odds) < .55
    assert roc_auc_score(truth, values[:, 1]) == 1


def fixture(root):
    import torch
    from experiments.charm_structure_audit.model import CHARM

    prepared = root / 'prepared'
    result = root / 'seed_0'
    directory = result / 'node_only'
    directory.mkdir(parents=True)
    rows, partitions = [], {}
    for split in ('fit', 'select', 'calibration', 'test'):
        identifiers = []
        for number in range(3):
            name = split + str(number)
            values, labels = data(seed=number + len(split), count=60)
            official = 'test' if split == 'test' else 'train'
            path = prepared / 'graphs' / official / (name + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            starts = np.flatnonzero(labels & ~np.r_[False, labels[:-1]])
            ends = np.flatnonzero(labels & ~np.r_[labels[1:], False]) + 1
            np.savez(path, x=np.vstack((np.zeros((2, 4)), values)).astype(np.float32),
                prompt_length=2, layers=2, heads=2, gold=labels, spans=np.stack((starts, ends), axis=1),
                response=' '.join(['word'] * len(labels)),
                offsets=np.array([[5*i, 5*i+4] for i in range(len(labels))]))
            rows.append(dict(id=name, source_id=name + 'source', split=official, response_tokens=60, positives=int(labels.sum())))
            identifiers.append(name)
        partitions[split] = identifiers
    (prepared / 'index.json').write_text(json.dumps(rows))
    (directory / 'training.json').write_text(json.dumps(dict(partitions=partitions)))
    hp = dict(hidden_dim=8, gnn_layers=2, residual_mp=True)
    model = CHARM(4, 4, hp)
    torch.save(dict(model_state=model.state_dict(), hp=hp), directory / 'checkpoint.pt')
    return result, prepared


def test_actual_cli_freezes_before_labels_and_reuses_finished_fit(tmp_path, monkeypatch):
    from experiments.charm_structure_audit import mixture_report
    result, prepared = fixture(tmp_path)
    output = result / 'audit_mixture'
    command = ['--mode', 'mixture', '--root', str(result), '--prepared', str(prepared),
               '--mixture-seeds', '0', '1', '--mixture-iterations', '30', '--bootstrap', '3']
    original = mixture_report.labels_for
    def checked(record, path):
        assert (output / 'frozen.json').exists()
        return original(record, path)
    monkeypatch.setattr(mixture_report, 'labels_for', checked)
    main(command)
    assert (output / 'mixture_review.tar.gz').exists()
    assert (output / 'cdf_checks.csv').exists()
    frozen = {path: path.read_bytes() for path in (output / 'frozen_scores').rglob('*.npz')}
    model_bytes = (output / 'model_0.npz').read_bytes()
    main(command + ['--mixture-stage', 'report'])
    assert (output / 'model_0.npz').read_bytes() == model_bytes
    assert all(path.read_bytes() == content for path, content in frozen.items())


def test_fit_reads_no_labels_or_edges_or_model(tmp_path, monkeypatch):
    import numpy.lib.npyio
    result, prepared = fixture(tmp_path)
    loaded = np.load
    class NamedArrays:
        def __init__(self, arrays): self.arrays = arrays
        def __enter__(self): return self
        def __exit__(self, *args): self.arrays.close()
        def __getitem__(self, key):
            assert key not in ('gold', 'spans', 'edge_index', 'edge_attr', 'embedding', 'score')
            return self.arrays[key]
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: NamedArrays(loaded(*args, **kwargs)))
    (result / 'node_only/checkpoint.pt').unlink()
    main(['--mode', 'mixture', '--mixture-stage', 'fit', '--root', str(result), '--prepared', str(prepared),
          '--mixture-seeds', '0', '--mixture-iterations', '10'])
    assert (result / 'audit_mixture/frozen.json').exists()


def test_test_label_change_cannot_change_fitted_models_or_scores(tmp_path):
    result, prepared = fixture(tmp_path)
    common = ['--mode', 'mixture', '--mixture-stage', 'fit', '--root', str(result), '--prepared', str(prepared),
              '--mixture-seeds', '0', '--mixture-iterations', '15']
    first, second = tmp_path / 'first', tmp_path / 'second'
    main(common + ['--output', str(first)])
    for path in (prepared / 'graphs/test').glob('*.npz'):
        with np.load(path) as saved:
            values = {key: saved[key] for key in saved.files}
        values['gold'] = ~values['gold']
        np.savez(path, **values)
    main(common + ['--output', str(second)])
    for name in ['model_0.npz', 'frozen_scores/test/test0.npz']:
        with np.load(first / name) as a, np.load(second / name) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
