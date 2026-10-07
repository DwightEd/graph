"""Independent rank metrics, persisted raw graph witnesses and source split checks."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import rankdata, multivariate_normal
from scipy.special import logsumexp
from scipy.linalg import solve_triangular
from experiments.token_backtrace.grounded_projection_data import write_json
from .flow import coordinate_projection
from .data import digest_files


def rank_metrics(truth, score):
    positive = int(truth.sum())
    negative = len(truth) - positive
    ranks = rankdata(score)
    auc = (ranks[truth].sum() - positive * (positive + 1) / 2) / (positive * negative)
    order = np.argsort(-score, kind='stable')
    cumulative = np.cumsum(truth[order])
    boundaries = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(score) - 1]
    recall = cumulative[boundaries] / positive
    precision = cumulative[boundaries] / (boundaries + 1)
    ap = np.sum(np.diff(np.r_[0., recall]) * precision)
    return float(auc), float(ap)


def raw_witness(capture, case):
    directory = capture / case['id']
    weights = np.load(directory / 'attention.npy', mmap_mode='r')
    values = np.load(directory / 'values.npy', mmap_mode='r')
    nodes = np.load(directory / 'node.npy', mmap_mode='r')
    stored = np.load(directory / 'features.npz')
    prompt = len(case['source']['prompt_with_source'])
    errors, head_errors = [], []
    forbidden = 0.
    for layer in (0, 15, 31):
        for target in sorted({0, len(nodes) // 2, len(nodes) - 1}):
            row = weights[layer, target].astype(np.float64)
            forbidden = max(forbidden, float(row[:, prompt + target:].max()))
            source = np.zeros(values.shape[0], bool)
            source[:prompt] = case['source']['source_mask']
            projected = values[:, layer].astype(np.float64) @ coordinate_projection()
            message = np.einsum('hk,khd->hd', row[:, source], projected[source])
            expected = stored['native_source'][target, layer]
            relative = np.linalg.norm(message - expected) / max(np.linalg.norm(expected), 1e-12)
            errors.append(float(relative))
            if target > 0:
                message = np.einsum('hk,khd->hd', row, values[:, layer].astype(np.float64))
                expected = nodes[target - 1, layer].astype(np.float64)
                relative = np.linalg.norm(message - expected) / max(np.linalg.norm(expected), 1e-12)
                head_errors.append(float(relative))
    assert forbidden == 0, 'Future key edge in persisted predictor graph'
    assert max(errors) < .005, 'Persisted float16 graph/value witness mismatch'
    assert max(head_errors) < .005, 'Persisted complete head message mismatch'
    return dict(relative_source_message_max=max(errors), relative_full_head_max=max(head_errors),
                future_mass_max=forbidden)


def density_witness(output, case, kind):
    from .run import features
    protocol = json.loads((output / 'protocol.json').read_text())
    capture = Path(protocol['raw_capture'])
    with (output / (kind + '_model.pkl')).open('rb') as stream:
        fitted = pickle.load(stream)
    nodes, context = features(capture, case, kind, protocol['version'])
    rows = [0, len(nodes) // 2, len(nodes) - 1]
    projection = fitted['node_projection']
    nodes = (nodes[rows] - projection['mean']) @ projection['axes'].T / projection['scale']
    if fitted['context_projection'] is not None:
        projection = fitted['context_projection']
        context = (context[rows] - projection['mean']) @ projection['axes'].T / projection['scale']
    else:
        context = context[rows]
    augmented = np.column_stack((np.ones(len(rows)), context))
    model = fitted['model']
    terms = []
    for index, coefficient in enumerate(model['coefficients']):
        residual = nodes - augmented @ coefficient
        terms.append(multivariate_normal.logpdf(residual, mean=np.zeros(nodes.shape[1]),
            cov=model['covariances'][index]) + np.log(model['weights'][index]))
    expected = -logsumexp(np.column_stack(terms), axis=1)
    actual = np.load(output / case['id'] / 'scores.npz')[kind][rows]
    error = float(np.max(np.abs(expected - actual)))
    assert error < 1e-8, 'Independent fitted Gaussian density mismatch'
    return error


def derived_witness(capture, derived, cases):
    axes = np.load(derived / 'axes.npy').astype(np.float64)
    gram = np.einsum('lhdr,lhds->lhrs', axes, axes)
    assert np.max(np.abs(gram - np.eye(16))) < 5e-7
    errors, path_errors = [], []
    for case in cases:
        if case['id'] not in ('15604', '12219', '12297', '9022'):
            continue
        weights = np.load(capture / case['id'] / 'attention.npy', mmap_mode='r')
        values = np.load(capture / case['id'] / 'values.npy', mmap_mode='r')
        features = np.load(derived / case['id'] / 'features.npz')
        prompt = len(case['source']['prompt_with_source'])
        source = np.zeros(values.shape[0], bool)
        source[:prompt] = case['source']['source_mask']
        for layer in (0, 15, 31):
            target = len(features['node']) // 2
            row = weights[layer, target].astype(np.float64)
            message = np.einsum('hk,khd->hd', row[:, source], values[source, layer].astype(np.float64))
            expected = np.einsum('hd,hdr->hr', message, axes[layer])
            actual = features['native_source'][target, layer]
            error = np.linalg.norm(expected - actual) / np.linalg.norm(expected)
            errors.append(float(error))
            head = 15
            graph = weights[layer, :, head, prompt:].astype(np.float64)
            operator = np.eye(len(graph)) - .5 * graph
            expected = solve_triangular(operator, features['native_source'][:, layer, head],
                                        lower=True, unit_diagonal=True)
            actual = features['native_root'][:, layer, head]
            error = np.linalg.norm(expected - actual) / np.linalg.norm(expected)
            path_errors.append(float(error))
    assert max(errors + path_errors) < 1e-5
    return dict(projection_orthogonal=True, source_message_relative_max=max(errors),
                finite_path_relative_max=max(path_errors))


def cohort_retention(capture, derived, cases, node_projection):
    """Token-weighted variance, centered within each cohort; axes only, no whitening."""
    axes = np.load(derived / 'axes.npy').astype(np.float64)
    final_axes = node_projection['axes']
    result = {}
    for cohort in ('fit', 'reference', 'regression'):
        count = 0
        sums = [np.zeros(32 * 32 * width) for width in (128, 16)]
        sums.append(np.zeros(len(final_axes)))
        squares = np.zeros(3)
        for case in cases:
            if case['cohort'] != cohort:
                continue
            raw = np.load(capture / case['id'] / 'node.npy', mmap_mode='r')
            for start in range(0, len(raw), 32):
                full = raw[start:start + 32].astype(np.float64)
                projected = np.einsum('tlhd,lhdr->tlhr', full, axes, optimize=True)
                vectors = [full.reshape(len(full), -1), projected.reshape(len(full), -1)]
                vectors.append(vectors[1] @ final_axes.T)
                count += len(full)
                for index, vector in enumerate(vectors):
                    sums[index] += vector.sum(axis=0)
                    squares[index] += np.square(vector).sum()
        variance = [(squares[i] - np.square(sums[i]).sum() / count) / count for i in range(3)]
        result[cohort] = dict(tokens=count, raw_variance=variance[0], head16_variance=variance[1],
            final128_variance=variance[2], head16_fraction=variance[1] / variance[0],
            final128_fraction=variance[2] / variance[0])
    result['message_retention'] = 'unmeasured; node retention does not estimate source/history/root retention'
    return result


def control_contract(capture, output, cases):
    """Validate exact base arrays, cache binding and parameter capacity for added controls."""
    import hashlib
    protocol = json.loads((output / 'protocol.json').read_text())
    base = Path(protocol['base_scores'])
    assert Path(protocol['raw_capture']).resolve() == capture.resolve()
    assert (capture / 'inputs.json').read_bytes() == (base / 'inputs.json').read_bytes()
    assert protocol['cache_manifest_hash'] == digest_files([capture / 'content_manifest.json'])
    assert protocol['base_native_model_sha256'] == hashlib.sha256((base / 'native_model.pkl').read_bytes()).hexdigest()
    assert protocol['base_frozen_manifest_sha256'] == digest_files([base / 'frozen_scores.json'])
    with (base / 'native_model.pkl').open('rb') as stream:
        native = pickle.load(stream)
    with (output / 'lag_pair_model.pkl').open('rb') as stream:
        control = pickle.load(stream)
    for key in ('coefficients', 'covariances', 'weights', 'loadings', 'noise'):
        assert np.shape(native['model'][key]) == np.shape(control['model'][key])
    assert native['context_projection']['axes'].shape == control['context_projection']['axes'].shape
    for case in cases:
        before = np.load(base / case['id'] / 'scores.npz')
        after = np.load(output / case['id'] / 'scores.npz')
        assert all(np.array_equal(before[key], after[key]) for key in before.files)
    return dict(base_scores_identical=True, base_model_bound=True, derived_cache_bound=True,
                capacity_matched=True)


def audit(capture, outputs):
    inputs = json.loads((capture / 'inputs.json').read_text())
    cases = inputs['cases']
    sources = [case['source_id'] for case in cases]
    assert len(set(sources)) == len(cases)
    counts = {group: sum(c['cohort'] == group for c in cases)
              for group in ('fit', 'reference', 'regression')}
    assert counts == dict(fit=24, reference=12, regression=17)
    witnesses = {case['id']: raw_witness(capture, case) for case in cases
                 if case['id'] in ('15604', '12219', '12297', '9022')}
    derived = capture.parent / "flow_latent_20261007_head16"
    numeric, density_checks, controls = [], [], {}
    for output in outputs:
        annotations = json.loads((output / 'evaluation_annotations.json').read_text())
        result = json.loads((output / 'results.json').read_text())
        frozen = json.loads((output / 'frozen_scores.json').read_text())
        assert digest_files([Path(name) for name in frozen]) == frozen
        if (output / 'lag_pair_model.pkl').exists():
            protocol = json.loads((output / 'protocol.json').read_text())
            if 'base_native_model_sha256' in protocol:
                controls[str(output)] = control_contract(derived, output, cases)
            else:
                controls[str(output)] = dict(binding='initial control before provenance fix; preserved')
        regression = [c for c in cases if c['cohort'] == 'regression']
        density_names = ('lag_pair',) if (output / 'lag_pair_model.pkl').exists() else ('node', 'native', 'rewired')
        for kind in density_names:
            error = density_witness(output, regression[0], kind)
            density_checks.append(dict(output=str(output), method=kind, max_error=error))
        for name in result['metrics']:
            truths, risks = [], []
            for case in regression:
                valid = np.asarray(annotations[case['id']]['valid_tokens'], bool)
                truth = np.asarray(annotations[case['id']]['labels'], bool)
                score = np.load(output / case['id'] / 'scores.npz')[name]
                truths.append(truth[valid])
                risks.append(score[valid])
            auc, ap = rank_metrics(np.concatenate(truths), np.concatenate(risks))
            expected = result['metrics'][name]
            assert abs(auc - expected['auroc']) < 1e-12
            assert abs(ap - expected['ap']) < 1e-12
            numeric.append(dict(output=str(output), method=name, auroc=auc, ap=ap))
    report = dict(status='PASS', source_disjoint=True, cohorts=counts,
                  raw_witnesses=witnesses, control_contracts=controls, independent_metric_checks=numeric,
                  independent_fitted_density_checks=density_checks)
    derived = capture.parent / 'flow_latent_20261007_head16'
    if (derived / 'execution.json').exists():
        report['derived_witness'] = derived_witness(capture, derived, cases)
        with (capture.parent / 'flow_latent_20261007_head16_v3' / 'node_model.pkl').open('rb') as stream:
            projection = pickle.load(stream)['node_projection']
        report['cohort_node_retention'] = cohort_retention(capture, derived, cases, projection)
    write_json(capture / 'numeric_audit.json', report)
    print(json.dumps(dict(status=report['status'], metric_checks=len(numeric),
        density_checks=len(density_checks), retention=report.get('cohort_node_retention')), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    audit(args.capture, args.outputs)
