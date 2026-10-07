"""Independent tie-aware metrics and frozen-model posterior reproduction on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from .features import transform
from .gaussian import posterior
from .run import load_bound_models, load_measurements


def read_json(path):
    return json.loads(path.read_text())


def rank_metrics(truth, score):
    """Group exact ties: pair-count AUC and stepwise precision-recall AP."""
    order = np.argsort(score, kind='stable')
    score, truth = score[order], truth[order]
    starts = np.r_[0, np.flatnonzero(np.diff(score)) + 1]
    positives = np.add.reduceat(truth.astype(float), starts)
    counts = np.diff(np.r_[starts, len(truth)])
    negatives = counts - positives
    below = np.cumsum(negatives) - negatives
    auc = (positives * (below + .5 * negatives)).sum() / (truth.sum() * (~truth).sum())
    precision = np.cumsum(positives[::-1]) / np.cumsum(counts[::-1])
    ap = (precision * positives[::-1]).sum() / truth.sum()
    return dict(auroc=float(auc), ap=float(ap))


def verify_hashes(mapping, snapshots=None):
    for filename, expected in mapping.items():
        path = Path(filename)
        if snapshots and filename.endswith('.py'):
            path = snapshots / path.name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    return len(mapping)


def reference_ranks(inputs, values):
    result = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        selected = [c for c in inputs['cases'] if c['task'] == task]
        reference = np.sort(np.concatenate([values[c['id']] for c in selected if c['cohort'] == 'reference']))
        for case in selected:
            score = values[case['id']]
            result[case['id']] = (np.searchsorted(reference, score, 'left')
                + np.searchsorted(reference, score, 'right')) / (2 * len(reference))
    return result


def check_metric(cases, annotations, values, expected):
    labels, scores = [], []
    for case in cases:
        annotation = annotations[case['id']]
        valid = np.asarray(annotation['valid_tokens'], bool)
        labels.append(np.asarray(annotation['labels'], bool)[valid])
        scores.append(values[case['id']][valid])
    actual = rank_metrics(np.concatenate(labels), np.concatenate(scores))
    for key in actual:
        assert abs(actual[key] - expected[key]) < 1e-12, (key, actual, expected)


def audit_run(output, measurement, replay_ids):
    protocol = read_json(output / 'protocol.json')
    count = verify_hashes(read_json(output / 'frozen_scores.json'))
    count += verify_hashes(protocol['hashes'], output / 'code_snapshot')
    inputs, records = load_measurements(measurement, protocol.get('source_gaps'))
    annotations = read_json(output / 'evaluation_annotations.json')
    cases = [c for c in inputs['cases'] if c['cohort'] == 'regression']
    scores = {c['id']: dict(np.load(output / c['id'] / 'scores.npz')) for c in inputs['cases']}
    results = read_json(output / 'results.json')
    with (output / 'models.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    fitting = read_json(output / 'fitting.json')
    for key, diagnostic in fitting.items():
        trace = np.asarray(diagnostic['objective'])
        assert (np.diff(trace) >= -1e-7 * np.maximum(1, np.abs(trace[:-1]))).all(), key
    kinds = protocol.get('kinds', ('node', 'chain', 'native', 'rewired'))
    checked = 0
    for name in results['metrics']:
        if name == 'legacy_fixed':
            with np.load(output / 'legacy_baseline.npz') as legacy:
                values = reference_ranks(inputs, dict(legacy))
        else:
            values = {c['id']: scores[c['id']][name] for c in inputs['cases']}
        check_metric(cases, annotations, values, results['metrics'][name])
        checked += 1
    max_error = 0.
    for kind in kinds:
        ensemble = {c['id']: np.mean([scores[c['id']][f'{kind}_{seed}_score']
            for seed in protocol['seeds']], axis=0) for c in inputs['cases']}
        calibrated = reference_ranks(inputs, ensemble)
        for case in inputs['cases']:
            assert np.array_equal(ensemble[case['id']], scores[case['id']]['raw_' + kind])
            assert np.array_equal(calibrated[case['id']], scores[case['id']][kind])
        for seed in protocol['seeds']:
            values = {c['id']: scores[c['id']][f'{kind}_{seed}_score'] for c in inputs['cases']}
            check_metric(cases, annotations, reference_ranks(inputs, values), results['seeds'][str(seed)][kind])
            checked += 1
            parameters = saved['models'][f'{kind}_{seed}']
            assert np.array_equal(parameters.loading[0], np.eye(parameters.loading.shape[1])[0])
            for identity in replay_ids:
                matrix, edges, _ = transform(records[identity], saved['feature_map'], kind, seed)
                mean, _, _ = posterior(matrix, edges, parameters)
                error = float(np.max(np.abs(mean - scores[identity][f'{kind}_{seed}_posterior_mean'])))
                assert error < 1e-10, (output.name, kind, seed, identity, error)
                max_error = max(max_error, error)
    for name, expected in results['raw_metrics'].items():
        check_metric(cases, annotations, {c['id']: scores[c['id']]['raw_' + name] for c in inputs['cases']}, expected)
        checked += 1
    return dict(frozen_hashes_checked=count, metric_pairs_checked=checked,
        posterior_replays=len(kinds) * len(protocol['seeds']) * len(replay_ids),
        posterior_max_error=max_error, all_models_converged=all(d['converged'] for d in fitting.values()))


def check_continuation(previous, output, measurement):
    inputs, records = load_measurements(measurement, read_json(output / 'protocol.json')['source_gaps'])
    with (output / 'models.pkl').open('rb') as handle:
        current = pickle.load(handle)
    protocol = read_json(output / 'protocol.json')
    load_bound_models(previous, inputs, protocol, current['feature_map'], 2, continuation=True)
    old, new = read_json(previous / 'fitting.json'), read_json(output / 'fitting.json')
    for key in new:
        if new[key].get('continued_from_frozen_model'):
            assert abs(new[key]['objective'][0] - old[key]['objective'][-1]) < 1e-7
    # A changed measurement digest must reject reuse even if pooled statistics agree.
    poisoned = dict(protocol, hashes=dict(protocol['hashes']))
    filename = next(p for p in poisoned['hashes'] if p.endswith('/observations.npz'))
    poisoned['hashes'][filename] = '0' * 64
    try:
        load_bound_models(previous, inputs, poisoned, current['feature_map'], 2, continuation=True)
    except AssertionError:
        return dict(binding='PASS', modified_measurement_rejected=True)
    raise AssertionError('changed measurement bytes accepted for continuation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'PASS', 'metric_implementation': 'independent tie groups; no sklearn metric calls', 'runs': {}}
    for output in args.runs:
        report['runs'][output.name] = audit_run(output, args.measurement, ['12219', '12297', '12471', '17199'])
        print(output.name, report['runs'][output.name], flush=True)
    report['continuation'] = check_continuation(args.runs[-2], args.runs[-1], args.measurement)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
