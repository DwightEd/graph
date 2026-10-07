"""Frozen evaluation and explicitly supervised signal diagnostics, never fitting risk."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.flow_latent.data import digest_files
from experiments.flow_latent.evaluate import detailed_case, within_answer_auc, write_visualization
from experiments.token_backtrace.grounded_projection_data import write_json
from experiments.token_backtrace.grounded_projection_evaluate import load_annotations, metrics
from .run import BASELINES, KINDS, load_measurements


def paired_bootstrap(cases, scores, annotations, first, second, draws=1000):
    groups = []
    for source in sorted({c['source_id'] for c in cases}):
        rows = [c for c in cases if c['source_id'] == source]
        combined = []
        for case in rows:
            valid = np.asarray(annotations[case['id']]['valid_tokens'], bool)
            combined.append((np.asarray(annotations[case['id']]['labels'])[valid],
                scores[case['id']][first][valid], scores[case['id']][second][valid]))
        groups.append(tuple(np.concatenate([row[k] for row in combined]) for k in range(3)))
    generator = np.random.default_rng(42)
    differences = []
    for _ in range(draws):
        sampled = generator.integers(len(groups), size=len(groups))
        truth, a, b = [np.concatenate([groups[i][k] for i in sampled]) for k in range(3)]
        if 0 < truth.sum() < len(truth):
            differences.append([roc_auc_score(truth, a) - roc_auc_score(truth, b),
                average_precision_score(truth, a) - average_precision_score(truth, b)])
    interval = np.quantile(differences, [.025, .975], axis=0)
    return dict(first=first, second=second, draws=len(differences),
        auroc_ci95=interval[:, 0].tolist(), ap_ci95=interval[:, 1].tolist())


def onset_metrics(cases, scores, annotations, names):
    truth, onset, values = [], [], {name: [] for name in names}
    for case in cases:
        annotation = annotations[case['id']]
        valid = np.asarray(annotation['valid_tokens'], bool)
        truth.append(np.asarray(annotation['labels'], bool)[valid])
        onset.append(np.asarray(annotation['span_onsets'], bool)[valid])
        for name in names:
            values[name].append(scores[case['id']][name][valid])
    truth, onset = np.concatenate(truth), np.concatenate(onset)
    at_onset = ~truth | onset
    continuation = ~truth | (truth & ~onset)
    result = {}
    for name in names:
        risk = np.concatenate(values[name])
        result[name] = dict(onset_auroc=float(roc_auc_score(truth[at_onset], risk[at_onset])),
            continuation_auroc=float(roc_auc_score(truth[continuation], risk[continuation])))
    return result


def supervised_diagnostic(inputs, records, cases, annotations):
    """Fixed-C probes on disjoint fit sources. Weights are not used by the detector."""
    fitting = [case for case in inputs['cases'] if case['cohort'] == 'fit']
    fit_annotations = load_annotations(dict(inputs, cases=fitting))
    task_index = {name: i for i, name in enumerate(('QA', 'Summary', 'Data2txt'))}

    def arrays(selected, gold, signed):
        features, truths = [], []
        for case in selected:
            record = records[case['id']]
            valid = np.asarray(gold[case['id']]['valid_tokens'], bool)
            position = np.arange(len(valid)) / max(1, len(valid) - 1)
            task = np.tile(np.eye(3)[task_index[case['task']]], (len(valid), 1))
            base = np.column_stack((record['nll'], position, task))
            if signed != 'base':
                base = np.column_stack((base, record['source_gap']))
            if signed == 'signed':
                base = np.column_stack((base, record['signed'].reshape(len(valid), -1)))
            if signed == 'abs':
                base = np.column_stack((base, np.abs(record['signed']).reshape(len(valid), -1)))
            features.append(base[valid])
            truths.append(np.asarray(gold[case['id']]['labels'])[valid])
        return np.concatenate(features), np.concatenate(truths)

    diagnostics, predictions = {}, {}
    for name in ('base', 'source', 'signed', 'abs'):
        train, train_truth = arrays(fitting, fit_annotations, name)
        test, test_truth = arrays(cases, annotations, name)
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced',
            max_iter=1500, solver='lbfgs', random_state=42))
        probe.fit(train, train_truth)
        prediction = probe.predict_proba(test)[:, 1]
        diagnostics[name] = dict(auroc=float(roc_auc_score(test_truth, prediction)),
            ap=float(average_precision_score(test_truth, prediction)), dimensions=train.shape[1],
            converged=int(probe[-1].n_iter_[0]) < 1500)
        begin = 0
        for case in cases:
            valid = np.asarray(annotations[case['id']]['valid_tokens'], bool)
            values = np.zeros(len(valid))
            values[valid] = prediction[begin:begin + int(valid.sum())]
            predictions.setdefault(case['id'], {})[name] = values
            begin += int(valid.sum())
    comparisons = [paired_bootstrap(cases, predictions, annotations, 'signed', name)
                   for name in ('source', 'abs')]
    return dict(supervised=True, diagnostic_only=True, natural_labels_used=True,
        weights_transferred_to_detector=False, C=.1, train_answers=len(fitting),
        results=diagnostics, comparisons=comparisons)


def evaluate(output, measurement, diagnostic=False):
    frozen = json.loads((output / 'frozen_scores.json').read_text())
    assert digest_files([Path(path) for path in frozen]) == frozen
    assert json.loads((output / 'execution.json').read_text())['status'] == 'DONE'
    inputs = json.loads((output / 'inputs.json').read_text())
    cases = [c for c in inputs['cases'] if c['cohort'] == 'regression']
    protocol = json.loads((output / 'protocol.json').read_text())
    kinds = tuple(protocol.get('kinds', KINDS))
    names = BASELINES + kinds
    scores = {c['id']: dict(np.load(output / c['id'] / 'scores.npz')) for c in inputs['cases']}
    thresholds = json.loads((output / 'thresholds.json').read_text())
    legacy = output / 'legacy_baseline.npz'
    if legacy.exists():
        legacy_freeze = json.loads((output / 'legacy_baseline_freeze.json').read_text())
        assert digest_files([legacy])[str(legacy)] == legacy_freeze['hashes'][str(legacy)]
        with np.load(legacy) as saved:
            historical = {case['id']: saved[case['id']] for case in inputs['cases']}
        for task in thresholds:
            reference_cases = [c for c in inputs['cases'] if c['task'] == task and c['cohort'] == 'reference']
            reference = np.sort(np.concatenate([historical[c['id']] for c in reference_cases]))
            for case in [c for c in inputs['cases'] if c['task'] == task]:
                values = historical[case['id']]
                scores[case['id']]['legacy_fixed'] = (np.searchsorted(reference, values, 'left')
                    + np.searchsorted(reference, values, 'right')) / (2 * len(reference))
            mixed = np.concatenate([scores[c['id']]['legacy_fixed'] for c in reference_cases])
            thresholds[task]['legacy_fixed'] = float(np.quantile(mixed, .95))
        names += ('legacy_fixed',)
    annotations = load_annotations(dict(inputs, cases=cases))
    write_json(output / 'evaluation_annotations.json', annotations)
    result = dict(scope='old exploratory source-disjoint regression; not blind confirmation',
        unsupervised_fit=True, metrics={name: metrics(cases, scores, annotations, thresholds, name) for name in names},
        within_answer={name: within_answer_auc(cases, scores, annotations, name) for name in names},
        onset=onset_metrics(cases, scores, annotations, names),
        comparisons=[paired_bootstrap(cases, scores, annotations, 'native', name)
                     for name in (('node', 'chain', 'rewired', 'source_raw', 'source_unit', 'legacy_fixed')
                                  if legacy.exists() else ('node', 'chain', 'rewired', 'source_raw', 'source_unit'))],
        cases=[detailed_case(c, scores[c['id']], annotations[c['id']], thresholds[c['task']], names)
               for c in cases])
    raw_names = BASELINES + kinds
    raw_thresholds = {}
    for task in thresholds:
        reference_cases = [c for c in inputs['cases'] if c['task'] == task and c['cohort'] == 'reference']
        raw_thresholds[task] = {'raw_' + name: float(np.quantile(np.concatenate([
            scores[c['id']]['raw_' + name] for c in reference_cases]), .95)) for name in raw_names}
    result['raw_metrics'] = {name: metrics(cases, scores, annotations, raw_thresholds, 'raw_' + name)
                             for name in raw_names}
    result['seeds'] = {}
    seeds = protocol['seeds']
    for seed in seeds:
        seed_scores = {}
        seed_thresholds = {task: {} for task in thresholds}
        for case in inputs['cases']:
            seed_scores[case['id']] = {}
        for task in thresholds:
            reference_cases = [c for c in inputs['cases'] if c['task'] == task and c['cohort'] == 'reference']
            for name in kinds:
                reference = np.sort(np.concatenate([scores[c['id']][f'{name}_{seed}_score'] for c in reference_cases]))
                for case in [c for c in inputs['cases'] if c['task'] == task]:
                    values = scores[case['id']][f'{name}_{seed}_score']
                    seed_scores[case['id']][name] = (np.searchsorted(reference, values, 'left')
                        + np.searchsorted(reference, values, 'right')) / (2 * len(reference))
                mixed = np.concatenate([seed_scores[c['id']][name] for c in reference_cases])
                seed_thresholds[task][name] = float(np.quantile(mixed, .95))
        result['seeds'][str(seed)] = {name: metrics(cases, seed_scores, annotations, seed_thresholds, name) for name in kinds}
    if 'edge_node' in kinds:
        result['comparisons'].append(paired_bootstrap(cases, scores, annotations, 'native', 'edge_node'))
    rewire = json.loads((output / 'rewire.json').read_text())
    nulls = [row for key, rows in rewire.items() if key.startswith('rewired') for row in rows.values()]
    result['null_strength'] = dict(valid_cases=sum(row['valid_null'] for row in nulls), total=len(nulls),
        adopted_mass_reassigned_mean=float(np.mean([row['adopted_mass_reassigned'] for row in nulls])),
        adopted_mass_reassigned_range=[min(row['adopted_mass_reassigned'] for row in nulls),
                                      max(row['adopted_mass_reassigned'] for row in nulls)])
    write_json(output / 'results.json', result)
    write_visualization(output, cases, scores, annotations, thresholds, names)
    if diagnostic:
        _, records = load_measurements(measurement)
        write_json(output / 'supervised_diagnostic.json', supervised_diagnostic(inputs, records, cases, annotations))
    print(json.dumps(result['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    arguments = parser.parse_args()
    evaluate(arguments.output, arguments.measurement, arguments.diagnostic)
