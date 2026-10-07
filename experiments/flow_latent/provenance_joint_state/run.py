"""Unlabelled predictive rank selection, joint fitting, and immutable token scores."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from experiments.flow_latent.data import digest_files
from experiments.token_backtrace.grounded_projection_data import write_json

from .features import (fit_features, initial_parameters, local_mean,
                       text_unit_mean, transform)
from .gaussian import covariance_block, fit, posterior


KINDS = ('node', 'chain', 'native', 'rewired')
BASELINES = ('source_raw', 'source_local', 'source_unit', 'nll')


def load_measurements(measurement, source_gaps=None):
    inputs = json.loads((measurement / 'inputs.json').read_text())
    assert json.loads((measurement / 'execution.json').read_text())['status'] == 'DONE'
    records = {}
    for case in inputs['cases']:
        identity = case['id']
        metadata = json.loads((measurement / identity / 'measurement.json').read_text())
        assert metadata['all_tokens_measured'], identity
        with np.load(measurement / identity / 'observations.npz') as saved:
            records[identity] = {name: saved[name] for name in (
                'rows', 'token_id', 'source_gap', 'signed', 'edge_signed', 'edge_mass',
                'self_diagonal', 'nll')}
        assert np.array_equal(records[identity]['token_id'], case['response']['answer_ids'])
    cohorts = {kind: {c['source_id'] for c in inputs['cases'] if c['cohort'] == kind}
               for kind in ('fit', 'reference', 'regression')}
    assert not cohorts['fit'] & (cohorts['reference'] | cohorts['regression'])
    assert not cohorts['reference'] & cohorts['regression']
    if source_gaps:
        with np.load(source_gaps) as saved:
            for identity, record in records.items():
                record['source_gap_native'] = record['source_gap']
                record['source_gap'] = saved[identity].mean(1)
    return inputs, records


def train_model(cases, records, feature_map, kind, rank, seed, iterations, head_weight=1.):
    sequences = [transform(records[c['id']], feature_map, kind, seed)[:2] for c in cases]
    matrix = np.concatenate([sequence[0] for sequence in sequences])
    initial = initial_parameters(matrix, rank, sequences[0][1].shape[2], seed, .01 * len(matrix), head_weight)
    parameters, diagnostics = fit(sequences, initial, iterations=iterations)
    return parameters, diagnostics


def predictive_anchor(cases, records, feature_map, kind, parameters, seed, mask_mode='all'):
    logp, squared_error = [], []
    rank = parameters.loading.shape[1]
    for case in cases:
        matrix, edges, _ = transform(records[case['id']], feature_map, kind, seed)
        if mask_mode == 'sparse':
            phase = int(hashlib.sha256(case['id'].encode()).hexdigest()[:8], 16) % 5
            hidden = np.arange(len(matrix)) % 5 == phase
        else:
            hidden = np.ones(len(matrix), dtype=bool)
        mean, covariance, _ = posterior(matrix, edges, parameters, hide_anchor=hidden)
        variance = np.array([covariance_block(covariance, t, t, rank)[0, 0]
                             for t in range(len(mean))]) + parameters.noise[0]
        residual = matrix[hidden, 0] - mean[hidden, 0]
        variance = variance[hidden]
        logp.extend(-.5 * (np.log(2 * np.pi * variance) + np.square(residual) / variance))
        squared_error.extend(np.square(residual))
    return dict(mean_logp=float(np.mean(logp)), mse=float(np.mean(squared_error)), tokens=len(logp))


def choose_rank(cases, records, ranks, iterations, encoder, view='full', head_weight=1., mask_mode='all'):
    # All rank selection is confined to source-disjoint, annotation-free fit data.
    ordered = sorted(cases, key=lambda c: hashlib.sha256(('joint-predictive:' + c['source_id']).encode()).hexdigest())
    validation = ordered[::4]
    excluded = {c['source_id'] for c in validation}
    training = [c for c in ordered if c['source_id'] not in excluded]
    feature_map = fit_features([records[c['id']] for c in training], encoder, view)
    results = []
    for rank in ranks:
        parameters, diagnostics = train_model(training, records, feature_map, 'native', rank, 42, iterations, head_weight)
        prediction = predictive_anchor(validation, records, feature_map, 'native', parameters, 42, mask_mode)
        results.append(dict(rank=rank, **prediction, fitting=diagnostics))
        print('PREDICTIVE', rank, prediction, flush=True)
    selected = max(results, key=lambda row: row['mean_logp'])['rank']
    return selected, dict(results=results, fit_ids=[c['id'] for c in training],
        validation_ids=[c['id'] for c in validation], labels_used=False,
        criterion='source-gap predictive likelihood', mask_mode=mask_mode)


def scores_for_model(inputs, records, feature_map, parameters, kind, seed):
    scores, diagnostics = {}, {}
    for case in inputs['cases']:
        identity = case['id']
        matrix, edges, rewire = transform(records[identity], feature_map, kind, seed)
        mean, covariance, _ = posterior(matrix, edges, parameters)
        score = mean[:, 0] * feature_map.scale[0] + feature_map.center[0]
        variance = covariance[0, np.arange(len(mean)) * mean.shape[1]] * feature_map.scale[0] ** 2
        scores[identity] = dict(score=score, posterior_mean=mean, source_variance=variance)
        if rewire:
            diagnostics[identity] = rewire
    return scores, diagnostics


def baseline_scores(case, record):
    return dict(source_raw=record['source_gap'], source_local=local_mean(record['source_gap']),
        source_unit=text_unit_mean(record['source_gap'], case['response']['token_text']), nll=record['nll'])


def calibrated_output(inputs, records, all_scores):
    names = BASELINES + tuple(all_scores)
    scores = {}
    for case in inputs['cases']:
        identity = case['id']
        values = baseline_scores(case, records[identity])
        values.update({kind: all_scores[kind][identity]['score'] for kind in all_scores})
        scores[identity] = values
    thresholds, references = {}, {}
    for task in ('QA', 'Summary', 'Data2txt'):
        cases = [c for c in inputs['cases'] if c['cohort'] == 'reference' and c['task'] == task]
        references[task] = {name: np.sort(np.concatenate([scores[c['id']][name] for c in cases]))
                            for name in names}
        thresholds[task] = {}
        for name in names:
            reference = references[task][name]
            percentile = (np.searchsorted(reference, reference, side='left')
                + np.searchsorted(reference, reference, side='right')) / (2 * len(reference))
            thresholds[task][name] = float(np.quantile(percentile, .95))
    for case in inputs['cases']:
        for name in names:
            reference = references[case['task']][name]
            values = scores[case['id']][name]
            scores[case['id']]['raw_' + name] = values
            scores[case['id']][name] = (np.searchsorted(reference, values, side='left')
                + np.searchsorted(reference, values, side='right')) / (2 * len(reference))
    return scores, thresholds


def load_bound_models(directory, inputs, protocol, feature_map, rank, continuation):
    """Reuse only models fitted on the same observations, weights and identities."""
    previous = json.loads((directory / 'protocol.json').read_text())
    frozen = json.loads((directory / 'frozen_scores.json').read_text())
    assert digest_files([Path(path) for path in frozen]) == frozen
    assert json.loads((directory / 'inputs.json').read_text()) == inputs
    assert json.loads((directory / 'rank_selection.json').read_text())['selected'] == rank
    # v1/v2 used full observations with unit head weights; those keys were not saved.
    defaults = {'view': 'full', 'head_weight': 1., 'source_gaps': None}
    for key in ('seeds', 'view', 'head_weight', 'source_gaps'):
        assert previous.get(key, defaults.get(key)) == protocol[key], key
    # Code revisions may differ; all measurement bytes must be identical.
    for path, digest in protocol['hashes'].items():
        if path.endswith('/observations.npz') or path.endswith('/inputs.json') or path == protocol['source_gaps']:
            assert previous['hashes'][path] == digest, path
    if continuation:
        assert previous['edge_encoder'] == protocol['edge_encoder']
    else:
        assert previous['iterations'] == protocol['iterations']
    with (directory / 'models.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    names = ('center', 'scale')
    if continuation:
        names += ('signed_basis', 'mass_basis', 'edge_scale', 'local_center', 'local_scale')
    for name in names:
        assert np.array_equal(getattr(saved['feature_map'], name), getattr(feature_map, name)), name
    saved['fitting'] = json.loads((directory / 'fitting.json').read_text())
    return saved


def optimize_model(cases, records, feature_map, kind, rank, seed, iterations,
                   head_weight, reused, continued):
    """Fit a new model or continue immutable parameters without label-based stopping."""
    key = f'{kind}_{seed}'
    if continued:
        parameters = continued['models'][key]
        previous = continued['fitting'][key]
        if previous['converged']:
            return parameters, dict(previous, reused_converged_model=True), 0
        sequences = [transform(records[c['id']], feature_map, kind, seed)[:2] for c in cases]
        parameters, diagnostic = fit(sequences, parameters, iterations=iterations)
        assert np.isclose(diagnostic['objective'][0], previous['objective'][-1])
        diagnostic['previous_iterations'] = previous['iterations']
        diagnostic['continued_from_frozen_model'] = True
        return parameters, diagnostic, 1
    if reused and kind in ('node', 'chain'):
        return reused['models'][key], dict(reused['fitting'][key], reused_frozen_model=True), 0
    parameters, diagnostic = train_model(cases, records, feature_map, kind, rank, seed, iterations, head_weight)
    return parameters, diagnostic, 1


def fit_and_score(measurement, output, ranks, seeds, iterations, encoder='svd', reuse=None,
                  source_gaps=None, view='full', head_weight=1., mask_mode='all', continuation=None):
    started = time.time()
    inputs, records = load_measurements(measurement, source_gaps)
    kinds = KINDS + (('edge_node',) if encoder == 'anchored' else ())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'inputs.json', inputs)
    files = list(Path(__file__).parent.glob('*.py')) + [measurement / 'inputs.json']
    files += [measurement / c['id'] / 'observations.npz' for c in inputs['cases']]
    if source_gaps:
        files.append(source_gaps)
    freeze = dict(hashes=digest_files(files), ranks=ranks, seeds=seeds, iterations=iterations,
        observation=('source-gap + 1024 post-token self diagonal' if view == 'self'
            else 'source-gap + 3x1024 signed adoption + 1024 post-token self diagonal'),
        fixed_anchor='loading[0]=(1,0,...); posterior source-coordinate mean',
        risk_is_not_density=True, labels_used=False, noise_floor=.02,
        prior='.01*N matrix-normal coefficient prior conditional on each row noise',
        inference='exact joint Gaussian; banded Cholesky + selected inverse',
        null='head/lag-group matched message descriptors; no exact BPE match',
        edge_encoder=encoder, kinds=kinds, reuse_node_chain=str(reuse) if reuse else None,
        source_gaps=str(source_gaps) if source_gaps else None, view=view, head_weight=head_weight,
        mask_mode=mask_mode, auxiliary_source_world=bool(source_gaps),
        generalized_bayes=head_weight != 1.,
        continuation=str(continuation) if continuation else None,
        scope='existing exploratory fit/reference/regression; no blind confirmation')
    write_json(output / 'protocol.json', freeze)
    snapshot = output / 'code_snapshot'
    snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        (snapshot / path.name).write_bytes(path.read_bytes())
    fitting = [c for c in inputs['cases'] if c['cohort'] == 'fit']
    rank, rank_selection = choose_rank(fitting, records, ranks, iterations, encoder, view, head_weight, mask_mode)
    write_json(output / 'rank_selection.json', dict(selected=rank, **rank_selection))
    feature_map = fit_features([records[c['id']] for c in fitting], encoder, view)
    reused = load_bound_models(reuse, inputs, freeze, feature_map, rank, continuation=False) if reuse else None
    continued = load_bound_models(continuation, inputs, freeze, feature_map, rank, continuation=True) if continuation else None
    models, diagnostics, all_scores, rewires = {}, {}, {}, {}
    optimized_models = len(ranks)
    for seed in seeds:
        for kind in kinds:
            parameters, diagnostic, optimized = optimize_model(
                fitting, records, feature_map, kind, rank, seed, iterations, head_weight,
                reused, continued)
            optimized_models += optimized
            models[f'{kind}_{seed}'] = parameters
            diagnostics[f'{kind}_{seed}'] = diagnostic
            scored, rewire = scores_for_model(inputs, records, feature_map, parameters, kind, seed)
            all_scores.setdefault(kind, []).append(scored)
            rewires[f'{kind}_{seed}'] = rewire
            print('FIT', kind, seed, diagnostic['iterations'], diagnostic['converged'], flush=True)
    with (output / 'models.pkl').open('wb') as handle:
        pickle.dump(dict(models=models, feature_map=feature_map), handle)
    write_json(output / 'fitting.json', diagnostics)
    write_json(output / 'rewire.json', rewires)
    means = {kind: {c['id']: {'score': np.mean([s[c['id']]['score'] for s in sets], axis=0)}
                   for c in inputs['cases']} for kind, sets in all_scores.items()}
    scores, thresholds = calibrated_output(inputs, records, means)
    write_json(output / 'thresholds.json', thresholds)
    paths = [output / 'models.pkl', output / 'protocol.json', output / 'inputs.json',
        output / 'rank_selection.json', output / 'fitting.json', output / 'thresholds.json', output / 'rewire.json']
    for case in inputs['cases']:
        directory = output / case['id']
        directory.mkdir()
        extras = {f'{kind}_{seed}_{key}': all_scores[kind][i][case['id']][key]
                  for i, seed in enumerate(seeds) for kind in kinds
                  for key in ('score', 'posterior_mean', 'source_variance')}
        np.savez_compressed(directory / 'scores.npz', token_id=records[case['id']]['token_id'],
            **scores[case['id']], **extras)
        paths.append(directory / 'scores.npz')
    write_json(output / 'frozen_scores.json', digest_files(paths))
    write_json(output / 'execution.json', dict(status='DONE', selected_rank=rank,
        fitted_models=optimized_models, reused_models=len(models) + len(ranks) - optimized_models,
        seconds=time.time() - started, labels_used=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--encoder', choices=('svd', 'anchored'), default='svd')
    parser.add_argument('--reuse', type=Path)
    parser.add_argument('--source-gaps', type=Path)
    parser.add_argument('--view', choices=('full', 'self'), default='full')
    parser.add_argument('--head-weight', type=float, default=1.)
    parser.add_argument('--mask-mode', choices=('all', 'sparse'), default='all')
    parser.add_argument('--continue-from', type=Path)
    args = parser.parse_args()
    fit_and_score(args.measurement, args.output, args.ranks, args.seeds, args.iterations,
                  args.encoder, args.reuse, args.source_gaps, args.view, args.head_weight, args.mask_mode,
                  args.continue_from)


if __name__ == '__main__':
    main()
