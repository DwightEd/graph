"""Freeze and score all cases before the separate annotation evaluation stage."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
from experiments.token_backtrace.grounded_projection_data import write_json
from .data import digest_files
from .model import fit_projection, project, fit_mixture, conditional_log_density, factor_posterior


def features(capture, case, kind, version):
    stored = np.load(capture / case['id'] / 'features.npz')
    nodes = stored['node'].reshape(len(stored['node']), -1).astype(np.float64)
    if kind == 'node':
        context = np.empty((len(nodes), 0))
    elif kind == 'lag_pair':
        predictor = sum(stored['native_' + name] for name in ('source', 'history', 'other'))
        previous = predictor[np.maximum(np.arange(len(nodes)) - 1, 0)]
        context = np.concatenate((predictor.reshape(len(nodes), -1),
                                  previous.reshape(len(nodes), -1)), axis=1).astype(np.float64)
    else:
        source_name = 'source' if version == 1 else 'root'
        names = [source_name, 'history', 'other'] if version < 3 else [source_name, 'other']
        context = np.concatenate([stored[kind + '_' + name].reshape(len(nodes), -1)
                                  for name in names], axis=1).astype(np.float64)
    return nodes, context


def train(capture, cases, kind, version, seed, node_projection, context_dimensions, factor_rank):
    fit = [features(capture, case, kind, version) for case in cases if case['cohort'] == 'fit']
    nodes = project(np.concatenate([pair[0] for pair in fit]), node_projection)
    context = np.concatenate([pair[1] for pair in fit])
    context_projection = None
    if context.shape[1]:
        context_projection = fit_projection(context, context_dimensions)
        context = project(context, context_projection)
    model = fit_mixture(nodes, context, seed=seed, rank=factor_rank)
    return dict(node_projection=node_projection, context_projection=context_projection, model=model)


def score(capture, case, kind, version, fitted):
    nodes, context = features(capture, case, kind, version)
    nodes = project(nodes, fitted['node_projection'])
    if fitted['context_projection'] is not None:
        context = project(context, fitted['context_projection'])
    density, posterior = conditional_log_density(nodes, context, fitted['model'])
    factor_mean, factor_covariance = factor_posterior(nodes, context, fitted['model'])
    return -density, posterior, factor_mean, factor_covariance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--node-dimensions', type=int, default=32)
    parser.add_argument('--context-dimensions', type=int, default=48)
    parser.add_argument('--factor-rank', type=int, default=4)
    args = parser.parse_args()
    inputs = json.loads((args.capture / 'inputs.json').read_text())
    assert json.loads((args.capture / 'execution.json').read_text())['status'] == 'DONE'
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'inputs.json', inputs)
    snapshot = args.output / 'code_snapshot'
    snapshot.mkdir()
    files = [Path(__file__).with_name(name) for name in ('run.py', 'model.py', 'evaluate.py')]
    for path in files:
        (snapshot / path.name).write_bytes(path.read_bytes())
    write_json(args.output / 'protocol.json', dict(version=args.version, seed=args.seed,
        clusters=4, factor_rank=args.factor_rank, node_dimensions=args.node_dimensions,
        context_dimensions=args.context_dimensions, iterations=30,
        coordinate_projection=int(np.load(args.capture / inputs['cases'][0]['id'] / 'features.npz')['node'].shape[-1]),
        threshold_quantile=.95, prior_alpha=1.2, ridge=.01,
        raw_capture=str(args.capture), hashes=digest_files(files), labels_for_fit=False,
        scope='previously studied regression; no blind confirmation', risk='conditional density anomaly',
        optimizer='consistent MAP EM; fixed lambda=.01*Nfit; matrix-normal covariance prior term'))
    started = time.time()
    cases = inputs['cases']
    fit_nodes = np.concatenate([features(args.capture, c, 'node', args.version)[0]
                               for c in cases if c['cohort'] == 'fit'])
    node_projection = fit_projection(fit_nodes, args.node_dimensions)
    scores = {}
    for case in cases:
        raw = np.load(args.capture / case['id'] / 'features.npz')
        scores[case['id']] = dict(nll=raw['nll'], token_id=raw['token_id'])
    diagnostics = {}
    for kind in ('node', 'native', 'rewired'):
        fitted = train(args.capture, cases, kind, args.version, args.seed, node_projection,
                       args.context_dimensions, args.factor_rank)
        with (args.output / (kind + '_model.pkl')).open('wb') as stream:
            pickle.dump(fitted, stream)
        diagnostics[kind] = dict(trace=fitted['model']['trace'],
            node_variance=node_projection['variance_retained'],
            context_variance=(None if kind == 'node' else fitted['context_projection']['variance_retained']))
        for case in cases:
            risk, posterior, factor_mean, factor_covariance = score(
                args.capture, case, kind, args.version, fitted)
            scores[case['id']][kind] = risk
            scores[case['id']][kind + '_posterior'] = posterior
            scores[case['id']][kind + '_factor_mean'] = factor_mean
            scores[case['id']][kind + '_factor_covariance'] = factor_covariance
        print(f'FIT v{args.version} {kind} seed={args.seed} '
              f'NLL={-fitted["model"]["trace"][-1]["mean_log_density"]:.3f}', flush=True)
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        reference = [scores[c['id']] for c in cases if c['cohort'] == 'reference' and c['task'] == task]
        thresholds[task] = {kind: float(np.quantile(np.concatenate([r[kind] for r in reference]), .95))
                            for kind in ('node', 'native', 'rewired', 'nll')}
    paths = []
    for case in cases:
        directory = args.output / case['id']
        directory.mkdir()
        raw = np.load(args.capture / case['id'] / 'features.npz')
        scores[case['id']].update(token_id=raw['token_id'], nll=raw['nll'])
        for kind in ('node', 'native', 'rewired'):
            assert np.isfinite(scores[case['id']][kind]).all()
        path = directory / 'scores.npz'
        np.savez(path, **scores[case['id']])
        paths.append(path)
    write_json(args.output / 'thresholds.json', thresholds)
    write_json(args.output / 'diagnostics.json', diagnostics)
    paths += [args.output / name for name in ('thresholds.json', 'protocol.json', 'inputs.json')]
    write_json(args.output / 'frozen_scores.json', digest_files(paths))
    write_json(args.output / 'execution.json', dict(status='DONE', seconds=time.time() - started,
        cases=len(cases), labels_read_for_scoring=False))


if __name__ == '__main__':
    main()
