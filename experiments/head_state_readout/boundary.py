"""Test future-window and unit-broadcast leakage without using gold boundaries."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import fit_cdf, percentile, valid_tokens
from .score import OLD

NEW = ('unsmoothed_fused', 'raw_backbone', 'token_backbone', 'token_source_fused')


def reference(rows, values, name, output):
    observed, weights = [], []
    for row in rows:
        if row['role'] == 'fit':
            valid = valid_tokens(row, output, OLD)
            observed.append(values[row['key']][name][valid])
            weights.append(np.full(valid.sum(), 1/valid.sum()))
    return fit_cdf(np.concatenate(observed), np.concatenate(weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('outputs/head_state_readout_20260928_v3'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous / 'manifest.json')
    write_json(args.output / 'manifest.json', dict(manifest, previous=str(args.previous.resolve())))
    values, scores = {}, {}
    for row in manifest['records']:
        key = row['key']
        directory = args.output / key
        directory.mkdir()
        (directory/'context.npz').symlink_to((args.previous/key/'context.npz').resolve())
        with np.load(args.previous/key/'scores.npz') as saved:
            scores[key] = {name: saved[name] for name in saved.files}
        with np.load(Path(manifest['route_base'])/key/'scores.npz') as saved:
            values[key] = dict(unit=saved['source_pair'], route=saved['raw_route'], association=scores[key]['association'])
        if row['kind'] == 'observer':
            with np.load(Path(row['root'])/row['directory']/'scores.npz') as saved:
                values[key]['token'] = .5*(saved['source_local']+saved['source_full'])
        else:
            values[key]['token'] = np.full(len(values[key]['route']), np.nan)
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [row for row in manifest['records'] if row['task'] == task]
        refs = {name: reference(rows, values, name, args.output) for name in ('unit', 'token', 'route', 'association')}
        for row in rows:
            key = row['key']
            ranks = {name: percentile(values[key][name], fitted) for name, fitted in refs.items()}
            raw = .75*ranks['unit']+.25*ranks['route']
            token = .75*ranks['token']+.25*ranks['route']
            scores[key].update(raw_backbone=raw, token_backbone=token,
                unsmoothed_fused=.75*raw+.25*ranks['association'],
                token_source_fused=.75*token+.25*ranks['association'])
    thresholds = read_json(args.previous/'thresholds.json')
    for task in thresholds:
        rows = [row for row in manifest['records'] if row['role']=='dev' and row['task']==task]
        for name in NEW:
            observed, weights = [], []
            for row in rows:
                valid = valid_tokens(row, args.output, OLD)
                observed.append(scores[row['key']][name][valid])
                weights.append(np.full(valid.sum(), 1/valid.sum()))
            ordered, cumulative = fit_cdf(np.concatenate(observed), np.concatenate(weights))
            thresholds[task][name] = float(ordered[np.searchsorted(cumulative[1:], .95)])
    for row in manifest['records']:
        np.savez_compressed(args.output/row['key']/'scores.npz', **scores[row['key']])
    methods = NEW+tuple(read_json(args.previous/'scores_frozen.json')['methods'])
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(methods=methods, main='unsmoothed_fused', status='complete',
        labels_used=False, head_selection=False, boundary_labels_used=False,
        design_informed_by='Observed v3 seven-token early alarm; no token-specific exception',
        scope='Reuse existing token source likelihoods; no new source deletion; natural source scores unavailable'))


if __name__ == '__main__':
    main()
