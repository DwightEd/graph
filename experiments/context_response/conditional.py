"""Calibrate sparse native response against matched unlabeled generation contexts."""
import argparse
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import valid_tokens
from experiments.head_state_readout.score import OLD
from experiments.source_relation.refine import fit_rank
from .score import calibrate


def condition_rank(context, value, reference_context, reference_value, reference_sources, query_source):
    center = np.median(reference_context, axis=0)
    scale = np.maximum(np.quantile(reference_context, .75, axis=0)-np.quantile(reference_context, .25, axis=0), 1.)
    differences = (context[:, None]-reference_context[None])/scale
    distance = np.mean(differences**2, axis=-1)
    ranks = []
    for source in np.unique(reference_sources):
        if source == query_source:
            continue
        indices = np.flatnonzero(reference_sources==source)
        nearest = np.argsort(distance[:, indices], axis=1, kind='stable')[:, :16]
        selected = reference_value[indices[nearest]]
        ranks.append(((selected<value[:, None])+.5*(selected==value[:, None])).mean(-1))
    return np.mean(ranks, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output/'manifest.json', manifest)
    all_context, all_scores = {}, {}
    for row in manifest['records']:
        key = row['key']
        directory = args.output/key
        directory.mkdir()
        for name in ('responses.npz', 'context.npz'):
            (directory/name).symlink_to((args.previous/key/name).resolve())
        # Entropy, surprisal, position, prompt length, copy, surface; route excluded.
        all_context[key] = np.load(directory/'context.npz')['values'][:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]]
        with np.load(args.previous/key/'scores.npz') as saved:
            all_scores[key] = {name: saved[name] for name in saved.files}
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        reference_context, reference_value, reference_sources = [], [], []
        for row in rows:
            if row['role']=='fit':
                valid = valid_tokens(row, args.output, OLD)
                reference_context.append(all_context[row['key']][valid])
                reference_value.append(all_scores[row['key']]['sparse_joint'][valid])
                reference_sources.extend([row['source_id']]*valid.sum())
        reference_context = np.concatenate(reference_context)
        reference_value = np.concatenate(reference_value)
        reference_sources = np.asarray(reference_sources)
        for row in rows:
            key = row['key']
            all_scores[key]['conditional_sparse'] = condition_rank(all_context[key],
                all_scores[key]['sparse_joint'], reference_context, reference_value, reference_sources, row['source_id'])
        scores = {r['key']: all_scores[r['key']] for r in rows}
        rank = fit_rank(scores, 'conditional_sparse', rows, args.output)
        for key in scores:
            scores[key]['conditional_sparse_fused'] = .75*scores[key]['source_route_fixed']+.25*rank[key]
            np.savez_compressed(args.output/key/'scores.npz', **scores[key])
        thresholds[task] = calibrate(args.output, rows, scores)
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(status='complete', main='conditional_sparse_fused',
        methods=list(next(iter(all_scores.values()))), labels_used=False,
        context='entropy, surprisal, position, prompt length, copy, surface; excludes route',
        neighbors='16 nearest context positions per fit source, same source excluded',
        meaning='conditional anomaly percentile in an unlabeled mixture; not factuality probability'))


if __name__ == '__main__':
    main()
