"""Supervised mechanism diagnostic only: learn direction on one local pair."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.role_free_flow.diagnostics import read_local_spans
from .density import standardize
from .features import BLOCKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output / 'evaluation.json')
    manifest = read_json(args.output / 'manifest.json')
    lookup = {row['trace']: row for row in manifest['records'] if row['role'] == 'natural'}
    local = read_local_spans(Path('outputs/role_free_flow_20260928'), Path('experiments/path_conflict/paired_cases.json'))
    scale = np.load(args.output / 'QA_scales.npz')
    pairs = {}
    for case, rows in local.groupby('case'):
        labels = rows.local_label.to_numpy()
        blocks = {}
        for block in BLOCKS:
            raw = np.array([np.load(args.output / lookup[row.trace]['key'] / (block+'.npy'), mmap_mode='r')[row.position] for row in rows.itertuples()])
            fitted = (scale[block+'_center'], scale[block+'_scale'])
            blocks[block] = standardize(raw, fitted)
        pairs[case] = (labels, blocks)
    results = []
    names = list(pairs)
    for train_name, test_name in (names, names[::-1]):
        train_labels, train_blocks = pairs[train_name]
        test_labels, test_blocks = pairs[test_name]
        predictions = {}
        for block in BLOCKS:
            train, test = train_blocks[block], test_blocks[block]
            direction = train[train_labels == 1].mean(0)-train[train_labels == 0].mean(0)
            midpoint = .5*(train[train_labels == 1].mean(0)+train[train_labels == 0].mean(0))
            predictions[block] = (test-midpoint) @ direction / train.shape[1]
        geometry = .5*(predictions['geometry_head']+predictions['geometry_gram'])
        jacobian = (predictions['jacobian_head']+predictions['tangent']+predictions['fisher'])/3
        predictions['joint'] = (geometry+predictions['js']+jacobian)/3
        for block, score in predictions.items():
            results.append(dict(train_pair=train_name, test_pair=test_name, block=block,
                heldout_pair_auroc=float(roc_auc_score(test_labels, score))))
    write_json(args.output / 'pair_diagnostic.json', dict(supervised=True, deployed=False, rows=results,
        warning='Labels fit a centroid direction on one source, evaluated on the other exposed source. Diagnostic only; two sources, no reliable generalization estimate.'))
    print(results, flush=True)


if __name__ == '__main__':
    main()
