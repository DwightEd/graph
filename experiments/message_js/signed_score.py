"""Fixed signed choice responses in the native Fisher metric; no label fitting."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .score import fit_cdf, valid_tokens

METHODS = ('signed_prompt_contrast', 'signed_prompt_raw', 'history_over_prompt')


def signed_scores(margin, gram):
    scale = np.sqrt(np.maximum(np.diagonal(gram, axis1=1, axis2=2), 0))
    normalized = np.divide(margin, scale, out=np.full_like(margin, np.nan), where=scale > 0)
    product = scale[:, 0] * scale[:, 1]
    correlation = np.divide(gram[:, 0, 1], product, out=np.full_like(product, np.nan), where=product > 0)
    difference_scale = np.sqrt(np.maximum(2 - 2 * np.clip(correlation, -1, 1), 0))
    opposition = np.divide(normalized[:, 1] - normalized[:, 0], difference_scale,
                           out=np.full_like(product, np.nan), where=difference_scale > 1e-8)
    return dict(signed_prompt_contrast=-normalized[:, 0], signed_prompt_raw=-margin[:, 0],
                history_over_prompt=opposition)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', type=Path, default=Path('outputs/message_operator_20260928_v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old', type=Path, default=Path('outputs/transport_topology_cases_20260928'))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.operator / 'manifest.json')
    manifest['operator'] = str(args.operator.resolve())
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'protocol.json', dict(methods=METHODS, main=METHODS[0],
        labels_used=False, parameter_fit=False, operator_recomputed=False,
        direction='actual token vs automatic highest-probability alternative; sign is choice, not truth',
        primary='negative prompt margin response / sqrt(prompt Fisher energy)',
        opposition='history-minus-prompt normalized margin response / Fisher norm of the difference direction',
        zero_energy='missing, not zero anomaly',
        caveat='exploratory known cases; no best-method selection'))
    for row in manifest['records']:
        with np.load(args.operator / row['key'] / 'operator.npz') as data:
            scores = signed_scores(data['margin'].astype(np.float64), data['gram'].astype(np.float64))
        directory = args.output / row['key']
        directory.mkdir()
        np.savez_compressed(directory / 'scores.npz', **scores)
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        scores, weights = {method: [] for method in METHODS}, []
        for row in manifest['records']:
            if row['task'] != task or row['role'] != 'dev':
                continue
            valid = valid_tokens(row, args.output, args.old)
            weights.append(np.full(valid.sum(), 1 / valid.sum()))
            with np.load(args.output / row['key'] / 'scores.npz') as saved:
                for method in METHODS:
                    scores[method].append(saved[method][valid])
        thresholds[task] = {}
        for method in METHODS:
            ordered, cumulative = fit_cdf(np.concatenate(scores[method]), np.concatenate(weights))
            thresholds[task][method] = float(ordered[np.searchsorted(cumulative[1:], .95)])
    write_json(args.output / 'thresholds.json', thresholds)
    write_json(args.output / 'scores_frozen.json', dict(status='complete', methods=METHODS, main=METHODS[0],
        labels_used=False, fit='none', threshold='source-equal unlabeled dev mixture .95 quantile',
        reference_sources_per_task=4, operator_cache=str(args.operator.resolve())))


if __name__ == '__main__':
    main()
