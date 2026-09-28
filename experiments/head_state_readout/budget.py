"""Posthoc ROC budget diagnostic; thresholds here are never deployed."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from experiments.decision_risk_flow.data import read_json, write_json, labels
from experiments.message_js.score import fit_cdf, percentile, valid_tokens
from .score import OLD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output / 'evaluation.json')
    records = read_json(args.output / 'manifest.json')['records']
    methods = read_json(args.output / 'scores_frozen.json')['methods']
    regression = [row for row in records if row['role'] == 'regression']
    truth = labels(regression)
    results = {}
    for method in methods:
        references = {}
        for task in ('QA', 'Summary', 'Data2txt'):
            values, weights = [], []
            for row in records:
                if row['role'] == 'dev' and row['task'] == task:
                    valid = valid_tokens(row, args.output, OLD)
                    with np.load(args.output / row['key'] / 'scores.npz') as saved:
                        values.append(saved[method][valid])
                    weights.append(np.full(valid.sum(), 1/valid.sum()))
            references[task] = fit_cdf(np.concatenate(values), np.concatenate(weights))
        targets, scores = [], []
        for row in regression:
            valid = valid_tokens(row, args.output, OLD)
            with np.load(args.output / row['key'] / 'scores.npz') as saved:
                scores.extend(percentile(saved[method][valid], references[row['task']]))
            targets.extend(truth[row['key']][valid])
        targets, scores = np.asarray(targets), np.asarray(scores)
        fpr, tpr, _ = roc_curve(targets, scores, drop_intermediate=False)
        false = np.rint(fpr*(targets == 0).sum()).astype(int)
        detected = np.rint(tpr*targets.sum()).astype(int)
        results[method] = {str(budget): int(detected[false <= budget].max()) for budget in (27, 61, 71, 140)}
    write_json(args.output / 'budget_diagnostic.json', dict(posthoc=True, deployed=False, error_tokens=134,
        normal_tokens=1353, normalization='task-specific unlabelled dev CDF; ties preserved',
        meaning='Best recall allowed by this ranking at each FP budget. Uses evaluation labels only for ROC envelope, not deployable threshold.',
        results=results))
    print(results, flush=True)


if __name__ == '__main__':
    main()
