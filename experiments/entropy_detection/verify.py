"""Recompute frozen-score ranking and audit source exclusion without changing scores."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from .run import read_json, write_json, exposed_sources, allowed_mask
from experiments.probabilistic_detection.data import load_pack, evaluation_labels
from experiments.probabilistic_detection.evaluation import source_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--packs', type=Path, required=True)
    args = parser.parse_args()
    results = {}
    excluded = exposed_sources(args.packs)
    for task in ('QA', 'Summary', 'Data2txt'):
        train, meta = load_pack(args.packs, task, 'train')
        allowed = allowed_mask(meta, excluded, len(train['target']))
        fit_sources = set(train['source_index'][allowed & ~train['development']])
        dev_sources = set(train['source_index'][allowed & train['development']])
        assert fit_sources.isdisjoint(dev_sources)
        pack, meta = load_pack(args.packs, task, 'test')
        truth = evaluation_labels(Path(meta['source_cache']), pack, meta)
        labels = truth['labels'].astype(bool)
        expected = read_json(args.run/task/'test_metrics.json')
        with np.load(args.run/task/'test_scores.npz') as saved:
            errors = []
            for name in saved.files:
                scores = saved[name]
                ranks = rankdata(scores, method='average')
                positive, negative = labels.sum(), (~labels).sum()
                auroc = (ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative)
                errors.append(abs(auroc-expected[name]['auroc']))
            interval = source_bootstrap(dict(pack, **truth), saved['joint'], saved['static'])
        assert max(errors) < 1e-12
        results[task] = dict(auroc_max_abs_error=max(errors), checked_scores=len(errors),
            fit_dev_disjoint=True, excluded_sources=sorted(excluded), joint_vs_static=interval)
    write_json(args.run/'verification.json', dict(status='passed', reviewer='same agent; independent rank-sum AUROC implementation', tasks=results))
    print('VERIFIED', results, flush=True)


if __name__ == '__main__':
    main()
