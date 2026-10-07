"""Additional onset and exact-replay diagnostics of already frozen matrix scores."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from experiments.token_backtrace.repetition_analysis import source_bootstrap
from experiments.token_backtrace.repetition_teacher import BASE, SelfReadout
from .matrix_audit import OUTPUT, RANKS, sha256, write_json


def independent_auc(labels, scores):
    ranks = rankdata(scores, method='average')
    positive = labels.astype(bool)
    count = int(positive.sum())
    return float((ranks[positive].sum() - count * (count + 1) / 2)
                 / (count * (~positive).sum()))


def onset_metrics(rows, scores):
    onsets = []
    normal = []
    continuing = []
    truth = []
    for row in rows:
        labels = row['labels'].astype(bool)
        starts = labels & ~np.r_[False, labels[:-1]]
        onsets.extend(starts)
        normal.extend(~labels)
        continuing.extend(labels & ~starts)
        truth.extend(labels)
    onsets, normal, continuing, truth = map(np.asarray, (onsets, normal, continuing, truth))
    result = {}
    for name in ('full', 'rank1', 'rank2', 'rank4', 'rank8'):
        values = np.concatenate([scores[f"{row['id']}/{name}"] for row in rows])
        onset_cohort = onsets | normal
        continuation_cohort = continuing | normal
        result[name] = dict(auroc=independent_auc(truth, values),
                            onset_auroc=independent_auc(onsets[onset_cohort], values[onset_cohort]),
                            continuation_auroc=independent_auc(
                                continuing[continuation_cohort], values[continuation_cohort]))
    return dict(onsets=int(onsets.sum()), continuing=int(continuing.sum()),
                normal_tokens=int(normal.sum()), metrics=result)


def dense_replay(rows, scores):
    """Use head-axis SVD, independent of the production temporal Gram solver."""
    teachers = [SelfReadout(seed) for seed in (42, 123)]
    selected = rows[:2] + [row for row in rows if row['id'] in ('12219', '12297')]
    errors = {}
    for row in selected:
        projected = {rank: [] for rank in RANKS}
        for teacher in teachers:
            values = teacher.standardized(row).numpy().astype(np.float64)
            direction = teacher.tangent()
            per_rank = {rank: [] for rank in RANKS}
            for target in range(len(values)):
                window = values[max(0, target - 7):target + 1]
                temporal, singular, heads = np.linalg.svd(window, full_matrices=False)
                for rank in RANKS:
                    reconstructed = (temporal[:, :rank] * singular[:rank]) @ heads[:rank]
                    per_rank[rank].append(reconstructed[-1] @ direction)
            for rank in RANKS:
                projected[rank].append(per_rank[rank])
        errors[row['id']] = {rank: float(np.max(np.abs(np.mean(projected[rank], axis=0)
                    - scores[f"{row['id']}/rank{rank}"]))) for rank in RANKS}
    return errors


def paired_changes(rows, scores):
    pairs = json.loads(Path('outputs/repetition_dominance_20261007/matched_tokens.json').read_text())['strict']
    sources = []
    differences = {rank: [] for rank in (1, 2, 4)}
    for row in rows:
        baseline = scores[f"{row['id']}/full"]
        for target, control in pairs[row['id']]:
            original = float(baseline[target] > baseline[control])
            sources.append(row['source_id'])
            for rank in differences:
                values = scores[f"{row['id']}/rank{rank}"]
                ordered = float(values[target] > values[control])
                differences[rank].append(ordered - original)
    return {rank: source_bootstrap(sources, values) for rank, values in differences.items()}


def main():
    scores_path = OUTPUT / 'frozen_scores.npz'
    score_freeze = json.loads((OUTPUT / 'SCORE_FREEZE.json').read_text())
    assert sha256(scores_path) == score_freeze['sha256']
    scores = np.load(scores_path)
    rows = torch.load(BASE / 'prepared.pt', weights_only=False)['test']
    result = dict(scope='post-score explanatory analysis; not a new detector',
                  score_sha256=sha256(scores_path), evaluation_code_sha256=sha256(Path(__file__)),
                  onset=onset_metrics(rows, scores), dense_svd_replay=dense_replay(rows, scores),
                  paired_ordering_changes=paired_changes(rows, scores))
    original = json.loads((OUTPUT / 'RESULTS.json').read_text())
    result['independent_auc_max_error'] = max(abs(result['onset']['metrics'][name]['auroc']
                    - original['metrics'][name]['auroc']) for name in original['metrics'])
    write_json(OUTPUT / 'CASE_AUDIT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
