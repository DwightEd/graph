"""Read natural labels only after all scores and thresholds exist."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.decision_risk_flow.data import inputs, labels, read_json, write_json
from experiments.role_free_flow.diagnostics import read_local_spans
from .score import METHODS, valid_tokens


def metrics(target, score, threshold):
    available = np.isfinite(score)
    alarm = available & (score > threshold)
    normal, error = target == 0, target == 1
    both = len(np.unique(target[available])) == 2
    minimum = float(score[error].min()) if error.any() and available[error].all() else None
    return dict(eligible_tokens=len(target), scored_tokens=int(available.sum()),
        error_tokens=int(error.sum()), normal_tokens=int(normal.sum()),
        detected_errors=int(alarm[error].sum()), false_alarms=int(alarm[normal].sum()),
        auroc=float(roc_auc_score(target[available], score[available])) if both else None,
        ap=float(average_precision_score(target[available], score[available])) if both else None,
        threshold=threshold,
        oracle_full_recall_false_alarms=int((score[normal] >= minimum).sum()) if minimum is not None else None,
        oracle_warning='posthoc lower bound for a scalar threshold, never used for deployment')


def spans(target):
    boundaries = np.diff(np.r_[0, target == 1, 0].astype(int))
    return list(zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)))


def evaluate(output, old):
    methods = read_json(output / 'scores_frozen.json')['methods']
    records = read_json(output / 'manifest.json')['records']
    regression = [row for row in records if row['role'] == 'regression']
    truth = labels(regression)
    thresholds = read_json(output / 'thresholds.json')
    table, token_rows, span_rows = [], [], []
    pooled = {name: [[], [], []] for name in methods}
    for row in regression:
        valid = valid_tokens(row, output, old)
        target = truth[row['key']]
        _, response = inputs(row)
        score = np.load(output / row['key'] / 'scores.npz')
        for name in methods:
            threshold = thresholds[row['task']][name]
            table.append(dict(key=row['key'], method=name, **metrics(target[valid], score[name][valid], threshold)))
            pooled[name][0].extend(target[valid])
            pooled[name][1].extend(score[name][valid])
            pooled[name][2].extend(np.repeat(threshold, valid.sum()))
            for start, stop in spans(target):
                positions = np.arange(start, stop)[valid[start:stop]]
                detected = positions[score[name][positions] > threshold]
                span_rows.append(dict(key=row['key'], method=name, start=int(start), stop=int(stop),
                    error_tokens=len(positions), detected=detected.tolist(),
                    all_detected=len(detected) == len(positions), onset=bool(start in detected)))
        for token in np.flatnonzero(valid):
            token_rows.append(dict(key=row['key'], token=int(token), text=response['token_text'][token],
                label=int(target[token]), **{name: float(score[name][token]) for name in methods}))
    aggregate = {}
    for name, (targets, scores, limits) in pooled.items():
        target, score, limit = np.array(targets), np.array(scores), np.array(limits)
        alarm = np.isfinite(score) & (score > limit)
        rows = [row for row in table if row['method'] == name and row['auroc'] is not None]
        selected = [row for row in span_rows if row['method'] == name]
        aggregate[name] = dict(mean_within_answer_auroc=float(np.mean([row['auroc'] for row in rows])),
            detected_errors=int(alarm[target == 1].sum()), error_tokens=int((target == 1).sum()),
            false_alarms=int(alarm[target == 0].sum()), normal_tokens=int((target == 0).sum()),
            any_span=sum(bool(row['detected']) for row in selected), full_span=sum(row['all_detected'] for row in selected),
            onset=sum(row['onset'] for row in selected), total_spans=len(selected))
    natural_rows, natural_pairs = evaluate_natural(output, records, thresholds, methods)
    main = read_json(output / 'scores_frozen.json')['main']
    write_json(output / 'evaluation.json', dict(aggregate=aggregate, cases=table, spans=span_rows,
        natural_local=natural_rows, natural_pairs=natural_pairs,
        natural_scope='only reviewed local claims; all other tokens unknown',
        goal_passed=aggregate[main]['detected_errors'] == aggregate[main]['error_tokens']
            and aggregate[main]['false_alarms'] < 61))
    with (output / 'tokens.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=token_rows[0])
        writer.writeheader()
        writer.writerows(token_rows)
    print(aggregate, flush=True)


def evaluate_natural(output, records, thresholds, methods=METHODS):
    local = read_local_spans(Path('outputs/role_free_flow_20260928'), Path('experiments/path_conflict/paired_cases.json'))
    lookup = {row['trace']: row for row in records if row['role'] == 'natural'}
    rows, pairs = [], []
    for (case, side, trace), selected in local.groupby(['case', 'side', 'trace']):
        record = lookup[trace]
        scores = np.load(output / record['key'] / 'scores.npz')
        positions = selected.position.to_numpy()
        target = selected.local_label.to_numpy()
        for name in methods:
            rows.append(dict(case=case, side=side, key=record['key'], method=name,
                positions=positions.tolist(), **metrics(target, scores[name][positions], thresholds['QA'][name])))
    for case, selected in local.groupby('case'):
        target = selected.local_label.to_numpy()
        for name in methods:
            scores = np.array([np.load(output / lookup[row.trace]['key'] / 'scores.npz')[name][row.position]
                               for row in selected.itertuples()])
            pairs.append(dict(case=case, method=name, **metrics(target, scores, thresholds['QA'][name])))
    return rows, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old', type=Path, default=Path('outputs/transport_topology_cases_20260928'))
    args = parser.parse_args()
    if (args.output / 'evaluation.json').exists():
        raise FileExistsError('Preserve completed evaluation.')
    evaluate(args.output, args.old)


if __name__ == '__main__':
    main()
