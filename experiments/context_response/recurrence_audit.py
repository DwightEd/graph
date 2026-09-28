"""After-score label audit of recovered tokens, propagation origins and boundaries."""
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.decision_risk_flow.data import read_json, write_json


def audit(output):
    read_json(output/'evaluation.json')
    records = {r['key']: r for r in read_json(output/'manifest.json')['records']}
    thresholds = read_json(output/'thresholds.json')
    with (output/'token_audit.csv').open() as stream:
        tokens = list(csv.DictReader(stream))
    truth = {(r['key'], int(r['position'])): int(r['label']) for r in tokens}
    methods = read_json(output/'scores_frozen.json')['methods']
    changes, edges, all_scores, targets = [], [], {m: [] for m in methods}, []
    miss_causes = []
    for key in dict.fromkeys(r['key'] for r in tokens):
        row = records[key]
        positions = sorted(t for k, t in truth if k==key)
        labels = np.array([truth[key, t] for t in positions])
        with np.load(output/key/'scores.npz') as saved:
            score = {name: saved[name] for name in methods}
        with np.load(output/key/'recurrence.npz') as saved:
            graph = {name: saved[name] for name in saved.files}
        base = score['strong_fused']
        old_alarm = base>thresholds[row['task']]['strong_fused']
        text = {int(r['position']): r['text'] for r in tokens if r['key']==key}
        targets.extend(labels)
        for method in methods:
            all_scores[method].extend(score[method][positions])
            if method not in graph:
                continue
            alarm = score[method]>thresholds[row['task']][method]
            seeds = np.flatnonzero(base>thresholds[row['task']][method])
            for t, label in zip(positions, labels):
                origin = int(graph[method][t])
                change = ('recovered' if label else 'new_false_alarm') if alarm[t] and not old_alarm[t] else (
                    ('lost_detection' if label else 'removed_false_alarm') if old_alarm[t] and not alarm[t] else (
                    'remaining_miss' if label and not alarm[t] else 'unchanged'))
                changes.append(dict(key=key, task=row['task'], method=method, token=t, text=text[t],
                    label=int(label), change=change, score=float(score[method][t]), baseline=float(base[t]),
                    origin=origin, origin_label=truth.get((key, origin), -1), origin_text=text.get(origin, ''),
                    raised=bool(score[method][t]>base[t]),
                    threshold=float(thresholds[row['task']][method])))
                if method=='recurrence_offline' and label and not alarm[t]:
                    near = seeds[np.abs(seeds-t)<=24]
                    reason = 'no_seed_in_answer_at_new_threshold' if not len(seeds) else (
                        'no_seed_within_24_tokens' if not len(near) else 'no_3hop_path_above_threshold')
                    miss_causes.append(dict(key=key, token=t, text=text[t], reason=reason,
                                            nearby_seeds=near.tolist()))
        for lag in range(1, 9):
            for start, weight in enumerate(graph[f'lag_{lag}']):
                end = start+lag
                if (key, start) in truth and (key, end) in truth:
                    edges.append(dict(key=key, lag=lag, weight=float(weight),
                        left_label=truth[key, start], right_label=truth[key, end]))
    with (output/'propagation_tokens.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=changes[0])
        writer.writeheader()
        writer.writerows(changes)
    summary = {}
    for method in methods:
        rows = [r for r in changes if r['method']==method]
        if rows:
            summary[method] = {name: sum(r['change']==name for r in rows) for name in
                ('recovered', 'new_false_alarm', 'lost_detection', 'removed_false_alarm', 'remaining_miss')}
            summary[method]['new_FP_from_normal_seed'] = sum(r['change']=='new_false_alarm' and r['origin_label']==0 for r in rows)
            summary[method]['new_FP_from_error_seed'] = sum(r['change']=='new_false_alarm' and r['origin_label']==1 for r in rows)
    edge_stats = {}
    for pattern in ((0,0), (0,1), (1,0), (1,1)):
        selected = [r for r in edges if (r['left_label'], r['right_label'])==pattern]
        edge_stats[str(pattern)] = dict(pairs=len(selected), active=sum(r['weight']>0 for r in selected))
    target = np.asarray(targets)
    matched = {}
    for method, values in all_scores.items():
        values = np.asarray(values)
        normal = np.sort(values[target==0])[::-1]
        # Diagnostic threshold only; never written into detector thresholds.
        cutoff = normal[86]
        matched[method] = dict(pooled_auroc=float(roc_auc_score(target, values)),
            errors_at_most_87_FP=int(((values>cutoff)&(target==1)).sum()),
            FP=int(((values>cutoff)&(target==0)).sum()))
    report = dict(changes=summary, edge_label_patterns=edge_stats, matched_FP_posthoc=matched,
        warning='labels only in evaluation; mixed-task pooled ordering and oracle threshold are diagnostics, not deployment')
    write_json(output/'propagation_audit.json', report)
    write_json(output/'remaining_miss_causes.json', dict(tokens=miss_causes,
        counts={reason: sum(r['reason']==reason for r in miss_causes) for reason in
                ('no_seed_in_answer_at_new_threshold', 'no_seed_within_24_tokens', 'no_3hop_path_above_threshold')}))
    print(report)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)
