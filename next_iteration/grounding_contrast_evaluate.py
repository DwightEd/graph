"""Annotation join AFTER registered candidate predictions are complete."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_metrics(y, score, weights=None):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if y.shape != score.shape or w.shape != y.shape or not np.isfinite(score).all():
        raise ValueError('invalid aligned scores')
    if not np.isin(y, [0, 1]).all() or np.any(w < 0):
        raise ValueError('invalid labels/weights')
    positives, negatives = np.sum(w * y), np.sum(w * (1 - y))
    if positives == 0 or negatives == 0:
        return dict(auroc=None, auprc=None)
    order = np.argsort(score, kind='stable')
    values, yy, ww = score[order], y[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    pos = np.add.reduceat(ww * yy, starts)
    neg = np.add.reduceat(ww * (1 - yy), starts)
    auc = np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (positives * negatives)
    pos, neg = pos[::-1], neg[::-1]
    denominator = np.cumsum(pos + neg)
    precision = np.divide(np.cumsum(pos), denominator, out=np.zeros_like(denominator), where=denominator > 0)
    ap = np.sum(pos * precision) / positives
    return dict(auroc=float(auc), auprc=float(ap))


def annotation_targets(offsets, labels):
    y = np.zeros(len(offsets), dtype=int)
    onset = np.zeros(len(offsets), dtype=int)
    for label in labels:
        start, end = int(label['start']), int(label['end'])
        overlap = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if end <= start or not overlap:
            raise ValueError('unmapped or empty annotation')
        y[overlap] = 1
        onset[overlap[0]] = 1
    through = np.ones(len(offsets), dtype=bool)
    if y.any():
        through[np.flatnonzero(y)[0] + 1:] = False
    return y, onset, through


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--primary-score', default='source_risk')
    parser.add_argument('--comparison-scores', nargs='+',
                        default=['native_nll', 'native_entropy', 'native_negative_margin'])
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError('positive bootstrap count required')
    root = Path(args.predictions)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('error') or manifest['completed_ids'] != manifest['planned_ids']:
        raise ValueError('predictions incomplete')
    if manifest['settings']['phase'] == 'pilot':
        raise ValueError('latency pilot may not join natural labels')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    # First load frozen predictions, only then annotations. No scoring code imported.
    records = []
    for rid in manifest['planned_ids']:
        path = root / f'response_{rid}.json'
        if sha(path) != manifest['output_sha256'][path.name]:
            raise ValueError('prediction hash mismatch')
        record = json.loads(path.read_text())
        if str(record['id']) != rid:
            raise ValueError('embedded response ID mismatch')
        records.append(record)
    wanted = {str(r['id']) for r in records}
    annotations = {}
    for line in Path(args.annotations).open():
        row = json.loads(line)
        if str(row['id']) in wanted:
            if str(row['id']) in annotations:
                raise ValueError('duplicate annotation ID')
            annotations[str(row['id'])] = row
    if set(annotations) != wanted:
        raise ValueError('missing annotations')
    names = list(records[0]['scores'])
    if not set([args.primary_score] + args.comparison_scores) <= set(names):
        raise ValueError('required scores missing')
    Ys, Os, Ts, Sources, Tasks = [], [], [], [], []
    scores = {name: [] for name in names}
    for r in records:
        annotation = annotations[str(r['id'])]
        if hashlib.sha256(annotation['response'].encode()).hexdigest() != r['response_sha256']:
            raise ValueError('response hash mismatch')
        if str(annotation['source_id']) != str(r['source_id']):
            raise ValueError('source mismatch')
        y, onset, through = annotation_targets(r['offsets'], annotation['labels'])
        if len(r['token_ids']) != len(y):
            raise ValueError('token alignment')
        Ys.extend(y); Os.extend(onset); Ts.extend(through)
        Sources.extend([str(r['source_id'])] * len(y)); Tasks.extend([r['task']] * len(y))
        for name in names:
            if len(r['scores'][name]) != len(y):
                raise ValueError('score alignment')
            scores[name].extend(r['scores'][name])
    y, onset, through = np.asarray(Ys), np.asarray(Os), np.asarray(Ts)
    sources, tasks = np.asarray(Sources), np.asarray(Tasks)
    scores = {name: np.asarray(values) for name, values in scores.items()}

    def table(mask, targets):
        yy, ss = targets[mask], sources[mask]
        unique, inverse, counts = np.unique(ss, return_inverse=True, return_counts=True)
        weights = 1. / counts[inverse] if len(ss) else np.array([])
        return dict(tokens=int(mask.sum()), positives=int(yy.sum()), sources=len(unique),
                    micro={n: weighted_metrics(yy, v[mask]) for n, v in scores.items()},
                    source_balanced={n: weighted_metrics(yy, v[mask], weights) for n, v in scores.items()})

    all_mask = np.ones(len(y), dtype=bool)
    results = dict(phase=manifest['settings']['phase'], responses=len(records), sources=len(set(sources)),
                   primary_score=args.primary_score, comparison_scores=args.comparison_scores,
                   official_split='train', generalization_claim=False,
                   annotation_sha256=sha(args.annotations), prediction_manifest_sha256=sha(root / 'manifest.json'),
                   evaluation_code_sha256=sha(__file__),
                   label_rule='any character overlap; onset = first token overlapping each annotated span',
                   token_timing='entropy/margin/transport pre-token; NLL conditions on observed target; verifier after current token; retrospective uses future',
                   all_token=table(all_mask, y), full_stream_onset=table(all_mask, onset),
                   through_first_error=table(through, y),
                   per_task={task: table(tasks == task, y) for task in sorted(set(tasks))})
    unique, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
    controls = args.comparison_scores
    # Paired source-cluster resampling; multiplicity applies to entire original source.
    for balanced in (False, True):
        rng = np.random.default_rng(20260913)
        base_weights = 1. / counts[inverse] if balanced else np.ones(len(y))
        replicates = {c: {'auroc': [], 'auprc': []} for c in controls}
        for _ in range(args.bootstrap):
            multiplicity = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
            weights = base_weights * multiplicity[inverse]
            primary = weighted_metrics(y, scores[args.primary_score], weights)
            if primary['auroc'] is None:
                continue
            for control in controls:
                other = weighted_metrics(y, scores[control], weights)
                for metric in ('auroc', 'auprc'):
                    replicates[control][metric].append(primary[metric] - other[metric])
        key = 'source_balanced' if balanced else 'micro'
        def point_difference(control, metric):
            a = results['all_token'][key][args.primary_score][metric]
            b = results['all_token'][key][control][metric]
            return None if a is None or b is None else a - b
        results[f'paired_source_bootstrap_{key}'] = {
            control: {metric: dict(valid=len(values),
                                   difference=point_difference(control, metric),
                                   ci95=np.quantile(values, [.025, .975]).tolist() if values else None)
                      for metric, values in metrics.items()} for control, metrics in replicates.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    code_snapshot = output.with_name(output.name + '.executed_code.py')
    with code_snapshot.open('xb') as handle:
        handle.write(Path(__file__).read_bytes())
    with output.open('x') as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: results[k] for k in ('responses', 'sources', 'all_token', 'through_first_error')}))


if __name__ == '__main__':
    main()
