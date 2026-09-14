"""Original development32 metrics with ID-scoped annotation access; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .grounding_contrast_evaluate import sha, annotation_targets, weighted_metrics
from .confirmation_assess_scoped import selected_annotations

DEV_IDS = {'11649','11650','11847','11848','13083','13084','13515','13516',
           '13857','13858','15603','15604','15939','15940','17019','17020',
           '1773','1774','219','220','3687','3688','4449','4450','4707','4708',
           '7305','7306','741','742','9021','9022'}


def frozen_records(root):
    root = Path(root)
    m = json.loads((root / 'manifest.json').read_text())
    if not m.get('complete') or m.get('error') or m['planned_ids'] != m['completed_ids']:
        raise ValueError('incomplete predictions')
    records = []
    for rid in m['planned_ids']:
        path = root / f'response_{rid}.json'
        if sha(path) != m['output_sha256'][path.name]:
            raise ValueError('prediction hash mismatch')
        r = json.loads(path.read_text())
        if str(r['id']) != str(rid):
            raise ValueError('embedded ID mismatch')
        records.append(r)
    if len(set(str(r['id']) for r in records)) != len(records):
        raise ValueError('duplicate predictions')
    return m, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--primary', required=True)
    p.add_argument('--reference-predictions', action='append', default=[])
    p.add_argument('--bootstrap', type=int, default=500)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    m, records = frozen_records(args.predictions)
    if m['settings']['phase'] != 'development' or {str(r['id']) for r in records} != DEV_IDS:
        raise ValueError('only complete original development32 may be evaluated here')
    reference_hashes = {}
    for root in args.reference_predictions:
        rm, rr = frozen_records(root)
        if set(rm['planned_ids']) != set(m['planned_ids']):
            raise ValueError('reference roster differs')
        reference_hashes[root] = sha(Path(root) / 'manifest.json')
        lookup = {str(r['id']): r for r in rr}
        for r in records:
            ref = lookup[str(r['id'])]
            for key in ('token_ids', 'offsets', 'response_sha256', 'source_id', 'task'):
                if ref[key] != r[key]:
                    raise ValueError('reference token/source alignment differs')
            for k, v in ref['scores'].items():
                if k in r['scores'] and r['scores'][k] != v:
                    raise ValueError('conflicting named score')
                r['scores'][k] = v
    wanted = {str(r['id']) for r in records}
    gt = selected_annotations(args.annotations, wanted)
    if set(gt) != wanted:
        raise ValueError('missing annotations')
    names = list(records[0]['scores'])
    if args.primary not in names:
        raise ValueError('primary absent')
    labels, onsets, sources, tasks, scores, per_response, cases = [], [], [], [], {n: [] for n in names}, [], []
    for r in records:
        a = gt[str(r['id'])]
        if hashlib.sha256(a['response'].encode()).hexdigest() != r['response_sha256'] or str(a['source_id']) != str(r['source_id']):
            raise ValueError('GT identity mismatch')
        y, onset, _ = annotation_targets(r['offsets'], a['labels'])
        if len(y) != len(r['token_ids']) or set(names) != set(r['scores']):
            raise ValueError('token/score mismatch')
        for n in names:
            if len(r['scores'][n]) != len(y) or not np.isfinite(r['scores'][n]).all():
                raise ValueError('invalid scores')
            scores[n].extend(r['scores'][n])
        labels.extend(y); onsets.extend(onset)
        sources.extend([str(r['source_id'])] * len(y)); tasks.extend([r['task']] * len(y))
        per_response.append(dict(id=r['id'], source_id=str(r['source_id']), task=r['task'],
            tokens=len(y), positives=int(y.sum()), pairs=int(y.sum() * (len(y) - y.sum())),
            metrics={n: weighted_metrics(y, r['scores'][n]) for n in names}))
        if m['settings']['phase'] == 'development':
            for label in a['labels']:
                start, end = label['start'], label['end']
                idx = [i for i, (lo, hi) in enumerate(r['offsets']) if lo < end and hi > start]
                cases.append(dict(id=r['id'], source_id=r['source_id'], task=r['task'],
                    start=start, end=end, text=a['response'][start:end],
                    context=a['response'][max(0, start-100):end+100],
                    risks=[r['scores'][args.primary][i] for i in idx]))
    y, onset, src, task = map(np.asarray, (labels, onsets, sources, tasks))
    scores = {k: np.asarray(v) for k, v in scores.items()}
    unique, inv, counts = np.unique(src, return_inverse=True, return_counts=True)
    balanced = 1.0 / counts[inv]
    all_metrics = {n: weighted_metrics(y, v) for n, v in scores.items()}
    source_metrics = {n: weighted_metrics(y, v, balanced) for n, v in scores.items()}
    mixed = [r for r in per_response if r['pairs']]
    within = {n: dict(pair_weighted_auroc=float(np.average([r['metrics'][n]['auroc'] for r in mixed],
                     weights=[r['pairs'] for r in mixed])),
                     macro_auroc=float(np.mean([r['metrics'][n]['auroc'] for r in mixed]))) for n in names} if mixed else {}
    rng = np.random.default_rng(20260914)
    boots = {n: [] for n in names}
    balanced_boot = []
    for _ in range(args.bootstrap):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        w = multiplicity[inv]
        b = {n: weighted_metrics(y, v, w) for n, v in scores.items()}
        if b[args.primary]['auroc'] is None:
            continue
        for n in names:
            boots[n].append([b[n]['auroc'], b[n]['auprc']])
        bb = weighted_metrics(y, scores[args.primary], w * balanced)
        balanced_boot.append([bb['auroc'], bb['auprc']])
    primary = np.asarray(boots[args.primary])
    report = dict(phase=m['settings']['phase'], retrospective=True, primary_score=args.primary,
        source_scope='R04 previously observed official-train sources; not pristine official test',
        responses=len(records), sources=len(unique), tokens=len(y), positives=int(y.sum()),
        prediction_manifest_sha256=sha(Path(args.predictions)/'manifest.json'), reference_manifest_sha256=reference_hashes,
        annotation_sha256=sha(args.annotations), executed_code_sha256=sha(__file__),
        all_token=all_metrics, source_balanced=source_metrics, within_answer=within,
        full_stream_onset={n: weighted_metrics(onset, v) for n, v in scores.items()},
        per_task={t: dict(tokens=int((task==t).sum()), positives=int(y[task==t].sum()),
                        metrics={n: weighted_metrics(y[task==t], v[task==t]) for n, v in scores.items()}) for t in sorted(set(task))},
        bootstrap=dict(seed=20260914, unit='entire source including both responses', requested=args.bootstrap,
            valid=len(primary), primary_absolute_95_ci=np.quantile(primary,[.025,.975],axis=0).T.tolist(),
            primary_source_balanced_absolute_95_ci=np.quantile(balanced_boot,[.025,.975],axis=0).T.tolist(),
            paired_primary_minus_control_95_ci={n: np.quantile(primary-np.asarray(boots[n]),[.025,.975],axis=0).T.tolist() for n in names}),
        per_response=per_response, development_error_spans=cases)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    with Path(args.output).open('x') as f:
        f.write(serialized + '\n')
    Path(str(args.output)+'.executed_code.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in report.items() if k not in ('per_response','development_error_spans')}))


if __name__ == '__main__':
    main()
