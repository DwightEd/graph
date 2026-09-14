"""Within-answer localization companion for frozen prediction evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--annotations',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--primary',required=True)
    args=p.parse_args()
    from next_iteration.grounding_contrast_evaluate import sha,annotation_targets,weighted_metrics
    root=Path(args.predictions)
    m=json.loads((root/'manifest.json').read_text())
    if not m['complete'] or m.get('error') or m['planned_ids']!=m['completed_ids']:raise ValueError('incomplete predictions')
    if m['settings']['phase']!='development':raise ValueError('development only')
    if Path(args.output).exists():raise FileExistsError(args.output)
    records=[]
    for rid in m['planned_ids']:
        path=root/f'response_{rid}.json'
        if sha(path)!=m['output_sha256'][path.name]:raise ValueError('prediction hash mismatch')
        records.append(json.loads(path.read_text()))
    gt={}
    for line in Path(args.annotations).open():
        a=json.loads(line);rid=str(a['id'])
        if rid in m['planned_ids']:
            if rid in gt:raise ValueError('duplicate GT')
            gt[rid]=a
    sources=sorted(set(str(r['source_id']) for r in records));rows=[]
    for r in records:
        a=gt[str(r['id'])]
        import hashlib
        if hashlib.sha256(a['response'].encode()).hexdigest()!=r['response_sha256'] or str(a['source_id'])!=str(r['source_id']):raise ValueError('GT mismatch')
        y,_,_=annotation_targets(r['offsets'],a['labels'])
        pairs=int(y.sum()*(len(y)-y.sum()))
        if pairs:
            rows.append(dict(id=r['id'],source_id=r['source_id'],pairs=pairs,
                auroc={k:weighted_metrics(y,v)['auroc'] for k,v in r['scores'].items()}))
    if not rows:raise ValueError('no mixed-class responses')
    names=list(rows[0]['auroc']);weights=np.array([r['pairs'] for r in rows])
    auc=np.array([[r['auroc'][k] for k in names] for r in rows]);primary=names.index(args.primary)
    rng=np.random.default_rng(20260914);boot=[]
    for _ in range(500):
        counts=np.bincount(rng.integers(0,len(sources),len(sources)),minlength=len(sources))
        w=weights*np.array([counts[sources.index(str(r['source_id']))] for r in rows])
        if w.sum():boot.append(np.average(auc,axis=0,weights=w))
    boot=np.array(boot)
    report=dict(scope='development within-answer localization; not generalization',
        prediction_manifest_sha256=sha(root/'manifest.json'),annotation_sha256=sha(args.annotations),
        code_sha256=sha(__file__),responses=len(records),mixed_responses=len(rows),
        metrics={name:dict(pair_weighted_auroc=float(np.average(auc[:,i],weights=weights)),macro_auroc=float(auc[:,i].mean()),
            primary_minus_this_source_bootstrap_95_ci=np.quantile(boot[:,primary]-boot[:,i],[.025,.975]).tolist()) for i,name in enumerate(names)},
        per_response=rows)
    serialized=json.dumps(report,indent=2,allow_nan=False)
    with Path(args.output).open('x') as f:f.write(serialized+'\n')
    Path(str(args.output)+'.executed_code.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='per_response'}))


if __name__=='__main__':
    main()
