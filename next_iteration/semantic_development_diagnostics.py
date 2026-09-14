"""Read-only frozen dev score diagnostics, not fitting or a new score candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from .confirmation_assess_scoped import selected_annotations
from .semantic_assess import frozen_records


def metric(y, s):
    return dict(auroc=float(roc_auc_score(y,s)), auprc=float(average_precision_score(y,s))) if len(set(y))==2 else None


def binary_zero(y, s):
    predicted=s>0  # A/B argmax, fixed by class semantics; never label-calibrated.
    tp=int((predicted & (y==1)).sum());fp=int((predicted & (y==0)).sum())
    fn=int((~predicted & (y==1)).sum());tn=int((~predicted & (y==0)).sum())
    return dict(threshold=0,tp=tp,fp=fp,fn=fn,tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--p5',required=True);p.add_argument('--p6',required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    m6,r6=frozen_records(args.p6);m5,r5=frozen_records(args.p5)
    assert m6['settings']['phase']==m5['settings']['phase']=='development'
    assert m6['planned_ids']==m5['planned_ids'] and len(r6)==32
    gt=selected_annotations(args.annotations,set(m6['planned_ids']))
    p5={str(r['id']):r for r in r5}
    ys=[];onsets=[];sources=[];s5=[];s6=[];spans=[];words=[]
    for r in r6:
        rid=str(r['id']);a=gt[rid];r5=p5[rid]
        assert r5['offsets']==r['offsets'] and r5['token_ids']==r['token_ids']
        assert hashlib.sha256(a['response'].encode()).hexdigest()==r['response_sha256']
        labels=a['labels']; y=np.zeros(len(r['offsets']),dtype=int);first=np.zeros_like(y)
        for label in labels:
            idx=[i for i,(lo,hi) in enumerate(r['offsets']) if lo<label['end'] and hi>label['start']]
            if not idx:raise ValueError('unmapped span')
            y[idx]=1;first[idx[0]]=1
            spans.append(dict(id=rid,source_id=r['source_id'],task=r['task'],
                text=a['response'][label['start']:label['end']],start=label['start'],end=label['end'],
                context=a['response'][max(0,label['start']-100):label['end']+100],
                p5=[r5['scores']['localized_source_risk'][i] for i in idx],
                p6=[r['scores']['reviewed_source_risk'][i] for i in idx],
                draft_audit=r['draft_audit']))
        for j,(lo,hi) in enumerate(r['word_spans']):
            z5=r5['word_results'][j]['logits'];z6=r['word_results'][j]['logits']
            words.append(dict(id=rid,source_id=r['source_id'],task=r['task'],start=lo,end=hi,
                word=a['response'][lo:hi],human_span_overlap=any(lo<l['end'] and hi>l['start'] for l in labels),
                p5=z5[1]-z5[0],p6=z6[1]-z6[0]))
        ys.extend(y);onsets.extend(first);sources.extend([str(r['source_id'])]*len(y))
        s5.extend(r5['scores']['localized_source_risk']);s6.extend(r['scores']['reviewed_source_risk'])
    y,first,src=map(np.asarray,(ys,onsets,sources)); scores={'P5':np.asarray(s5),'P6':np.asarray(s6)}
    unique=sorted(set(src)); jackknife=[]
    for sid in unique:
        keep=src!=sid
        out={name:metric(y[keep],s[keep]) for name,s in scores.items()}
        jackknife.append(dict(removed_source=sid,tokens=int(keep.sum()),positives=int(y[keep].sum()),metrics=out))
    operating={}
    for name,s in scores.items():
        fpr,tpr,threshold=roc_curve(y,s,drop_intermediate=False)
        operating[name]={str(b):float(tpr[fpr<=b].max()) for b in (.01,.05,.10)}
    subsets={}
    for name,positive in [('onset_vs_normal',first.astype(bool)),('continuation_vs_normal',(y==1)&(first==0))]:
        keep=(y==0)|positive
        subsets[name]=dict(diagnostic_subset_not_primary=True,tokens=int(keep.sum()),positives=int(positive.sum()),
                          metrics={n:metric(positive[keep].astype(int),s[keep]) for n,s in scores.items()})
    report=dict(scope='original development32 only; all results descriptive, no new score/candidate/threshold calibration',
        annotation_rows_decoded=len(gt),responses=32,sources=len(unique),tokens=len(y),positives=int(y.sum()),
        all_token={n:metric(y,s) for n,s in scores.items()},fixed_zero_decision={n:binary_zero(y,s) for n,s in scores.items()},
        low_fpr_roc_envelope=operating,
        low_fpr_warning='Descriptive empirical ROC envelope using development labels; NOT a deployable threshold or confirmation endpoint',
        leave_one_source_out=jackknife,diagnostic_subsets=subsets,all_annotated_spans=spans,all_word_records=words,
        manifest_sha256={n:hashlib.sha256((Path(path)/'manifest.json').read_bytes()).hexdigest() for n,path in [('P5',args.p5),('P6',args.p6)]},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with Path(args.output).open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    Path(args.output+'.executed_code.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:report[k] for k in ['all_token','fixed_zero_decision','low_fpr_roc_envelope','diagnostic_subsets','leave_one_source_out']}))


if __name__=='__main__':main()
