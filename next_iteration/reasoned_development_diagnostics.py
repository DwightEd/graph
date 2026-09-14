"""Descriptive full-development P7/P6/P5 diagnostics, never a score fit."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from .development_assess_scoped import DEV_IDS, frozen_records
from .confirmation_assess_scoped import selected_annotations
from .grounding_contrast_evaluate import sha, annotation_targets, weighted_metrics

ROOT=Path(__file__).resolve().parents[1]
COHORTS={
    'P7':('p7_reasoned_development_20260914_v1','reasoned_source_risk'),
    'P6':('p6_review_development_20260914_v1','reviewed_source_risk'),
    'P5':('p5_local_development_20260914_v3_fp32','localized_source_risk'),
}

def zero_decision(y,s):
    pred=s>0
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum())
    fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    return dict(threshold=0,tp=tp,fp=fp,fn=fn,tn=tn,
                precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    launch=json.loads((ROOT/'outputs/P7_DEVELOPMENT_LAUNCH_20260914.json').read_text())
    if launch.get('actual_subprocess_returncode')!=0:raise ValueError('P7 incomplete, no GT access')
    manifests={};lookups={}
    for name,(directory,score) in COHORTS.items():
        m,rows=frozen_records(ROOT/'outputs'/directory)
        if len(rows)!=32 or {str(r['id']) for r in rows}!=DEV_IDS or m['settings']['phase']!='development':
            raise ValueError('only exact complete original development32 is allowed')
        manifests[name]=sha(ROOT/'outputs'/directory/'manifest.json')
        lookups[name]={str(r['id']):r for r in rows}
    for rid in sorted(DEV_IDS):
        refs=[lookups[name][rid] for name in COHORTS]
        for key in ['offsets','token_ids','response_sha256','source_id','task','word_spans']:
            if any(r[key]!=refs[0][key] for r in refs[1:]):raise ValueError('cross-method alignment mismatch')
    annotation_path=ROOT.parents[1]/'data/RAGTruth/dataset/response.jsonl'
    gt=selected_annotations(annotation_path,DEV_IDS)
    labels=[];sources=[];onsets=[];scores={n:[] for n in COHORTS};spans=[];drafts=[];words=[]
    for rid in sorted(DEV_IDS):
        r=lookups['P7'][rid];a=gt[rid]
        if hashlib.sha256(a['response'].encode()).hexdigest()!=r['response_sha256'] or str(a['source_id'])!=str(r['source_id']):
            raise ValueError('annotation identity mismatch')
        y,onset,_=annotation_targets(r['offsets'],a['labels'])
        labels.extend(y);onsets.extend(onset);sources.extend([str(r['source_id'])]*len(y))
        for name,(_,score) in COHORTS.items():
            s=lookups[name][rid]['scores'][score]
            if len(s)!=len(y) or not np.isfinite(s).all():raise ValueError('invalid full-token score')
            scores[name].extend(s)
        for label in a['labels']:
            lo,hi=label['start'],label['end']
            indices=[i for i,(start,end) in enumerate(r['offsets']) if start<hi and end>lo]
            if not indices:raise ValueError('unmapped annotation span')
            spans.append(dict(id=rid,source_id=r['source_id'],task=r['task'],start=lo,end=hi,
                text=a['response'][lo:hi],context=a['response'][max(0,lo-100):hi+100],
                token_risks={n:[lookups[n][rid]['scores'][key][i] for i in indices] for n,(_,key) in COHORTS.items()}))
        for i,(lo,hi) in enumerate(r['word_spans']):
            words.append(dict(id=rid,source_id=r['source_id'],task=r['task'],start=lo,end=hi,
                word=a['response'][lo:hi],human_span_overlap=any(lo<l['end'] and hi>l['start'] for l in a['labels']),
                risks={n:lookups[n][rid]['word_results'][i]['logits'][1]-lookups[n][rid]['word_results'][i]['logits'][0] for n in COHORTS}))
        drafts.append(dict(id=rid,source_id=r['source_id'],task=r['task'],
            p6=lookups['P6'][rid]['draft_audit'],p7=r['draft_audit']))
    y,src,onset=map(np.asarray,(labels,sources,onsets));scores={n:np.asarray(s) for n,s in scores.items()}
    if len(y)!=5170:raise ValueError('full denominator mismatch')
    jackknife=[]
    for sid in sorted(set(src)):
        mask=src!=sid
        jackknife.append(dict(removed_source=sid,tokens=int(mask.sum()),positives=int(y[mask].sum()),
            metrics={n:weighted_metrics(y[mask],s[mask]) for n,s in scores.items()}))
    subsets={}
    for name,positive in [('onset_vs_normal',onset==1),('continuation_vs_normal',(y==1)&(onset==0))]:
        keep=(y==0)|positive
        subsets[name]=dict(diagnostic_only=True,tokens=int(keep.sum()),positives=int(positive.sum()),
            metrics={n:weighted_metrics(positive[keep].astype(int),s[keep]) for n,s in scores.items()})
    report=dict(scope='original development32, descriptive only; no tuning, no new score, not a confirmation endpoint or causal mediation test',
        responses=32,sources=len(set(src)),tokens=len(y),positives=int(y.sum()),annotation_rows_decoded=len(gt),
        manifest_sha256=manifests,annotation_sha256=sha(annotation_path),code_sha256=sha(__file__),
        all_token={n:weighted_metrics(y,s) for n,s in scores.items()},
        fixed_zero_decision={n:zero_decision(y,s) for n,s in scores.items()},
        leave_one_source_out=jackknife,diagnostic_subsets=subsets,
        p7_draft_engineering=dict(forced_close=sum(d['p7']['closure']['forced_close'] for d in drafts),
            budget_exhausted=sum(d['p7']['reasoning_budget_exhausted'] for d in drafts),
            final_truncated=sum(d['p7']['truncated'] for d in drafts)),
        all_annotated_spans=spans,all_word_records=words,all_audits=drafts)
    with Path(args.output).open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    Path(args.output+'.executed_code.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ['all_annotated_spans','all_word_records','all_audits']}))

if __name__=='__main__':main()
