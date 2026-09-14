"""Fit natural-source-disjoint structural fusion, freeze predictions, then evaluate test."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from .features import ARMS,NAMES,features,select_sources


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def annotations(path,wanted):
    pattern=re.compile(rb'^\s*\{\s*"id"\s*:\s*(?:"([0-9]+)"|([0-9]+))\s*,')
    result={}
    with Path(path).open('rb') as f:
        for raw in f:
            match=pattern.match(raw)
            if not match:raise ValueError('annotation schema must be ID-first')
            rid=(match.group(1) or match.group(2)).decode()
            if rid in wanted:
                if rid in result:raise ValueError('duplicate annotation')
                result[rid]=json.loads(raw)
    if set(result)!=set(wanted):raise ValueError('missing scoped annotations')
    return result


def targets(row,offsets,gold):
    if str(gold['source_id'])!=row['source_id'] or gold['split']!=row['official_split']:
        raise ValueError('annotation source/split mismatch')
    if hashlib.sha256(gold['response'].encode()).hexdigest()!=row['response_sha256']:
        raise ValueError('annotation response identity mismatch')
    y=np.zeros(len(offsets),int);onset=y.copy()
    for span in gold['labels']:
        overlap=(offsets[:,0]<span['end']) & (offsets[:,1]>span['start'])
        if not overlap.any() or span['end']<=span['start']:raise ValueError('unmapped annotation')
        y[overlap]=1;onset[np.flatnonzero(overlap)[0]]=1
    return {'error':y,'onset':onset}


def fit(x,y,w,cx,cy,cw):
    mean=np.average(x,axis=0,weights=w);std=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=w)),1e-5)
    head=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-6,random_state=20260914)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');head.fit((x-mean)/std,y,sample_weight=w)
    raw=head.decision_function((cx-mean)/std)
    def objective(ab):
        z=ab[0]*raw+ab[1]
        return np.average(np.logaddexp(0,z)-cy*z,weights=cw)
    cal=minimize(objective,[1.,0.],method='L-BFGS-B',bounds=[(.05,20.),(-20.,20.)])
    if not cal.success:raise RuntimeError(str(cal.message))
    return dict(mean=mean.tolist(),std=std.tolist(),coef=head.coef_[0].tolist(),intercept=float(head.intercept_[0]),
        calibration=cal.x.tolist(),iterations=int(head.n_iter_[0]),warnings=[str(c.message) for c in caught])


def predict(model,x):
    z=((x-np.array(model['mean']))/np.array(model['std']))@np.array(model['coef'])+model['intercept']
    return expit(model['calibration'][0]*z+model['calibration'][1])


def metrics(y,p,probability=False):
    result=dict(tokens=len(y),positives=int(y.sum()),auroc=None,ap=None)
    if len(np.unique(y))==2:
        result.update(auroc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))
    if probability:
        q=np.clip(p,1e-7,1-1e-7)
        result.update(logloss=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),brier=float(np.mean((p-y)**2)))
    return result


def bootstrap(y,scores,source_ids,replicates=200):
    sources=sorted(set(source_ids));blocks=[np.flatnonzero(source_ids==s) for s in sources]
    rng=np.random.default_rng(20260914);out={arm:[] for arm in ['instant','raw_entropy','raw_remote_history_excl16_minus_source','raw_negative_margin']}
    for _ in range(replicates):
        idx=np.concatenate([blocks[i] for i in rng.integers(len(sources),size=len(sources))]);yy=y[idx]
        if len(np.unique(yy))<2:continue
        main=metrics(yy,scores['combined'][idx])
        for arm in out:
            b=metrics(yy,scores[arm][idx]);out[arm].append([main['auroc']-b['auroc'],main['ap']-b['ap']])
    return {arm:dict(valid=len(values),auroc_delta_ci95=np.quantile(values,[.025,.975],axis=0)[:,0].tolist(),
        ap_delta_ci95=np.quantile(values,[.025,.975],axis=0)[:,1].tolist()) for arm,values in out.items() if values}


def run(args):
    started=time.monotonic();data,out=Path(args.features),Path(args.output)
    manifest=json.loads((data/'manifest.json').read_text())
    if not manifest.get('complete') or manifest['completed']!=989 or sha(data/'records.jsonl')!=manifest['records_sha256']:
        raise ValueError('complete frozen 989-record features required')
    rows=[json.loads(s) for s in (data/'records.jsonl').open()]
    if len(rows)!=989 or len({r['id'] for r in rows})!=989:raise ValueError('record count/uniqueness')
    split=select_sources(rows);out.mkdir(parents=True,exist_ok=False)
    code={}
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(p,out/('executed_'+p.name));code[p.name]=sha(p)
    write(out/'protocol_freeze.json',dict(feature_manifest_sha256=sha(data/'manifest.json'),split=split,
        code_sha256=code,feature_names=NAMES,arms=ARMS,labels_read=False,seed=20260914,
        timing='all features causal <=t; pre next-token distributions; no oracle onset or total length'))
    offsets={};slices={};xx=[];index=0
    for r in rows:
        p=data/(r['id']+'.npz')
        if sha(p)!=r['artifact_sha256']:raise ValueError('feature hash')
        with np.load(p) as a:
            x=features(a['values']);offsets[r['id']]=a['offsets'].copy()
        if len(x)!=r['tokens']:raise ValueError('feature length')
        xx.append(x);slices[r['id']]=slice(index,index+len(x));index+=len(x)
    x=np.concatenate(xx)
    wanted={r['id'] for r in rows if split[r['id']]!='test'}
    gold=annotations(args.annotations,wanted);labels={r['id']:targets(r,offsets[r['id']],gold[r['id']]) for r in rows if r['id'] in wanted}
    def arrays(part,target):
        selected=[r for r in rows if split[r['id']]==part]
        idx=np.concatenate([np.arange(slices[r['id']].start,slices[r['id']].stop) for r in selected])
        y=np.concatenate([labels[r['id']][target] for r in selected])
        counts={s:sum(r['tokens'] for r in selected if r['source_id']==s) for s in {r['source_id'] for r in selected}}
        w=np.concatenate([np.full(r['tokens'],1/counts[r['source_id']]) for r in selected]);w*=len(w)/w.sum()
        return idx,y,w
    models={};predictions={};thresholds={}
    for target in ['error','onset']:
        ti,ty,tw=arrays('train',target);ci,cy,cw=arrays('calibration',target)
        models[target]={};thresholds[target]={}
        for arm,cols in ARMS.items():
            m=fit(x[ti][:,cols],ty,tw,x[ci][:,cols],cy,cw);p=predict(m,x[:,cols])
            threshold=float(np.quantile(p[ci][cy==0],.95,method='higher'))
            m['calibration_negative_fpr']=float(np.mean(p[ci][cy==0]>threshold))
            models[target][arm]=m;predictions[target+'__'+arm]=p;thresholds[target][arm]=threshold
            print(json.dumps(dict(target=target,arm=arm,iterations=m['iterations'],warnings=m['warnings'])),flush=True)
    write(out/'models.json',models);write(out/'thresholds.json',thresholds)
    write(out/'prediction_index.json',[dict(id=r['id'],source_id=r['source_id'],split=split[r['id']],
        start=slices[r['id']].start,end=slices[r['id']].stop) for r in rows])
    np.savez_compressed(out/'predictions.npz',**predictions,raw_entropy=x[:,0],raw_remote_history_excl16_minus_source=x[:,5],
        raw_negative_margin=x[:,1],raw_position=np.concatenate([np.log1p(np.arange(r['tokens'])) for r in rows]))
    write(out/'prediction_freeze.json',dict(test_labels_read=False,
        hashes={name:sha(out/name) for name in ['models.json','thresholds.json','predictions.npz','prediction_index.json','protocol_freeze.json']}))
    test=[r for r in rows if split[r['id']]=='test'];gold=annotations(args.annotations,{r['id'] for r in test})
    labels.update({r['id']:targets(r,offsets[r['id']],gold[r['id']]) for r in test})
    idx=np.concatenate([np.arange(slices[r['id']].start,slices[r['id']].stop) for r in test]);results={}
    source_ids=np.concatenate([np.repeat(r['source_id'],r['tokens']) for r in test])
    with np.load(out/'predictions.npz') as saved:
        for target in ['error','onset']:
            y=np.concatenate([labels[r['id']][target] for r in test])
            scores={arm:saved[target+'__'+arm][idx] for arm in ARMS}
            scores.update({name:saved[name][idx] for name in ['raw_entropy','raw_remote_history_excl16_minus_source','raw_negative_margin','raw_position']})
            results[target]={'metrics':{a:metrics(y,p,a in ARMS) for a,p in scores.items()},'threshold_metrics':{},'within_response_auroc':{}}
            for arm in ARMS:
                p=scores[arm];thr=thresholds[target][arm];alarms=p>thr
                normal=[];within=[]
                for r in test:
                    yy=labels[r['id']][target];pp=saved[target+'__'+arm][slices[r['id']]]
                    if not yy.any():normal.append(bool((pp>thr).any()))
                    elif not yy.all():within.append(float(roc_auc_score(yy,pp)))
                results[target]['within_response_auroc'][arm]=float(np.mean(within)) if within else None
                results[target]['threshold_metrics'][arm]=dict(threshold=thr,recall=float(alarms[y==1].mean()),
                    negative_token_fpr=float(alarms[y==0].mean()),normal_response_any_alarm=float(np.mean(normal)) if normal else None)
            for arm in scores:
                if arm in ARMS:continue
                within=[]
                for r in test:
                    yy=labels[r['id']][target];pp=saved[arm][slices[r['id']]]
                    if yy.any() and not yy.all():within.append(float(roc_auc_score(yy,pp)))
                results[target]['within_response_auroc'][arm]=float(np.mean(within)) if within else None
            results[target]['source_bootstrap']=bootstrap(y,scores,source_ids)
    results['scope']=dict(train_responses=sum(s=='train' for s in split.values()),
        calibration_responses=sum(s=='calibration' for s in split.values()),test_responses=len(test),test_sources=len(set(source_ids)),
        observer=True,official_test_previously_used_by_historical_research=True,no_new_gpu_forwards=True)
    write(out/'evaluation.json',results)
    write(out/'complete.json',dict(complete=True,seconds=time.monotonic()-started,models=8,
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(complete=True,seconds=time.monotonic()-started,results=results['error']['metrics'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
