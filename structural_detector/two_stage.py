"""S11: entropy entry -> head-resolved inherited risk -> continuation readout.

No GNN/backprop through the language model. Supervised logistic heads are explicit;
the entropy-tail transport controls use no hallucination labels for scoring.
"""
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from .local_capture import digest, write
from .transport import MODES, continuation_features, mixture, onset_features, propagate
from binding_detector.evaluation import evaluate_records, label_views


def fit_linear(x, y, w, nonnegative=()):
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(w)
    if len(np.unique(y)) != 2:
        raise ValueError('training fold requires both classes; expand the source roster')
    mean = np.average(x, axis=0, weights=w)
    std = np.maximum(np.sqrt(np.average((x-mean)**2, axis=0, weights=w)), 1e-5)
    if nonnegative:
        xx=(x-mean)/std; n=xx.shape[1]; total=float(w.sum())
        def objective(ab):
            z=xx@ab[:n]+ab[n]; delta=(expit(z)-y)*w/total
            loss=float(np.dot(w,np.logaddexp(0,z)-y*z)/total+.5*np.dot(ab[:n],ab[:n])/total)
            grad=np.r_[xx.T@delta+ab[:n]/total,delta.sum()]
            return loss,grad
        start=np.zeros(n+1);prevalence=np.average(y,weights=w)
        start[-1]=np.log(prevalence/(1-prevalence))
        bounds=[(0,None) if j in nonnegative else (None,None) for j in range(n)]+[(None,None)]
        fitted=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=bounds,
                        options={'maxiter':1000,'ftol':1e-10,'gtol':1e-7})
        if not fitted.success:raise RuntimeError('continuation optimizer: '+str(fitted.message))
        return dict(mean=mean.tolist(),std=std.tolist(),coef=fitted.x[:n].tolist(),
                    intercept=float(fitted.x[n]),iterations=int(fitted.nit),warnings=[],
                    nonnegative_columns=list(nonnegative))
    model = LogisticRegression(C=1.,max_iter=1000,tol=1e-6,solver='lbfgs',random_state=20260914)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit((x-mean)/std, y, sample_weight=w)
    return dict(mean=mean.tolist(),std=std.tolist(),coef=model.coef_[0].tolist(),
                intercept=float(model.intercept_[0]),iterations=int(model.n_iter_[0]),
                warnings=[str(c.message) for c in caught])


def predict(model, x):
    return expit(((x-np.asarray(model['mean']))/model['std'])@np.asarray(model['coef'])+model['intercept'])


def splits(rows):
    source = {part:{str(r['source_id']) for r in rows if r['official_split']==part} for part in ('train','test')}
    if source['train'] & source['test']:
        raise ValueError('official source overlap')
    order = sorted(source['train'],key=lambda s:hashlib.sha256(f'20260914:{s}'.encode()).hexdigest())
    cut = int(.8*len(order))
    if cut < 3 or cut == len(order) or not source['test']:
        raise ValueError('need >=4 train sources and an official test set')
    train = set(order[:cut])
    return {str(r['id']):('test' if str(r['source_id']) in source['test'] else
                         'train' if str(r['source_id']) in train else 'calibration') for r in rows}


def read_labels(path, rows, wanted):
    """Parse only requested labels. Training calls never parse official-test JSON."""
    by_id = {str(r['id']):r for r in rows}; found = {}
    pattern = re.compile(rb'^\s*\{\s*"id"\s*:\s*(?:"([0-9]+)"|([0-9]+))\s*,')
    with Path(path).open('rb') as stream:
        for raw in stream:
            match = pattern.match(raw)
            if not match: raise ValueError('RAGTruth annotation file must use ID-first records')
            rid = (match.group(1) or match.group(2)).decode()
            if rid not in wanted: continue
            if rid in found: raise ValueError('duplicate annotation')
            g, r = json.loads(raw), by_id[rid]
            if str(g['source_id']) != str(r['source_id']) or g['split'] != r['official_split'] or hashlib.sha256(g['response'].encode()).hexdigest()!=r['response_sha256']:
                raise ValueError('annotation/input mismatch: '+rid)
            views = label_views(r['offsets'],g['labels'],len(g['response']))
            found[rid] = (views['all_error'][0],views['span_onset_full_stream'][0])
    if set(found)!=set(wanted): raise ValueError('missing scoped labels')
    return found


def source_weights(rows, sizes):
    counts = {}
    for r,n in zip(rows,sizes): counts[str(r['source_id'])]=counts.get(str(r['source_id']),0)+n
    w = np.concatenate([np.full(n,1/counts[str(r['source_id'])]) for r,n in zip(rows,sizes) if n])
    return w * len(w) / w.sum()


def choose_channels(rows, data, maximum):
    """Unlabelled TRAIN-only variability; fixed for real/null readouts alike."""
    means, squares = [], []
    for r in rows:
        a = data[str(r['id'])]
        if 'local_moments' in a:
            m,s = a['local_moments']; means.append(m); squares.append(s)
        else:
            local = a['edges'].sum(-1)
            means.append(local.mean(0)); squares.append((local*local).mean(0))
    # Equal-source weighting even when one source supplies several answers.
    counts = {str(r['source_id']):sum(str(z['source_id'])==str(r['source_id']) for z in rows) for r in rows}
    w = np.array([1/counts[str(r['source_id'])] for r in rows])
    variance = np.average(squares,0,weights=w)-np.average(means,0,weights=w)**2
    n = len(variance) if maximum == 0 else min(maximum,len(variance))
    if n < 1: raise ValueError('positive max_channels or 0 for all required')
    selected = np.argsort(-variance,kind='stable')[:n]
    return np.sort(selected)


def train_entry(train, data, labels):
    """Out-of-source seed probabilities: training continuation never sees gold past errors."""
    def build(rows):
        x = [onset_features(data[str(r['id'])]['values']) for r in rows]
        y = [labels[str(r['id'])][1] for r in rows]
        return np.concatenate(x),np.concatenate(y),source_weights(rows,[len(v) for v in y])
    ordered = sorted({str(r['source_id']) for r in train},key=lambda s:digest(s))
    folds = {s:i%3 for i,s in enumerate(ordered)}
    seeds = {}
    for fold in range(3):
        fit_rows = [r for r in train if folds[str(r['source_id'])]!=fold]
        model = fit_linear(*build(fit_rows))
        for r in train:
            if folds[str(r['source_id'])]==fold:
                seeds[str(r['id'])]=predict(model,onset_features(data[str(r['id'])]['values']))
    return fit_linear(*build(train)),seeds,folds


def normal_answer_threshold(probabilities, labels, rows, alpha=.05):
    """Quantile of maximum risk on fully correct calibration answers, never test."""
    maxima = [float(np.max(probabilities[str(r['id'])])) for r in rows if not labels[str(r['id'])][0].any()]
    k = int(np.ceil((len(maxima)+1)*(1-alpha)))
    threshold = float(np.sort(maxima)[k-1]) if 0<k<=len(maxima) else None
    return dict(threshold=threshold,normal_calibration_answers=len(maxima),alpha=alpha,
                scope='answer maximum, finite-sample rank; not a distribution-shift guarantee')


def train_and_evaluate(features, annotations, output, *, max_channels=64, bootstrap=200, oracle=True):
    root,out = Path(features),Path(output)
    if not json.loads((root/'complete.json').read_text())['complete']: raise ValueError('capture incomplete')
    rows = json.loads((root/'records.json').read_text())
    # A separate fit for each task/generator; never pool their test denominators silently.
    groups = sorted({(r['task'],r['generator']) for r in rows})
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(method='s11_local_ancestry',max_channels=max_channels,controls=MODES,
              supervision='entropy-entry and conditional-continuation logistic readouts use natural train labels',
              raw_controls='entropy-tail seeds and raw transport use no labels',oracle_after_prediction_freeze=oracle,
              baseline='S10 retained; instant LR retrained on identical recollected inputs',historical_test_reuse=True))
    for task,generator in groups:
        group_rows = [r for r in rows if (r['task'],r['generator'])==(task,generator)]
        run_group(root,out/(task+'__'+generator),group_rows,annotations,max_channels,bootstrap,oracle)
    write(out/'complete.json',dict(complete=True,groups=len(groups),new_llm_forwards=0))


def run_group(root,out,rows,annotations,max_channels,bootstrap,oracle):
    out.mkdir()
    partition=splits(rows); data={}
    channels=None
    for r in rows:
        rid=str(r['id'])
        with np.load(root/(rid+'.npz'),allow_pickle=False) as f:
            data[rid]={k:f[k].copy() for k in ('values','offsets','channels')}
            if partition[rid]=='train':
                local=f['edges'].sum(-1,dtype=np.float64)
                data[rid]['local_moments']=(local.mean(0),(local*local).mean(0))
        if channels is not None and not np.array_equal(channels,data[rid]['channels']): raise ValueError('physical channel layout differs')
        channels=data[rid]['channels']; r['offsets']=data[rid]['offsets'].tolist()
    train=[r for r in rows if partition[str(r['id'])]=='train']
    cal=[r for r in rows if partition[str(r['id'])]=='calibration']
    test=[r for r in rows if partition[str(r['id'])]=='test']
    labels=read_labels(annotations,rows,{str(r['id']) for r in train+cal})
    selected=choose_channels(train,data,max_channels)
    for rid,a in data.items():
        with np.load(root/(rid+'.npz'),allow_pickle=False) as f:
            a['edges']=f['edges'][:,selected];a['evidence']=f['evidence'][:,selected]
    entry,oof,folds=train_entry(train,data,labels)
    seeds={str(r['id']):oof[str(r['id'])] if r in train else predict(entry,onset_features(data[str(r['id'])]['values'])) for r in rows}
    # Only TWO conditional targets, not a third classifier fed gold onset states.
    models={}; predicted={str(r['id']):{'onset':seeds[str(r['id'])]} for r in rows if r not in train}
    traces={}
    for mode in MODES:
        xx,yy,rr,sizes=[],[],[],[]
        for r in train:
            rid=str(r['id']);a=data[rid];y,o=labels[rid]
            x,_,_,_=continuation_features(seeds[rid],a['edges'],a['evidence'],mode)
            keep=~o
            xx.append(x[keep]);yy.append(y[keep]);rr.append(r);sizes.append(int(keep.sum()))
        model=fit_linear(np.concatenate(xx),np.concatenate(yy),source_weights(rr,sizes),
                         nonnegative=tuple(range(len(selected))));models[mode]=model
        for r in cal+test:
            rid=str(r['id']);a=data[rid]
            x,u,state,parent=continuation_features(seeds[rid],a['edges'],a['evidence'],mode)
            conditional=predict(model,x)
            predicted[rid]['error__'+mode]=mixture(seeds[rid],conditional)
            predicted[rid]['continuation__'+mode]=(1-seeds[rid])*conditional
            if mode=='real':traces[rid]=dict(inherited=u.astype(np.float32),ancestry=state.astype(np.float32),dominant_parent=parent)
        print(f'{out.name}: fit continuation {mode}; iterations={model["iterations"]}',flush=True)
    # Same-cohort instantaneous supervised baseline, not the old batch's metrics.
    def instant(a):
        v=a['values'];return np.column_stack((v,v[:,3]-v[:,2]))
    for target,column in (('error',0),('onset',1)):
        models['instant_'+target]=fit_linear(np.concatenate([instant(data[str(r['id'])]) for r in train]),
            np.concatenate([labels[str(r['id'])][column] for r in train]),source_weights(train,[len(labels[str(r['id'])][0]) for r in train]))
        for r in cal+test:
            rid=str(r['id']);predicted[rid]['instant_'+target]=predict(models['instant_'+target],instant(data[rid]))
    href=np.sort(np.concatenate([np.quantile(np.concatenate([data[str(r['id'])]['values'][:,0]
                      for r in train if str(r['source_id'])==sid]),np.linspace(0,1,32))
                      for sid in sorted({str(r['source_id']) for r in train})]))
    for r in cal+test:
        rid=str(r['id']);a=data[rid];v=a['values'];p=predicted[rid]
        p.update(raw_entropy=v[:,0],raw_negative_margin=v[:,1],raw_remote_displacement=v[:,3]-v[:,2],raw_position=np.log1p(np.arange(len(v))))
        tail=np.clip((np.searchsorted(href,v[:,0],side='right')/len(href)-.95)/.05,0,1)
        p['unlabeled_seed']=tail
        for mode in ('real','uniform'):
            _,state,_=propagate(tail,a['edges'],mode)
            p['unlabeled_'+mode]=state.max(1)
        np.savez_compressed(out/(rid+'.npz'),offsets=r['offsets'],channels=channels[selected],**p,**traces[rid])
    thresholds={}
    for name in ('onset','error__real','error__uniform','error__permuted','error__one_hop','error__no_edges','instant_error','instant_onset'):
        thresholds[name]=normal_answer_threshold({rid:p[name] for rid,p in predicted.items()},labels,cal)
    write(out/'models.json',dict(entry=entry,continuation=models,selected_channels=channels[selected].tolist(),
              split=partition,oof_source_folds=folds,entry_train_features='H,dH',tail_quantile=.95))
    write(out/'thresholds.json',thresholds)
    write(out/'prediction_freeze.json',dict(complete=True,test_labels_read=False,
              files={r['id']:hashlib.sha256((out/(str(r['id'])+'.npz')).read_bytes()).hexdigest() for r in cal+test}))
    # Label-bearing work starts here, AFTER all test scores and thresholds are frozen.
    test_labels=read_labels(annotations,rows,{str(r['id']) for r in test});labels.update(test_labels)
    names=('span_onset_full_stream','span_onset_vs_normal','first_error_full_stream','first_error_until_first')
    primary={'default':'error__real',**{n:'onset' for n in names}}
    controls={'default':('error__uniform','error__permuted','error__one_hop','error__no_edges','instant_error','raw_entropy'),
              **{n:('instant_onset','raw_entropy','raw_negative_margin') for n in names}}
    result=evaluate_records(test,{str(r['id']):predicted[str(r['id'])] for r in test},annotations,
                             primary=primary,controls=controls,bootstrap=bootstrap)
    alarm={}
    for name,item in thresholds.items():
        threshold=item['threshold']; normal=[];first=[];before=[];cont_hits=[]
        if threshold is not None:
            for r in test:
                rid=str(r['id']);y,o=labels[rid];a=predicted[rid][name]>threshold
                if not y.any():normal.append(bool(a.any()))
                else:
                    t=np.flatnonzero(y)[0];first.append(bool(a[t]));before.append(bool(a[:t].any()))
                cont_hits.extend(a[y&~o].tolist())
        avg=lambda x:float(np.mean(x)) if x else None
        alarm[name]={**item,'normal_answer_any_alarm':avg(normal),'first_error_exact_recall':avg(first),
                     'pre_first_any_alarm':avg(before),'continuation_recall':avg(cont_hits)}
    result['answer_budget']=alarm
    write(out/'evaluation.json',result)
    if oracle:
        diagnostic={}
        for r in test:
            rid=str(r['id']);y,o=labels[rid];w=data[rid]['edges'];d={}
            for mode in ('real','uniform','permuted'):
                u,_,_=propagate(o.astype(float),w,mode);d['oracle_seed_'+mode]=u.max(1)
            u,_,_=propagate(y.astype(float),w,'one_hop');d['oracle_history_real']=u.max(1)
            diagnostic[rid]=d;np.savez_compressed(out/(rid+'.oracle.npz'),**d)
        diag=evaluate_records(test,diagnostic,annotations,primary='oracle_seed_real',
                              controls=('oracle_seed_uniform','oracle_seed_permuted','oracle_history_real'),bootstrap=bootstrap)
        diag['warning']='USES TEST LABELS: mechanism diagnostic, NOT detector performance or guaranteed upper bound'
        write(out/'oracle_diagnostic.json',diag)
    write(out/'complete.json',dict(complete=True,train=len(train),calibration=len(cal),test=len(test),channels=len(selected)))
    print(json.dumps(dict(group=out.name,all_error=result['views']['all_error']['metrics']['error__real'],
                          continuation=result['views']['continuation_vs_normal']['metrics']['error__real'])),flush=True)
