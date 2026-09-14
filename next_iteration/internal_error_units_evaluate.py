"""N9 auxiliary-supervised binding readouts; frozen natural transfer is evaluated last."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from .internal_error_units import LAYERS, SEED, digest, token_targets, write_json
from .grounding_contrast_evaluate import weighted_metrics, annotation_targets

ARMS = ['baseline', 'nodes', 'graph', 'shuffled_graph']


def projection(layer, dim=32):
    return (np.random.default_rng(SEED+layer).standard_normal((4096,dim))/np.sqrt(dim)).astype(np.float32)


def interaction(a,b):
    a = a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-6)
    b = b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-6)
    return a*b


def features(a):
    p = int(a['prompt_length'])
    n = len(a['controls'])
    base = np.column_stack([a['controls'], a['source_mass'], a['history_mass'],
                            a['displacement'], np.log1p(np.arange(n)), a['revisit']]).astype(np.float32)
    nodes, graph, shuffled = [base], [], []
    for li in LAYERS:
        r = projection(li)
        x = a[f'nodes_{li}'].astype(np.float32) @ r
        pre, post = x[p-1:-1], x[p:]
        source = np.broadcast_to(x[a['source_mask'].astype(bool)].mean(0), post.shape)
        mlp = a[f'mlp_{li}'].astype(np.float32) @ r
        nodes.extend([pre,post,source,mlp,interaction(post,source)])
        messages = a[f'messages_{li}'].astype(np.float32) @ r
        for sink, indices in [(graph,[0,1]),(shuffled,[2,3])]:
            for k in indices:
                sink.extend([messages[:,k],interaction(post,messages[:,k])])
    node = np.concatenate(nodes,axis=1)
    result = dict(baseline=base, nodes=node, graph=np.column_stack([node,*graph]),
                  shuffled_graph=np.column_stack([node,*shuffled]))
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('invalid projected feature')
    return result


def causal_window(events, width):
    mask = np.zeros(len(events),bool)
    for i in np.flatnonzero(events):
        mask[i:min(len(mask),i+width+1)] = True
    return mask


def logloss(y,p,w=None):
    p = np.clip(p,1e-7,1-1e-7)
    return float(np.average(-y*np.log(p)-(1-y)*np.log1p(-p), weights=w))


def metrics(y,p,weights=None):
    result = weighted_metrics(y,p,weights)
    result.update(log_loss=logloss(y,p,weights),brier=float(np.average((y-p)**2,weights=weights)),
                  tokens=len(y),positives=int(np.sum(y)))
    return result


def fit_head(x,y,weight,cx,cy,cweight):
    mean = np.average(x,axis=0,weights=weight)
    std = np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weight)),1e-4)
    # C and iteration count are fixed before test; no hyperparameter search.
    head = LogisticRegression(C=1.,solver='lbfgs',max_iter=300,random_state=SEED,tol=1e-5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        head.fit((x-mean)/std,y,sample_weight=weight)
    raw = head.decision_function((cx-mean)/std)
    def objective(ab):
        z = ab[0]*raw+ab[1]
        return float(np.average(np.logaddexp(0,z)-cy*z,weights=cweight))
    cal = minimize(objective,np.array([1.,0.]),method='L-BFGS-B',bounds=[(.05,20.),(-20.,20.)])
    if not cal.success:
        raise RuntimeError('calibration optimizer failed: '+str(cal.message))
    model = dict(mean=mean.tolist(),std=std.tolist(),coef=head.coef_[0].tolist(),
        intercept=float(head.intercept_[0]),calibration=cal.x.tolist(),
        iterations=int(head.n_iter_[0]),convergence_warning=bool(caught),
        dimensions=x.shape[1],C=1.,calibration_orientation='positive slope only')
    return model


def predict(model,x):
    raw = ((x-np.array(model['mean']))/np.array(model['std'])) @ np.array(model['coef']) + model['intercept']
    return expit(model['calibration'][0]*raw+model['calibration'][1])


def fixed_fpr_threshold(y,p):
    normal = p[y==0]
    if not len(normal):
        raise ValueError('calibration has no negatives')
    return float(np.quantile(normal,.95,method='higher'))


def summarize(rows,probs,labels,events,threshold):
    yy = np.concatenate([labels[r['id']]['core'] for r in rows])
    pp = np.concatenate([probs[r['id']] for r in rows])
    weights = np.concatenate([np.full(len(labels[r['id']]['core']),1/len(labels[r['id']]['core'])) for r in rows])
    mixed, normal_fpr, per_row = [],[],[]
    for r in rows:
        y,p = labels[r['id']]['core'],probs[r['id']]
        m = metrics(y,p)
        per_row.append(m)
        if y.any() and not y.all():
            mixed.append(m['auroc'])
        if not y.any():
            normal_fpr.append(float(np.mean(p>threshold)))
    out = dict(micro=metrics(yy,pp),equal_response=metrics(yy,pp,weights),
        within_error_response_macro_auroc=float(np.mean(mixed)) if mixed else None,
        normal_response_macro_fpr=float(np.mean(normal_fpr)) if normal_fpr else None,
        threshold_5pct_calibration=threshold,
        precision_at_half=float(np.sum(yy[pp>=.5])/max(1,np.sum(pp>=.5))),
        recall_at_half=float(np.sum(yy[pp>=.5])/max(1,yy.sum())),
        recall_at_calibration_threshold=float(np.sum(yy[pp>threshold])/max(1,yy.sum())),
        onset=weighted_metrics(np.concatenate([labels[r['id']]['onset'] for r in rows]),pp),
        windows={})
    for width in [0,4,8]:
        masks=[causal_window(events[r['id']],width) for r in rows]
        mask=np.concatenate(masks)
        onsets=np.concatenate([labels[r['id']]['onset'] for r in rows])
        cascade=np.where(mask,pp,0.)
        out['windows'][str(width)] = dict(token_coverage=float(mask.mean()),
            onset_coverage=float(onsets[mask].sum()/max(1,onsets.sum())),
            core_coverage=float(yy[mask].sum()/max(1,yy.sum())),
            event_count=int(sum(np.sum(events[r['id']]) for r in rows)),
            unconditional_cascade=metrics(yy,cascade),
            unconditional_recall_at_calibration_threshold=float(np.sum(yy[(pp>threshold)&mask])/max(1,yy.sum())),
            conditional_inside_window=metrics(yy[mask],pp[mask]) if mask.any() else None,
            normal_response_window_coverage=float(np.mean([m.mean() for r,m in zip(rows,masks) if not labels[r['id']]['core'].any()])) if normal_fpr else None)
    regions={k:[] for k in ['onset','remaining_core','completion_only','background']}
    for r in rows:
        lab,pr=labels[r['id']],probs[r['id']]
        masks={'onset':lab['onset'].astype(bool),
               'remaining_core':(lab['core']>0)&(lab['onset']==0),
               'completion_only':(lab['scope']>0)&(lab['core']==0),
               'background':lab['scope']==0}
        for k,m in masks.items():
            regions[k].extend(pr[m].tolist())
    out['disjoint_region_scores']={k:dict(tokens=len(v),mean_probability=float(np.mean(v)) if v else None,
        fraction_above_threshold=float(np.mean(np.array(v)>threshold)) if v else None) for k,v in regions.items()}
    return out


def paired_binding(rows, probs, gold, events, width=None):
    """Evaluation-only candidate coordinates; source-cluster uncertainty, missed anchors score zero."""
    pairs={};coverage={'supported':[], 'misbound':[]}
    for r in rows:
        g=gold[r['id']];lo,hi=g['candidate_value']
        candidate=np.array([max(a,lo)<min(b,hi) for a,b in r['offsets']])
        window=np.ones(len(candidate),bool) if width is None else causal_window(events[r['id']],width)
        hit=bool(np.any(candidate & window))
        risk=float(np.where(window,probs[r['id']],0.)[candidate].mean())
        item=dict(risk=risk,hit=hit,coverage=float(window[candidate].mean()),source=r['source_id'])
        pairs.setdefault(g['pair_id'],{})[g['condition']]=item
        coverage[g['condition']].append(item)
    sources=sorted({r['source_id'] for r in rows})
    deltas=[];orders=[];both=[];neither=[];pair_sources=[]
    for v in pairs.values():
        a,b=v['misbound'],v['supported'];d=a['risk']-b['risk']
        deltas.append(d);orders.append(float(d>0));both.append(a['hit'] and b['hit'])
        neither.append(not a['hit'] and not b['hit']);pair_sources.append(a['source'])
    def clustered(values, ids):
        per=np.array([np.mean([v for v,k in zip(values,ids) if k==sid]) for sid in sources])
        draws=np.random.default_rng(SEED).integers(len(sources),size=(500,len(sources)))
        return dict(point=float(per.mean()),ci95=np.quantile(per[draws].mean(1),[.025,.975]).tolist())
    result=dict(pairs=len(pairs),sources=len(sources),mean_risk_delta=clustered(deltas,pair_sources),
        ordering_accuracy=clustered(orders,pair_sources),tie_fraction=float(np.mean(np.array(deltas)==0)),
        both_hit_pairs=int(sum(both)),neither_hit_pairs=int(sum(neither)),
        one_hit_pairs=int(len(both)-sum(both)-sum(neither)),
        conditional_both_hit_ordering=float(np.mean(np.array(orders)[both])) if any(both) else None,
        candidate_coverage={k:clustered([a['coverage'] for a in v],[a['source'] for a in v]) for k,v in coverage.items()},
        candidate_any_hit={k:clustered([float(a['hit']) for a in v],[a['source'] for a in v]) for k,v in coverage.items()})
    supported=[r for r in rows if gold[r['id']]['condition']=='supported']
    result['supported_response_any_event']=float(np.mean([np.any(events[r['id']]) for r in supported]))
    result['supported_events_per_response']=float(np.mean([np.sum(events[r['id']]) for r in supported]))
    return result,dict(zip(pairs,deltas)),dict(zip(pairs,pair_sources))


def localization(rows,probs,labels,threshold):
    output=[];normal=[]
    for r in rows:
        y=labels[r['id']]['core'].astype(bool);p=probs[r['id']];active=p>threshold
        if not y.any():
            normal.append(bool(active.any()));continue
        top=int(np.argmax(p));component=np.zeros(len(p),bool)
        # Highest probability token, earliest tie; retain its connected above-threshold component.
        if active[top]:
            a=b=top
            while a>0 and active[a-1]:a-=1
            while b+1<len(p) and active[b+1]:b+=1
            component[a:b+1]=True
        else:a=b=None
        truth=np.flatnonzero(y)
        output.append(dict(id=r['id'],top1_in_core=bool(y[top]),
            iou=float((component & y).sum()/max(1,(component | y).sum())),
            start_error_tokens=None if a is None else int(a-truth[0]),
            end_error_tokens=None if b is None else int(b-truth[-1]),detected=bool(component.any())))
    return dict(error_responses=len(output),top1_accuracy=float(np.mean([x['top1_in_core'] for x in output])) if output else None,
        mean_iou=float(np.mean([x['iou'] for x in output])) if output else None,
        normal_response_any_alarm=float(np.mean(normal)) if normal else None,rows=output)


def run(args):
    started=time.monotonic()
    root,inputs,out=Path(args.capture),Path(args.inputs),Path(args.output)
    cm=json.loads((root/'manifest.json').read_text())
    im=json.loads((inputs.parent/'manifest.json').read_text())
    if not cm.get('complete') or cm.get('error') or cm['planned']!=800 or cm['input_sha256']!=digest(inputs):
        raise ValueError('complete frozen 800 capture required')
    if len(set(cm['completed_ids']))!=800:
        raise ValueError('incomplete or duplicate capture IDs')
    rows=[json.loads(s) for s in inputs.read_text().splitlines()]
    if set(cm['completed_ids'])!={r['id'] for r in rows}:
        raise ValueError('capture input set mismatch')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'executed_internal_error_units_evaluate.py')
    split_gold={}
    for split in ['train','calibration']:
        p=inputs.parent/f'gold_{split}.json'
        if digest(p)!=im['split_gold_sha256'][split]:
            raise ValueError('split gold identity mismatch')
        split_gold.update(json.loads(p.read_text()))
    fs={arm:[] for arm in ARMS};slices={};events={};labels={};base_scores={};offset=0
    for i,row in enumerate(rows):
        rid=row['id'];p=root/(rid+'.npz')
        if digest(p)!=cm['output_sha256'][p.name]:
            raise ValueError('capture artifact mismatch')
        with np.load(p,allow_pickle=False) as a:
            f=features(a);n=len(a['controls'])
            events[rid]=a['event'].copy()
            base_scores[rid]=dict(nll=a['controls'][:,1].copy(),entropy=a['controls'][:,0].copy(),
                                  displacement=a['displacement'].copy(),negative_margin=a['controls'][:,2].copy(),
                                  source_mass=a['source_mass'].copy(),history_mass=a['history_mass'].copy())
            if a['full_token_ids'].tolist()!=row['token_ids'] or a['offsets'].tolist()!=row['offsets']:
                raise ValueError('feature-to-input alignment mismatch')
        for arm in ARMS:fs[arm].append(f[arm])
        slices[rid]=slice(offset,offset+n);offset+=n
        if rid in split_gold:labels[rid]=token_targets(row['offsets'],split_gold[rid])
        if i%64==0:print(json.dumps(dict(projected=i+1,total=len(rows))),flush=True)
    fs={k:np.concatenate(v) for k,v in fs.items()}
    np.savez_compressed(out/'projected_features.npz',**fs)
    def training_arrays(selected):
        idx=np.concatenate([np.arange(slices[r['id']].start,slices[r['id']].stop) for r in selected])
        y=np.concatenate([labels[r['id']]['core'] for r in selected])
        w=np.concatenate([np.full(len(labels[r['id']]['core']),1/len(labels[r['id']]['core'])) for r in selected])
        return idx,y,w*len(y)/w.sum()
    fits={};predictions={};thresholds={}
    families=sorted({r['family'] for r in rows if r['split']=='train'})
    for excluded in [None]+families:
        name='main' if excluded is None else 'loto-'+excluded
        train=[r for r in rows if r['split']=='train' and r['family']!=excluded]
        cal=[r for r in rows if r['split']=='calibration' and r['family']!=excluded]
        ti,ty,tw=training_arrays(train);ci,cy,cw=training_arrays(cal)
        fits[name]={};predictions[name]={};thresholds[name]={}
        for arm in ARMS:
            fit=fit_head(fs[arm][ti],ty,tw,fs[arm][ci],cy,cw)
            pp=predict(fit,fs[arm]);thr=fixed_fpr_threshold(cy,pp[ci])
            fit['achieved_calibration_negative_token_fpr']=float(np.mean(pp[ci][cy==0]>thr))
            fits[name][arm]=fit;thresholds[name][arm]=thr
            predictions[name][arm]={r['id']:pp[slices[r['id']]] for r in rows}
            print(json.dumps(dict(fit=name,arm=arm,iterations=fit['iterations'],warning=fit['convergence_warning'])),flush=True)
        write_json(out/'models.json',fits)
    # All test/natural predictions and fitted models freeze BEFORE test/natural label access.
    saved={f'{name}__{arm}':np.concatenate([predictions[name][arm][r['id']] for r in rows])
           for name in fits for arm in ARMS}
    np.savez_compressed(out/'predictions.npz',**saved)
    write_json(out/'prediction_index.json',dict(rows=[dict(id=r['id'],start=slices[r['id']].start,
        end=slices[r['id']].stop) for r in rows],thresholds=thresholds))
    write_json(out/'prediction_freeze.json',dict(complete=True,test_labels_read=False,natural_labels_read=False,
        predictions_sha256=digest(out/'predictions.npz'),models_sha256=digest(out/'models.json'),
        index_sha256=digest(out/'prediction_index.json'),capture_manifest_sha256=digest(root/'manifest.json'),
        code_sha256=digest(__file__)))
    gp=inputs.parent/'gold_test.json'
    if digest(gp)!=im['split_gold_sha256']['test']:raise ValueError('test gold mismatch')
    testgold=json.loads(gp.read_text());split_gold.update(testgold)
    for row in rows:
        if row['id'] in testgold:labels[row['id']]=token_targets(row['offsets'],testgold[row['id']])
    result=dict(primary='core equal-response log-loss and same-draft binding contrast',
        scope='constructed auxiliary supervision; observer natural transfer; no causal adoption proof',
        heldout={},loto={},binding_pairs={},baseline_raw={},paired_source_bootstrap={})
    test=[r for r in rows if r['split']=='test']
    for arm in ARMS:
        result['heldout'][arm]=summarize(test,predictions['main'][arm],labels,events,thresholds['main'][arm])
        result['heldout'][arm]['localization']=localization(test,predictions['main'][arm],labels,thresholds['main'][arm])
        result['binding_pairs'][arm]={}
        for width in [None,0,4,8]:
            key='all' if width is None else str(width)
            result['binding_pairs'][arm][key]=paired_binding(test,predictions['main'][arm],testgold,events,width)[0]
    result['binding_graph_contrasts']={}
    for width in [None,0,4,8]:
        key='all' if width is None else str(width)
        _,gd,ps=paired_binding(test,predictions['main']['graph'],testgold,events,width)
        sources=sorted(set(ps.values()));draws=np.random.default_rng(SEED).integers(len(sources),size=(500,len(sources)))
        result['binding_graph_contrasts'][key]={}
        for arm in ['baseline','nodes','shuffled_graph']:
            _,ad,_=paired_binding(test,predictions['main'][arm],testgold,events,width)
            per=np.array([np.mean([gd[k]-ad[k] for k in gd if ps[k]==sid]) for sid in sources])
            result['binding_graph_contrasts'][key][arm]=dict(point=float(per.mean()),ci95=np.quantile(per[draws].mean(1),[.025,.975]).tolist())
    for fam in families:
        rr=[r for r in test if r['family']==fam];name='loto-'+fam
        result['loto'][fam]={arm:summarize(rr,predictions[name][arm],labels,events,thresholds[name][arm]) for arm in ARMS}
    y=np.concatenate([labels[r['id']]['core'] for r in test])
    for key in ['nll','entropy','negative_margin','source_mass','history_mass','displacement']:
        result['baseline_raw'][key]=weighted_metrics(y,np.concatenate([base_scores[r['id']][key] for r in test]))
    sources=sorted({r['source_id'] for r in test})
    per_source={arm:np.array([np.mean([logloss(labels[r['id']]['core'],predictions['main'][arm][r['id']])
        for r in test if r['source_id']==sid]) for sid in sources]) for arm in ARMS}
    rng=np.random.default_rng(SEED)
    draws=rng.integers(len(sources),size=(500,len(sources)))
    for arm in ['baseline','nodes','shuffled_graph']:
        delta=per_source[arm]-per_source['graph']
        values=delta[draws].mean(1)
        result['paired_source_bootstrap']['graph_logloss_gain_vs_'+arm]=dict(
            point=float(delta.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),sources=len(sources))
    write_json(out/'controlled_evaluation.json',result)
    if args.annotations:
        from .confirmation_assess_scoped import selected_annotations
        from .development_assess_scoped import DEV_IDS
        natural=[r for r in rows if r['split']=='natural']
        wanted={r['id'].removeprefix('natural-') for r in natural}
        if wanted!=DEV_IDS:raise ValueError('natural scope changed')
        gt=selected_annotations(args.annotations,wanted)
        for r in natural:
            g=gt[r['id'].removeprefix('natural-')]
            if hashlib.sha256(g['response'].encode()).hexdigest()!=r['response_sha256']:
                raise ValueError('natural annotation response identity')
            if str(g['source_id'])!=r['source_id'].removeprefix('natural-'):
                raise ValueError('natural annotation source identity')
            y,o,_=annotation_targets(r['offsets'],g['labels'])
            labels[r['id']]=dict(core=y,onset=o,scope=y.copy())
        # scope is only populated for common summary plumbing; never exposed as semantic-unit truth.
        nat={arm:summarize(natural,predictions['main'][arm],labels,events,thresholds['main'][arm]) for arm in ARMS}
        for v in nat.values():v.pop('disjoint_region_scores')
        yy=np.concatenate([labels[r['id']]['core'] for r in natural])
        nat['raw_baselines']={key:weighted_metrics(yy,np.concatenate([base_scores[r['id']][key] for r in natural])) for key in ['nll','entropy','negative_margin','source_mass','history_mass','displacement']}
        nat['limitations']='core head transferred to annotated error tokens; entity-core/semantic-unit gold unavailable; known development observer data'
        nat['annotation_sha256']=digest(args.annotations)
        write_json(out/'natural_evaluation.json',nat)
    write_json(out/'complete.json',dict(complete=bool(args.annotations),controlled_complete=True,natural_complete=bool(args.annotations),seconds=time.monotonic()-started,
        model_forwards=0,fit_models=len(fits)*4,
        artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(complete=bool(args.annotations),seconds=time.monotonic()-started,heldout=result['heldout']['graph']['equal_response'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True);p.add_argument('--capture',required=True)
    p.add_argument('--output',required=True);p.add_argument('--annotations')
    run(p.parse_args())
