"""P3 event coverage and localization audit. Reads annotations only after freeze."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cover(events, lead):
    event=np.asarray(events,bool)
    out=event.copy()
    for lag in range(1,min(lead+1,len(event))):
        out[lag:] |= event[:-lag]
    return out


def random_events(events,rng,mode):
    event=np.asarray(events,bool)
    result=np.zeros(len(event),bool)
    if mode=='uniform':
        pool=np.arange(17,len(event))
        result[rng.choice(pool,size=int(event.sum()),replace=False)]=True
    elif mode=='position_matched':
        for start in range(0,len(event),32):
            stop=min(start+32,len(event))
            count=int(event[start:stop].sum())
            pool=np.arange(max(17,start),stop)
            result[rng.choice(pool,size=count,replace=False)]=True
    else:
        raise ValueError('unknown random control')
    return result


def selector_counts(events,onset,y,lead):
    mask=cover(events,lead)
    event_hits=sum(bool(np.any(onset[t:min(t+lead+1,len(onset))])) for t in np.flatnonzero(events))
    return dict(tokens=len(y),events=int(np.sum(events)),onsets=int(np.sum(onset)),
        eligible_onsets=int(np.sum(onset[17:])),onset_hits=int(np.sum(mask*onset)),
        covered_tokens=int(mask.sum()),covered_error_tokens=int(np.sum(mask*y)),
        event_windows_with_onset=int(event_hits))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--annotations',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    from next_iteration.grounding_contrast_evaluate import annotation_targets, weighted_metrics
    root=Path(args.predictions)
    manifest=json.loads((root/'manifest.json').read_text())
    if not manifest['complete'] or manifest.get('error') or manifest['settings']['phase']!='development':
        raise ValueError('requires completed development predictions')
    if manifest['completed_ids']!=manifest['planned_ids']:
        raise ValueError('incomplete roster')
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    records=[]
    for rid in manifest['planned_ids']:
        path=root/f'response_{rid}.json'
        if digest(path)!=manifest['output_sha256'][path.name]:
            raise ValueError('score hash mismatch')
        r=json.loads(path.read_text())
        if str(r['id'])!=rid:
            raise ValueError('ID mismatch')
        graph=root/f'attention_{rid}.npz'
        if digest(graph)!=manifest['graph_sha256'][graph.name] or digest(graph)!=r['graph_sha256']:
            raise ValueError('graph hash mismatch')
        records.append(r)
    # Scoring is already complete and hashed. Only now load independent human GT.
    annotations={}
    for line in Path(args.annotations).open():
        row=json.loads(line)
        rid=str(row['id'])
        if rid in manifest['planned_ids']:
            if rid in annotations:raise ValueError('duplicate annotations')
            annotations[rid]=row
    if set(annotations)!=set(manifest['planned_ids']):raise ValueError('missing annotations')
    rng=np.random.default_rng(20260914)
    rows=[]
    for r in records:
        annotation=annotations[str(r['id'])]
        if hashlib.sha256(annotation['response'].encode()).hexdigest()!=r['response_sha256'] or str(annotation['source_id'])!=str(r['source_id']):
            raise ValueError('GT identity mismatch')
        y,onset,_=annotation_targets(r['offsets'],annotation['labels'])
        n=len(y)
        scores={k:np.array(v) for k,v in r['scores'].items()}
        if any(v.shape!=(n,) or not np.isfinite(v).all() for v in scores.values()):
            raise ValueError('nonfinite or misaligned score')
        revisit=np.array(r['event'],bool)
        entropy_top=np.zeros(n,bool)
        eligible=np.arange(17,n)
        k=int(revisit.sum())
        entropy_top[eligible[np.argsort(-scores['native_entropy'][eligible],kind='stable')[:k]]]=True
        selectors=dict(revisit=revisit,mean_revisit=np.array(r['mean_event'],bool),
            entropy_causal=np.array(r['entropy_event'],bool),entropy_topk_retrospective=entropy_top)
        counts={name:{str(lead):selector_counts(event,onset,y,lead) for lead in (0,4,8)} for name,event in selectors.items()}
        random_counts={}
        for mode in ('uniform','position_matched'):
            draws=[random_events(revisit,rng,mode) for _ in range(500)]
            random_counts[mode]={str(lead):dict(
                onset_hits_mean=float(np.mean([np.sum(cover(e,lead)*onset) for e in draws])),
                covered_tokens_mean=float(np.mean([np.sum(cover(e,lead)) for e in draws]))) for lead in (0,4,8)}
        # All candidates in actual revisit windows; no GT feeds event construction.
        matched={}
        for lead in (0,4,8):
            wrong,normal=[],[]
            for t in np.flatnonzero(revisit):
                stop=min(n,t+lead+1)
                if onset[t:stop].any():wrong.append(int(t))
                elif not y[t:stop].any():normal.append(int(t))
            matched[str(lead)]=dict(onset_event_positions=wrong,normal_event_positions=normal,
                scores={name:dict(onset=[float(scores[name][t]) for t in wrong],
                    normal=[float(scores[name][t]) for t in normal]) for name in ('native_entropy','revisit_magnitude')})
        rows.append(dict(id=str(r['id']),source_id=str(r['source_id']),task=r['task'],
            y=y,onset=onset,scores=scores,counts=counts,random=random_counts,matched=matched,
            onset_cases=[dict(token_index=int(t),
                text=annotation['response'][r['offsets'][t][0]:r['offsets'][t][1]],
                visible_prefix=annotation['response'][max(0,r['offsets'][t][0]-100):r['offsets'][t][1]],
                previous_revisit=int(np.flatnonzero(revisit[:t+1])[-1]) if revisit[:t+1].any() else None,
                revisit_here=bool(revisit[t]),entropy_here=float(scores['native_entropy'][t]))
                for t in np.flatnonzero(onset)]))
    sources=sorted(set(r['source_id'] for r in rows))
    source_index={s:i for i,s in enumerate(sources)}
    multiplicities=np.array([np.bincount(rng.integers(0,len(sources),len(sources)),minlength=len(sources)) for _ in range(500)])
    coverage={}
    for selector in rows[0]['counts']:
        coverage[selector]={}
        for lead in ('0','4','8'):
            total={key:sum(r['counts'][selector][lead][key] for r in rows) for key in rows[0]['counts'][selector][lead]}
            total.update(onset_recall=total['onset_hits']/max(1,total['onsets']),
                token_coverage_fraction=total['covered_tokens']/total['tokens'],
                onset_event_precision=total['event_windows_with_onset']/max(1,total['events']))
            if selector=='revisit':
                total['random_controls']={}
                for mode in ('uniform','position_matched'):
                    expected=sum(r['random'][mode][lead]['onset_hits_mean'] for r in rows)
                    per_source=np.zeros((len(sources),2))
                    for r in rows:
                        i=source_index[r['source_id']]
                        per_source[i,0]+=r['counts']['revisit'][lead]['onset_hits']-r['random'][mode][lead]['onset_hits_mean']
                        per_source[i,1]+=r['onset'].sum()
                    sampled=multiplicities@per_source
                    ratios=sampled[sampled[:,1]>0,0]/sampled[sampled[:,1]>0,1]
                    total['random_controls'][mode]=dict(expected_hits=expected,
                        expected_token_coverage=sum(r['random'][mode][lead]['covered_tokens_mean'] for r in rows)/total['tokens'],
                        recall_excess=(total['onset_hits']-expected)/max(1,total['onsets']),
                        source_bootstrap_95_ci=np.quantile(ratios,[.025,.975]).tolist())
            coverage[selector][lead]=total
    within={}
    for name in rows[0]['scores']:
        vals=[]
        differences=np.zeros((len(sources),2))
        for r in rows:
            y=r['y'];pairs=int(y.sum()*(len(y)-y.sum()))
            if pairs:
                auc=weighted_metrics(y,r['scores'][name])['auroc']
                base=weighted_metrics(y,r['scores']['native_entropy'])['auroc']
                vals.append((auc,pairs))
                differences[source_index[r['source_id']]] += [(auc-base)*pairs,pairs]
        boot=multiplicities@differences
        valid=boot[:,1]>0
        delta=boot[valid,0]/boot[valid,1]
        within[name]=dict(mixed_responses=len(vals),pair_weighted_auroc=sum(a*w for a,w in vals)/sum(w for _,w in vals),
            macro_auroc=float(np.mean([a for a,_ in vals])),
            delta_to_entropy_source_bootstrap_95_ci=np.quantile(delta,[.025,.975]).tolist())
    conditional={}
    for lead in ('0','4','8'):
        conditional[lead]={}
        for name in ('native_entropy','revisit_magnitude'):
            yy,ss,pairs=[],[],[]
            for r in rows:
                m=r['matched'][lead]['scores'][name]
                y=np.r_[np.ones(len(m['onset'])),np.zeros(len(m['normal']))]
                s=np.array(m['onset']+m['normal'])
                yy.extend(y);ss.extend(s)
                count=len(m['onset'])*len(m['normal'])
                if count:pairs.append((weighted_metrics(y,s)['auroc'],count))
            conditional[lead][name]=dict(onset_windows=int(sum(yy)),normal_windows=int(len(yy)-sum(yy)),
                metrics=weighted_metrics(yy,ss),within_response_auroc=sum(a*w for a,w in pairs)/sum(w for _,w in pairs) if pairs else None)
    report=dict(scope='development diagnostic; window association is not causal mechanism or relation applicability',
        annotation_sha256=digest(args.annotations),manifest_sha256=digest(root/'manifest.json'),code_sha256=digest(__file__),
        responses=len(rows),sources=len(sources),tokens=sum(len(r['y']) for r in rows),
        positive_tokens=int(sum(r['y'].sum() for r in rows)),onsets=int(sum(r['onset'].sum() for r in rows)),
        coverage=coverage,within_response=within,conditional_event_comparison=conditional,
        per_response=[{k:v for k,v in r.items() if k not in ('y','onset','scores')} for r in rows],
        uncertainty='500 source-cluster bootstrap; random expectations from 500 draws; seed20260914; no multiple-comparison claim')
    serialized=json.dumps(report,indent=2,allow_nan=False)
    with Path(args.output).open('x') as f:
        f.write(serialized+'\n')
    Path(str(args.output)+'.executed_code.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k!='per_response'}))


if __name__=='__main__':
    main()
