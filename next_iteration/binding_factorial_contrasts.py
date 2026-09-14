"""Predetermined B8 synthetic contrasts. Not a natural detection evaluation."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def summarize(rows, results):
    if len(rows)!=160 or len(results)!=160 or len({r['id'] for r in rows})!=160:
        raise ValueError('complete160 required')
    if len({r['id'] for r in results})!=160 or {r['id'] for r in rows}!={r['id'] for r in results}:
        raise ValueError('exact160 prediction IDs required')
    index={r['id']:r for r in results}
    cells={}
    for row in rows:
        result=index[row['id']]
        ab=result['logits']
        if (len(ab)!=2 or not all(math.isfinite(v) for v in ab)
            or not math.isfinite(result['risk']) or result['risk']!=ab[1]-ab[0]):
            raise ValueError('raw B-A risk or finite integrity')
        key=(row['family'],row['source_assignment'],row['source_order'],row['draft_assignment'],row['draft_order'])
        if key in cells:raise ValueError('duplicate factorial cell')
        cells[key]=result['risk']
    families=sorted({r['family'] for r in rows})
    if len(families)!=8:raise ValueError('eight families required')
    contrasts=[]
    for family in families:
        for source_order in (0,1):
            for draft_assignment,draft_order in [(None,None),(0,0),(0,1),(1,0),(1,1)]:
                a=cells[(family,0,source_order,draft_assignment,draft_order)]
                b=cells[(family,1,source_order,draft_assignment,draft_order)]
                contrasts.append(dict(family=family,type='source_assignment',source_order=source_order,
                    draft_assignment=draft_assignment,draft_order=draft_order,delta=b-a,
                    explanation='risk(unsupported source)-risk(supported source) with other factors fixed'))
        for source_assignment in (0,1):
            for source_order in (0,1):
                for draft_order in (0,1):
                    a=cells[(family,source_assignment,source_order,0,draft_order)]
                    b=cells[(family,source_assignment,source_order,1,draft_order)]
                    contrasts.append(dict(family=family,type='draft_assignment',source_assignment=source_assignment,
                        source_order=source_order,draft_order=draft_order,delta=b-a,
                        explanation='risk(contradicting-answer draft)-risk(supporting-answer draft), source fixed'))
            for draft_assignment,draft_order in [(None,None),(0,0),(0,1),(1,0),(1,1)]:
                a=cells[(family,source_assignment,0,draft_assignment,draft_order)]
                b=cells[(family,source_assignment,1,draft_assignment,draft_order)]
                contrasts.append(dict(family=family,type='source_order',source_assignment=source_assignment,
                    draft_assignment=draft_assignment,draft_order=draft_order,delta=b-a))
            for source_order in (0,1):
                for draft_assignment in (0,1):
                    a=cells[(family,source_assignment,source_order,draft_assignment,0)]
                    b=cells[(family,source_assignment,source_order,draft_assignment,1)]
                    contrasts.append(dict(family=family,type='draft_order',source_assignment=source_assignment,
                        source_order=source_order,draft_assignment=draft_assignment,delta=b-a))
        for source_order in (0,1):
            for draft_order in (0,1):
                effect0=cells[(family,1,source_order,0,draft_order)]-cells[(family,0,source_order,0,draft_order)]
                effect1=cells[(family,1,source_order,1,draft_order)]-cells[(family,0,source_order,1,draft_order)]
                contrasts.append(dict(family=family,type='source_draft_interaction',source_order=source_order,
                    draft_order=draft_order,delta=effect1-effect0))
    accuracy=[]
    flips=[]
    for family in families:
        for mode in ('no_draft','faithful_draft','corrupted_draft'):
            selected=[r for r in rows if r['family']==family and
                      ('no_draft' if r['draft'] is None else 'faithful_draft' if r['draft_agrees_with_source'] else 'corrupted_draft')==mode]
            correct=sum(('B' if index[r['id']]['risk']>0 else 'A')==r['constructed_expected'] for r in selected)
            ties=sum(index[r['id']]['risk']==0 for r in selected)
            accuracy.append(dict(family=family,mode=mode,correct=correct,total=len(selected),accuracy=correct/len(selected),ties=ties))
        for row in [r for r in rows if r['family']==family and r['draft'] is not None]:
            base=cells[(family,row['source_assignment'],row['source_order'],None,None)]
            risk=index[row['id']]['risk']
            truth=row['constructed_expected']
            before=('B' if base>0 else 'A')==truth
            after=('B' if risk>0 else 'A')==truth
            flips.append(dict(id=row['id'],family=family,faithful=row['draft_agrees_with_source'],
                no_draft_correct=before,draft_correct=after,corrected=not before and after,
                induced_error=before and not after,raw_delta_from_no_draft=risk-base,
                caveat='No-draft comparison changes context length; not a semantic-only causal contrast.'))
    if any(not math.isfinite(c['delta']) for c in contrasts):
        raise ValueError('non-finite derived contrast')
    if any(not math.isfinite(f['raw_delta_from_no_draft']) for f in flips):
        raise ValueError('non-finite no-draft difference')
    return dict(scope='synthetic_by_construction; not RAGTruth AUROC or original-generator mechanism',
        total=160,families=8,threshold=0,tie_rule='A on exact tie, disclosed; no threshold fitting',
        contrasts=contrasts,accuracy=accuracy,flips=flips,
        aggregation='All family/order/cell effects reported; no selected best subset, statistical independence or generalization claim.')

def validate_prediction_manifest(m,rows,queries,code_sha,executed_sha,input_shas,model_metadata,weights,dependency_sha):
    ids=[r['id'] for r in rows]
    if (len(ids)!=160 or len(set(ids))!=160 or m.get('complete') is not True
        or m.get('error') is not None or m.get('planned_ids')!=ids or m.get('completed_ids')!=ids
        or [r.get('id') for r in m.get('results',[])]!=ids):
        raise ValueError('exact complete ordered unique160 required')
    if not code_sha or m.get('code_sha256')!=code_sha or executed_sha!=code_sha:
        raise ValueError('audited scorer/executed snapshot identity')
    if (m.get('input_sha256')!=input_shas or m.get('preflight_sha256')!=input_shas['preflight.json']
        or m.get('label_ids')!=[32,33] or m.get('model_metadata_sha256')!=model_metadata
        or m.get('model_shard_sha256')!=weights or m.get('dependency_sha256')!=dependency_sha
        or m.get('forwards')!=168):
        raise ValueError('input/model/dependency/forward identity')
    if [q['id'] for q in queries]!=ids:raise ValueError('query order')
    for result,query in zip(m['results'],queries,strict=True):
        token_sha=hashlib.sha256(json.dumps(query['input_ids']).encode()).hexdigest()
        if result.get('input_ids_sha256')!=token_sha or result.get('input_tokens')!=len(query['input_ids']):
            raise ValueError('exact input token stream binding')
    checks=m.get('checks',[])
    if [c.get('id') for c in checks]!=ids[::20]:raise ValueError('exact eight repeat checks')
    for c,index in zip(checks,range(0,160,20),strict=True):
        if c.get('ab')!=m['results'][index]['logits'] or len(c.get('repeat',[]))!=2:
            raise ValueError('repeat raw logits identity')
        if not all(math.isfinite(v) for v in c['ab']+c['repeat']):raise ValueError('repeat finite values')
        delta=max(abs(a-b) for a,b in zip(c['ab'],c['repeat']))
        odds=abs((c['ab'][1]-c['ab'][0])-(c['repeat'][1]-c['repeat'][0]))
        if (not math.isfinite(delta) or not math.isfinite(odds) or max(delta,odds)>0.005
            or c.get('max_ab_delta')!=delta or c.get('odds_delta')!=odds):
            raise ValueError('repeat numerical guard')

def validate_launch(launch,manifest_sha,scorer_sha,contrast_sha,output,p7_identity,weights):
    expected=dict(status='completed',attempt=1,actual_subprocess_returncode=0,
        prediction_manifest_sha256=manifest_sha,scorer_sha256=scorer_sha,
        contrast_code_sha256=contrast_sha,output=output,p7_completion=p7_identity,
        model_shard_sha256=weights)
    if any(launch.get(k)!=v for k,v in expected.items()):
        raise ValueError('independent actual-exit0 launch/manifest/code/model/P7 binding')

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True)
    p.add_argument('--predictions',required=True)
    p.add_argument('--launch-record',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    pred=Path(args.predictions)
    output=Path(args.output)
    sidecar=Path(args.output+'.executed_code.py')
    if output.exists() or sidecar.exists():raise FileExistsError('preserve primary/sidecar before rerun')
    from .binding_factorial_score import (load_inputs,EXPECTED,WEIGHT_SHA,DEPENDENCY_SHA,
        ROOT,INPUTS,sha,p7_completion_identity)
    rows,queries,preflight=load_inputs()
    if Path(args.inputs).resolve()!=(INPUTS/'inputs.jsonl').resolve():
        raise ValueError('only exact frozen inputs path accepted')
    m=json.loads((pred/'manifest.json').read_text())
    scorer_sha=sha(ROOT/'next_iteration/binding_factorial_score.py')
    validate_prediction_manifest(m,rows,queries,scorer_sha,sha(pred/'executed_code.py'),
        EXPECTED,preflight['tokenizer_metadata_sha256'],WEIGHT_SHA,DEPENDENCY_SHA)
    launch=json.loads(Path(args.launch_record).read_text())
    p7_identity=p7_completion_identity()
    if m.get('p7_completion')!=p7_identity:raise ValueError('P7 prerequisite changed')
    validate_launch(launch,sha(pred/'manifest.json'),scorer_sha,sha(__file__),str(pred.resolve()),p7_identity,WEIGHT_SHA)
    if sha(Path(launch['log']))!=launch.get('log_sha256'):raise ValueError('actual launch log binding')
    result=summarize(rows,m['results'])
    result['prediction_manifest_sha256']=sha(pred/'manifest.json')
    result['executed_code_sha256']=sha(__file__)
    result['actual_launch_record_sha256']=sha(args.launch_record)
    with sidecar.open('xb') as f:f.write(Path(__file__).read_bytes())
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(complete=True,rows=160,contrasts=len(result['contrasts']),output=args.output)))


if __name__=='__main__':main()
