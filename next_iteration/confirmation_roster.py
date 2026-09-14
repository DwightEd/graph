"""P6 new official-test source holdout; no annotation input, quality or score selection."""
import argparse
import hashlib
import json
from pathlib import Path
from .grounding_contrast import INPUT_SHA, digest, validate_rows, write_json

POPULATION_SHA='be388df51e83f60beafbcb4075bde85da11a40f0941279e715f0f334e28d67bb'
QUOTAS={'QA':12,'Summary':12,'Data2txt':8}
GENERATORS=['llama-2-7b-chat','llama-2-13b-chat']


def text_sha(s):return hashlib.sha256(s.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--r04',required=True)
    p.add_argument('--population-inputs',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    assert digest(args.r04)==INPUT_SHA and digest(args.population_inputs)==POPULATION_SHA
    old=list(map(json.loads,Path(args.r04).read_text().splitlines()))
    validate_rows(old)
    excluded_ids={str(r['source_id']) for r in old}
    excluded_text={text_sha(r['prompt'][r['source_span'][0]:r['source_span'][1]]) for r in old}
    grouped={}
    for line in Path(args.population_inputs).open():
        r=json.loads(line)
        if r['official_split']!='test' or r['generator'] not in GENERATORS:continue
        sid=str(r['source_id']);lo,hi=r['source_span'];s_sha=text_sha(r['prompt'][lo:hi])
        if sid in excluded_ids or s_sha in excluded_text:continue
        if sid not in grouped:
            grouped[sid]={'source_id':sid,'source_sha256':s_sha,'task':r['task'],
                          'select_sha256':text_sha(f'P6-confirm-v1|20260914|{sid}|{s_sha}'),'responses':{}}
        g=grouped[sid]
        if g['source_sha256']!=s_sha or g['task']!=r['task']:raise ValueError('inconsistent source')
        if r['generator'] in g['responses']:raise ValueError('duplicate response')
        g['responses'][r['generator']]=r
    selected=[];seen_text=set()
    for task,quota in QUOTAS.items():
        pool=sorted([g for g in grouped.values() if g['task']==task],key=lambda g:g['select_sha256'])
        taken=0
        for g in pool:
            if g['source_sha256'] in seen_text:continue
            selected.append(g);seen_text.add(g['source_sha256']);taken+=1
            if taken==quota:break
        if taken!=quota:raise ValueError('insufficient source candidates')
    rows=[];missing=[]
    for g in selected:
        for generator in GENERATORS:
            if generator not in g['responses']:
                missing.append({'source_id':g['source_id'],'generator':generator});continue
            r=dict(g['responses'][generator]);r['split']='validation'
            r['prompt_sha256']=text_sha(r['prompt']);r['source_text_sha256']=g['source_sha256']
            r['observer']=old[0]['observer'];r['original_generator_causality']=old[0]['original_generator_causality']
            r['prediction_query_positions']=list(range(r['prompt_length']-1,len(r['token_ids'])-1))
            if len(r['prediction_query_positions'])!=len(r['offsets']):raise ValueError('token alignment')
            if text_sha(r['response'])!=r['response_sha256']:raise ValueError('response hash')
            rows.append(r)
    validate_rows(rows)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with (out/'inputs.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    roster=[{k:v for k,v in g.items() if k!='responses'} | {'response_ids':{k:str(v['id']) for k,v in g['responses'].items()}} for g in selected]
    write_json(out/'roster.json',roster)
    (out/'executed_code.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(status='complete',settings=vars(args),protocol='P6-confirm-v1|20260914; SHA order; source+exact evidence dedup',
        code_sha256=digest(__file__),r04_inputs_sha256=INPUT_SHA,population_inputs_sha256=POPULATION_SHA,
        input_sha256=digest(out/'inputs.jsonl'),roster_sha256=digest(out/'roster.json'),
        planned_sources=32,planned_responses=64,sources=len(selected),responses=len(rows),missing=missing,
        by_task=QUOTAS,excluded_r04_source_ids=sorted(excluded_ids),labels_joined=False,model_forwards=0,
        selection_does_not_use=['labels','quality','scores','response length','confidence','resource fit'],
        scope='official-test source holdout relative to R04; previous population measurements touched these sources; not globally unseen',
        confirmation_policy='all64 responses as single primary denominator; no partial-cohort peeking or sample replacement')
    write_json(out/'manifest.json',manifest)
    print(json.dumps(manifest))


if __name__=='__main__':main()
