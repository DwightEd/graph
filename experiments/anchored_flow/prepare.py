"""Assemble cached RAG cases and source-disjoint GSM scale/calibration answers."""
import argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from experiments.decision_risk_flow.data import inputs, read_json, write_json
from experiments.constraint_uptake.data import MODEL
from experiments.native_support.evidence_contrast.views import unit_intervals

PREVIOUS = Path('outputs/context_response_20260928_v4')
UPTAKE = Path('outputs/constraint_uptake_20260929_dense')
NATURAL = {'00005', '00006', '00012', '00013'}


def rag_records(tokenizer):
    manifest = read_json(PREVIOUS/'manifest.json')
    rows = manifest['records']
    root = Path(next(r['root'] for r in rows if r['kind']=='observer'))
    lookup = {str(r['source_id']):r['source_file'] for r in read_json(root/'manifest.json')['records']}
    result = []
    for row in rows:
        if row['role']!='regression' and row['key'] not in NATURAL:
            continue
        source = read_json(root/lookup[row['source_id']])
        if row['kind']=='observer':
            prompt, response = inputs(row)
            cached = str(Path(row['root'])/row['directory'])
        else:
            with np.load(Path(manifest['samples'])/row['trace']) as saved:
                first = int(saved['prompt_length'])
                prompt = saved['token_ids'][:first].tolist()
                answer = saved['token_ids'][first:].tolist()
            text = [tokenizer.decode([i]) for i in answer]
            response = dict(answer_ids=answer, token_text=text, special_ids=tokenizer.all_special_ids,
                units=unit_intervals(dict(token_text=text, prompt_length=0),64))
            cached = None
        assert prompt==source['prompt_with_source']
        result.append(dict(key=row['key'],dataset='ragtruth',task=row['task'],role='case',
            pair=row['source_id'],original=row,prompt=prompt,source=source,response=response,cached=cached))
    return result


def gsm_record(row, tokenizer, role):
    with np.load(row['cache']) as saved:
        first = int(saved['response_idx'])
        prompt = saved['token_ids'][:first].tolist()
        answer = saved['token_ids'][first:].tolist()
        ranges = (saved['step_ranges']-first).tolist()
    # Raw GSM prompt: retain special template tokens, remove question text only.
    mask = [i not in tokenizer.all_special_ids for i in prompt]
    absent = [i for i, removed in zip(prompt,mask) if not removed]
    assert absent, 'GSM source deletion must retain its original BOS/template'
    units = [dict(start=a,stop=b) for a,b in ranges]
    covered = np.zeros(len(answer),dtype=bool)
    for a,b in ranges:
        covered[a:b] = True
    # Separators outside annotated ranges are still observed, not silently dropped.
    edges = np.r_[0,np.flatnonzero(np.diff(covered.astype(int)))+1,len(answer)]
    units += [dict(start=int(a),stop=int(b)) for a,b in zip(edges[:-1],edges[1:]) if not covered[a]]
    units.sort(key=lambda u:u['start'])
    response = dict(answer_ids=answer,token_text=[tokenizer.decode([i]) for i in answer],
        units=units,special_ids=tokenizer.all_special_ids)
    return dict(key=row['id'],dataset='gsm8k',task='GSM8K',role=role,pair=row['problem'],
        original=row,prompt=prompt,response=response,cached=None,step_ranges=ranges,
        source=dict(prompt_with_source=prompt,prompt_without_source=absent,source_mask=mask))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    records = rag_records(tokenizer)
    cases = {r['key'] for r in read_json(UPTAKE/'manifest.json')['records'] if r['dataset']=='gsm8k'}
    gsm = read_json('outputs/gsm8k_states_20260929_v1/manifest.json')['records']
    excluded = {r['problem'] for r in gsm if r['id'] in cases}
    for role in ('fit','dev'):
        selected = set()
        for row in sorted(gsm,key=lambda r:r['id']):
            if row['role']==role and row['problem'] not in excluded|selected:
                records.append(gsm_record(row,tokenizer,role))
                selected.add(row['problem'])
                if len(selected)==8:
                    break
    records += [gsm_record(r,tokenizer,'case') for r in gsm if r['id'] in cases]
    write_json(args.output/'manifest.json',dict(model=MODEL,records=records,labels_used=False,
        scope='exposed small-sample development; original full-token input; no prompt entity annotations'))
    print([(r['key'],r['role'],len(r['response']['answer_ids'])) for r in records],flush=True)


if __name__=='__main__':
    main()
