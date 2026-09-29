"""Prepare exposed paired cases without annotating evidence inside prompts."""
import argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from experiments.decision_risk_flow.data import inputs, read_json, write_json
from experiments.route_complement.capture import get_inputs
from experiments.role_free_flow.diagnostics import read_local_spans

MODEL = '/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct'
RAG = Path('outputs/context_response_20260928_v1/manifest.json')
GSM = Path('outputs/gsm8k_states_20260929_v1/manifest.json')


def source_blocks(token_text):
    blocks = []
    start = 0
    for index,text in enumerate(token_text):
        if '\n' in text or any(mark in text for mark in ('.','?','!',';')):
            blocks.append([start,index+1])
            start = index+1
    if start<len(token_text):
        blocks.append([start,len(token_text)])
    return blocks


def rag_records(tokenizer):
    manifest = read_json(RAG)
    records = manifest['records']
    root = Path(records[0]['root'])
    lookup = {str(row['source_id']):row['source_file'] for row in read_json(root/'manifest.json')['records']}
    local = read_local_spans(Path('outputs/role_free_flow_20260928'),Path('experiments/path_conflict/paired_cases.json'))
    selected = []
    for row in records:
        if row['key'] not in ('00005','00006','00012','00013','15604','9022'):
            continue
        prompt,answer,_ = get_inputs(row,Path(manifest['samples']),root,lookup)
        if row['kind']=='observer':
            positions = list(range(len(answer)))
            response = inputs(row)[1]
            text = response['token_text']
        else:
            claim = local[local.trace==row['trace']].position.to_numpy()
            positions = np.arange(max(0,int(claim.min())-6),min(len(answer),int(claim.max())+7)).tolist()
            text = [tokenizer.decode([token]) for token in answer]
        selected.append(dict(key=row['key'],dataset='ragtruth',pair=str(row['source_id']),
            prompt=prompt,answer=answer,positions=positions,text=text,original=row,
            blocks=source_blocks([tokenizer.decode([token]) for token in prompt])))
    return selected


def gsm_records(tokenizer):
    manifest = read_json(GSM)
    selected = []
    keys = ('gsm8k-49','gsm8k-243','gsm8k-43','gsm8k-240','gsm8k-123','gsm8k-313')
    for row in manifest['records']:
        if row['id'] not in keys:
            continue
        with np.load(row['cache']) as saved:
            first = int(saved['response_idx'])
            tokens = saved['token_ids'].tolist()
            ranges = saved['step_ranges']-first
        positions = sorted({int(p) for start,end in ranges for p in np.linspace(start,end-1,min(8,end-start)).astype(int)})
        prompt,answer = tokens[:first],tokens[first:]
        selected.append(dict(key=row['id'],dataset='gsm8k',pair=row['problem'],
            prompt=prompt,answer=answer,positions=positions,step_ranges=ranges.tolist(),
            text=[tokenizer.decode([token]) for token in answer],original=row,
            blocks=source_blocks([tokenizer.decode([token]) for token in prompt])))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    records = rag_records(tokenizer)+gsm_records(tokenizer)
    write_json(args.output/'manifest.json',dict(model=MODEL,records=records,
        roster='exposed paired mechanisms and known failures; not independent evaluation',
        prompt_annotations=False,candidates=3,query_batch=8,
        scope='fixed-past observer; local labels only for case selection and post-freeze evaluation'))
    print([(row['key'],len(row['positions'])) for row in records],flush=True)


if __name__=='__main__':
    main()
