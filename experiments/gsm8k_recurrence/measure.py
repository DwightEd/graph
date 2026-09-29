"""Read full-key physical heads and validate the cached step/token alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'demo/data/hf_datasets/ProcessBench/gsm8k.json'
CACHE = ROOT/'demo/outputs/attention_traces/gsm8k_llama31_layer14/balanced'
MODEL = ROOT.parent/'models/Meta-Llama-3.1-8B-Instruct'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def partition(records):
    problems = sorted({row['problem'] for row in records},
                      key=lambda value:hashlib.sha256(value.encode()).hexdigest())
    fit_end = len(problems)//5
    dev_end = 2*fit_end
    return {problem:('fit' if i<fit_end else 'dev' if i<dev_end else 'evaluation')
            for i,problem in enumerate(problems)}


def step_positions(saved, record, tokenizer):
    prompt, response = str(saved['rendered_prompt']), str(saved['response_text'])
    assert response=='\n\n'.join(record['steps'])
    encoded = tokenizer(prompt+response, return_offsets_mapping=True)
    assert np.array_equal(encoded['input_ids'], saved['token_ids']), record['id']
    ranges = []
    offset = len(prompt)
    for step in record['steps']:
        end = offset+len(step)
        selected = [i for i,(start,stop) in enumerate(encoded['offset_mapping'])
                    if start<end and stop>offset]
        ranges.append((min(selected), max(selected)+1))
        offset = end+2
    assert np.array_equal(ranges, saved['step_ranges']), record['id']
    first = int(saved['response_idx'])
    count = len(saved['token_ids'])-first
    step_id = np.full(count, -1, dtype=int)
    for step,(start,stop) in enumerate(ranges):
        step_id[start-first:stop-first] = step
    return np.array(ranges)-first, step_id


def measure_heads(attention, first):
    # Output y_t is predicted by query first-1+t, before y_t has been input.
    values = attention[0,:,first-1:-1,:].astype(np.float32)
    queries = np.arange(first-1, attention.shape[-1]-1)
    mass = values.sum(-1)
    assert np.isfinite(values).all() and (values>=0).all()
    assert np.max(abs(mass-1))<.02
    future = np.arange(attention.shape[-1])[None,:]>queries[:,None]
    assert not np.any(values*future[None,:,:])
    values /= mass[:,:,None]
    source = values[:,:,:first].sum(-1)
    raw = -np.log(np.maximum(source, 1e-8)).mean(0)
    roots = np.sqrt(values)
    rng = np.random.default_rng(713)
    shuffled = np.stack([roots[rng.permutation(32),t] for t in range(len(queries))], axis=1)
    pairs = {}
    for lag in range(1,9):
        pairs[f'head_{lag}'] = np.einsum('htk,htk->ht', roots[:,lag:],roots[:,:-lag])
        pairs[f'shuffled_{lag}'] = np.einsum('htk,htk->t', shuffled[:,lag:],shuffled[:,:-lag])/32
    return raw, pairs, source, float(np.max(abs(mass-1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    records = json.loads(DATA.read_text())
    split = partition(records)
    manifest = json.loads((CACHE/'manifest.json').read_text())
    traces = {row['sample_id']:row for row in manifest['traces']}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    roster, errors = [], []
    for index,record in enumerate(records):
        trace = traces[record['id']]
        with np.load(CACHE/trace['file']) as saved:
            assert saved['attention_layers'].tolist()==[14]
            assert saved['attention_heads'].tolist()==list(range(32))
            ranges, steps = step_positions(saved, record, tokenizer)
            first = int(saved['response_idx'])
            raw,pairs,source,error = measure_heads(saved['attention'],first)
            np.savez_compressed(args.output/(record['id']+'.npz'), raw=raw,
                source_heads=source, step_ranges=ranges, step_id=steps,
                response_ids=saved['token_ids'][first:], **pairs)
            roster.append(dict(id=record['id'], problem=record['problem'],
                role=split[record['problem']], generator=record['generator'],
                tokens=len(raw), prompt_tokens=first, steps=len(ranges),
                cache=str((CACHE/trace['file']).resolve()), replay=str(saved['replay_fidelity'])))
            errors.append(error)
        if (index+1)%25==0:
            print('measured',index+1,'/400',flush=True)
    write_json(args.output/'manifest.json',dict(records=roster, data=str(DATA),
        cache=str(CACHE), layer=14, physical_heads=32, labels_used=False,
        limitations='attention-only observer transfer; not original 1024-head response/Jacobian method'))
    write_json(args.output/'measurement_complete.json',dict(status='complete', answers=len(roster),
        tokens=sum(r['tokens'] for r in roster), maximum_attention_row_mass_error=max(errors),
        original_token_ids_and_step_ranges='exactly verified against local tokenizer and original step text',
        split='problem SHA256 sorted; 20% fit / 20% dev / 60% evaluation; labels not stratified'))


if __name__=='__main__':
    main()
