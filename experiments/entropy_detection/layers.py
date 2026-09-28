"""Native layer uncertainty under the complete original prompt; no source edits."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from .run import exposed_sources, read_json, write_json
from experiments.probabilistic_detection.cases import KNOWN_CASES

LAYERS = (16, 24, 28, 31, 32)


def roster(packs, fixtures, fit_sources=24, dev_sources=12):
    excluded = exposed_sources(packs)
    result, originals = [], {}
    for task in ('QA', 'Summary', 'Data2txt'):
        for split in ('train', 'test'):
            meta = read_json(packs/f'{task}_{split}.json')
            records = meta['records']
            for record in records:
                if record['id'] in KNOWN_CASES:
                    row = dict(record, key=record['id'], role='regression', root=meta['source_cache'])
                    originals[record['id']] = row
                    result.append(row)
            if split != 'train':
                continue
            for partition, count in (('fit', fit_sources), ('dev', dev_sources)):
                eligible = [r for r in records if r['partition'] == partition and r['source_id'] not in excluded]
                eligible.sort(key=lambda r: hashlib.sha256(('entropy-layers-42/'+r['id']).encode()).hexdigest())
                seen = set()
                for record in eligible:
                    if record['source_id'] in seen:
                        continue
                    result.append(dict(record, key=record['id'], role=partition, root=meta['source_cache']))
                    seen.add(record['source_id'])
                    if len(seen) == count:
                        break
                if len(seen) != count:
                    raise ValueError('Insufficient independent reference sources')
    inputs = {r['key']: r for r in read_json(fixtures/'REGRESSION_INPUTS.json')['records']}
    for row in read_json(fixtures/'REGRESSION_EVALUATION.json')['records']:
        if row.get('manual_positive'):
            result.append(dict(originals[row['response_id']], key=row['key'], role='manual_positive',
                manual_answer=inputs[row['key']]))
    return result


def layer_statistics(model, hidden_states, positions, targets, layers=LAYERS):
    """Final checkpoint already includes RMSNorm; intermediate ones do not."""
    chunks = []
    for begin in range(0, len(positions), 32):
        index = positions[begin:begin+32]
        target = targets[begin:begin+32]
        logps = []
        for layer in layers:
            hidden = hidden_states[layer][0, index]
            if layer != len(hidden_states)-1:
                hidden = model.model.norm(hidden)
            logps.append(model.lm_head(hidden).float().log_softmax(-1))
        final = logps[-1]
        top = final.argmax(-1)
        values = []
        for lp in logps:
            probability = lp.exp()
            values.extend((-(probability*lp).sum(-1),
                -lp.gather(-1, target[:, None]).squeeze(-1),
                (final.exp()*(final-lp)).sum(-1).clamp_min(0),
                lp.gather(-1, top[:, None]).squeeze(-1)))
        chunks.append(torch.stack(values, -1).cpu().numpy())
    return np.concatenate(chunks)


def capture(model, prompt, answer):
    ids = torch.tensor([prompt+answer[:-1]], device=model.device)
    positions = torch.arange(len(prompt)-1, ids.shape[1], device=model.device)
    targets = torch.tensor(answer, device=model.device)
    with torch.inference_mode():
        output = model.model(ids, use_cache=False, output_hidden_states=True)
        return layer_statistics(model, output.hidden_states, positions, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packs', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-sources', type=int, default=24)
    parser.add_argument('--dev-sources', type=int, default=12)
    parser.add_argument('--reuse', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = roster(args.packs, args.fixtures, args.fit_sources, args.dev_sources)
    write_json(args.output/'manifest.json', dict(records=records, model=str(args.model), layers=LAYERS,
        selection='sha256 identity order, one response per source',
        fit_sources=args.fit_sources, dev_sources=args.dev_sources, reused_from=str(args.reuse),
        source_deletion=False, labels_used_for_roster=False, observer_not_original_generator=True))
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True).to('cuda:0').eval()
    started = perf_counter()
    calls = 0
    for number, record in enumerate(records):
        prior = args.reuse/f"{record['key']}.npz" if args.reuse else None
        if prior is not None and prior.exists():
            shutil.copyfile(prior, args.output/prior.name)
            continue
        root = Path(record['root'])
        prompt = read_json(root/record['source_file'])['prompt_with_source']
        response = read_json(root/record['directory']/'response.json')
        answer = record.get('manual_answer', {}).get('token_ids', response['answer_ids'])
        values = capture(model, prompt, answer)
        calls += 1
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite native uncertainty')
        np.savez_compressed(args.output/f"{record['key']}.npz", values=values, token_ids=answer)
        print(json.dumps(dict(done=number+1, total=len(records), key=record['key'],
            tokens=len(answer), seconds=perf_counter()-started)), flush=True)
    write_json(args.output/'completed.json', dict(forwards=calls, reused=len(records)-calls, seconds=perf_counter()-started,
        max_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
