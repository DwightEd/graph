"""Pre-registered second operator: current-query-conditioned semantic cycles."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter
from experiments.decision_risk_flow.run import load_model
from .grounded_projection import observed_attention
from .grounded_projection_cycle import cycle_transport
from .grounded_projection_data import write_json
from .grounded_projection_run import CUTS, capture, response


@torch.no_grad()
def score_case(adapter, case, previous, output):
    started = time.time()
    prompt = len(case['source']['prompt_with_source'])
    records, cosine, sine, target, baseline, source, kernel, similarity = capture(adapter, case)
    scores = dict(np.load(previous / case['id'] / 'scores.npz'))
    assert np.array_equal(scores['token_id'], target.cpu().numpy())
    assert np.max(np.abs(scores['nll'] + baseline.numpy())) < 2e-4
    collected, directions = {'graph': [], 'rewired': []}, {}
    positions = torch.arange(prompt - 1, len(records[0]['key']), device=target.device)
    for layer in CUTS:
        attention, values = observed_attention(adapter, layer, records[layer], positions, cosine, sine)
        delta = cycle_transport(attention, values, source, kernel, prompt)
        directions[layer] = delta
        identity = response(adapter, records, layer, torch.zeros_like(delta['graph']), cosine, sine, target, prompt)
        assert (identity - baseline).abs().max() < 2e-4
        for name in collected:
            patched = response(adapter, records, layer, delta[name], cosine, sine, target, prompt)
            collected[name].append(identity - patched)
    for name, effects in collected.items():
        scores[name + '_v1'] = scores[name]
        scores[name + '_layers'] = torch.stack(effects, 1).numpy()
        scores[name] = scores[name + '_layers'].mean(1)
    scores['edge_increment'] = scores['graph'] - scores['rewired']
    directory = output / case['id']
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / 'scores.npz', **scores)
    torch.save(directions, directory / 'directions.pt')
    write_json(directory / 'execution.json', dict(tokens=len(target), seconds=time.time() - started))
    print(f'CYCLE {case["id"]} T={len(target)} seconds={time.time()-started:.1f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((args.previous / 'inputs.json').read_text())
    parent_protocol = json.loads((args.previous / 'protocol.json').read_text())
    if 'numerical_baseline' not in parent_protocol:
        raise ValueError('Previous scores need a matched zero-dose baseline; run grounded_projection_match')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'inputs.json', inputs)
    files = [Path(__file__), Path(__file__).with_name('grounded_projection_cycle.py'),
             Path(__file__).with_name('grounded_projection.py'),
             Path(__file__).with_name('grounded_projection_run.py'),
             Path(__file__).with_name('grounded_projection_evaluate.py'), args.output / 'inputs.json']
    protocol = parent_protocol
    protocol.update(operator='query-conditioned A(q,s)A(i,s)K(i,s) source/history cycle',
        previous=str(args.previous), hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    write_json(args.output / 'protocol.json', protocol)
    snapshot = args.output / 'code_snapshot'
    snapshot.mkdir()
    for path in files[:-1]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ModelAdapter(load_model(inputs['model']))
    started = time.time()
    for case in inputs['cases']:
        score_case(adapter, case, args.previous, args.output)
    write_json(args.output / 'execution.json', dict(status='DONE', cases=len(inputs['cases']),
        seconds=time.time() - started, peak_memory=torch.cuda.max_memory_allocated(),
        natural_annotation_fields_used_for_scoring=False))


if __name__ == '__main__':
    main()
