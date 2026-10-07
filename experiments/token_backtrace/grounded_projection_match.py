"""Match finite-patch readouts to zero-dose replay in the same numerical layout.

Full-sequence GEMMs and independent-query GEMMs differ slightly in FP32.
Recompute only the zero-dose baselines; retain all frozen finite interventions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter
from experiments.decision_risk_flow.run import load_model
from .grounded_projection_data import write_json
from .grounded_projection_run import CUTS, KINDS, capture, response


def corrected_scores(previous, identity, baseline):
    scores = dict(previous)
    difference = baseline[:, None].numpy() - identity.numpy()
    for kind in KINDS:
        scores[kind + '_layers'] = previous[kind + '_layers'] - difference
        scores[kind] = scores[kind + '_layers'].mean(1)
    scores['edge_increment'] = scores['graph'] - scores['rewired']
    scores['zero_replay_bias'] = difference
    return scores


def initialize(previous, output):
    inputs = json.loads((previous / 'inputs.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'inputs.json', inputs)
    protocol = json.loads((previous / 'protocol.json').read_text())
    original = [previous / c['id'] / 'scores.npz' for c in inputs['cases']]
    protocol.update(numerical_baseline='zero-dose replay, same token batch and structural cut',
        original=str(previous), correction_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_score_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in original})
    write_json(output / 'protocol.json', protocol)
    return inputs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1', type=Path, required=True)
    parser.add_argument('--v2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = initialize(args.v1, args.output / 'v1')
    other = initialize(args.v2, args.output / 'v2')
    assert inputs == other
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ModelAdapter(load_model(inputs['model']))
    started = time.time()
    for case in inputs['cases']:
        records, cosine, sine, target, baseline, source, kernel, similarity = capture(adapter, case)
        heads, _, width = adapter.head_layout(0)
        delta = torch.zeros(len(target), heads, width)
        prompt = len(case['source']['prompt_with_source'])
        identity = torch.stack([response(adapter, records, layer, delta, cosine, sine, target, prompt)
                                for layer in CUTS], 1)
        for name, previous in (('v1', args.v1), ('v2', args.v2)):
            old = dict(np.load(previous / case['id'] / 'scores.npz'))
            assert np.array_equal(old['token_id'], target.cpu().numpy())
            assert np.max(np.abs(old['nll'] + baseline.numpy())) < 2e-4
            scores = corrected_scores(old, identity, baseline)
            directory = args.output / name / case['id']
            directory.mkdir()
            np.savez_compressed(directory / 'scores.npz', **scores)
        print(f'ZERO-MATCH {case["id"]} max={float((identity-baseline[:,None]).abs().max()):.6g}', flush=True)
    for name in ('v1', 'v2'):
        write_json(args.output / name / 'execution.json', dict(status='DONE', cases=len(inputs['cases']),
            correction_seconds=time.time()-started, scoring_unchanged=True,
            correction_used_annotations=False, peak_memory=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
