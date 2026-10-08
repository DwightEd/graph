"""Frozen transfer to two exposed manual candidate pairs, not automatic detection."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch
from transformers import AutoTokenizer
from state_audit.model.adapter import ModelAdapter

from experiments.decision_risk_flow.run import load_model
from ..provenance_joint_state.matrix_audit import sha256, write_json
from ..provenance_joint_state.native_prefix_pilot import INPUTS
from ..provenance_joint_state.prechoice_states_pilot import OUTPUT as OLD_STATES
from .observe import collect_control, WINDOW
from .readout import OrderedCompatibility, WideCurrentCompatibility, row_permutations, transform_view
from .run import MODEL


def natural_controls(tokenizer):
    inputs = json.loads(INPUTS.read_text())
    controls = []
    for case in inputs['cases']:
        end = case['prompt_length'] + case['target']
        prefix = case['token_ids'][:end]
        prompt = tokenizer.decode(case['token_ids'][:case['prompt_length']], skip_special_tokens=False)
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        assert encoded['input_ids'] == case['token_ids'][:case['prompt_length']]
        begin = re.search(r'(?i)passage 1:', prompt).start()
        finish = prompt.index('\n\nIn case the passages', begin)
        # All three source passages, not the manually identified correct Passage3.
        source_mask = [right > begin and left < finish for left, right in encoded['offset_mapping']]
        source_mask += [False] * case['target']
        controls.append(dict(id=case['id'], source_id=case['source_id'], token_ids=prefix,
            source_mask=source_mask, candidate_ids=[case['rival_token'], case['token_ids'][end]],
            correct=0, family='manual_old_pair', source_char_span=[begin, finish]))
    return controls


def capture(output):
    output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    controls = natural_controls(tokenizer)
    write_json(output / 'controls.json', controls)
    write_json(output / 'PROTOCOL.json', dict(scope='2 exposed old manual candidates; no selection or fit',
        source_range='automatically delimited all3 passages; no gold Passage3 mask',
        input='ends before current target; entire prior prefix', planned_full_forwards=2,
        hashes={str(p): sha256(p) for p in [Path(__file__), INPUTS, output / 'controls.json']}))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = ModelAdapter(load_model(MODEL))
    states, embeddings, observations, source_heads = [], [], [], []
    for control in controls:
        values, heads, _, native, errors = collect_control(adapter, control)
        old = np.load(OLD_STATES / control['id'] / 'states.npy', mmap_mode='r')[-WINDOW:]
        error = float(np.max(np.abs(values[:, :, :3] - old)))
        assert error == 0, 'Natural r/a/m must reproduce existing pre-choice cache'
        with torch.no_grad():
            candidate = adapter.native.lm_head.weight[control['candidate_ids']].float()
            embeddings.append(torch.nn.functional.normalize(candidate, dim=-1).cpu().numpy())
        states.append(values)
        source_heads.append(heads)
        observations.append(dict(id=control['id'], native=native, errors=errors,
                                 old_state_max_error=error))
    np.save(output / 'states.npy', np.stack(states))
    np.save(output / 'source_heads.npy', np.stack(source_heads))
    np.save(output / 'candidates.npy', np.stack(embeddings))
    write_json(output / 'observations.json', observations)
    write_json(output / 'COLLECTION.json', dict(status='DONE', full_forwards=2, fits=0,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), observations=observations))
    print(json.dumps(observations, indent=2), flush=True)


@torch.no_grad()
def transfer(args):
    args.output.mkdir(exist_ok=False)
    paths = list(args.cache.glob('*.npy')) + list(args.cache.glob('*.json'))
    paths += [Path(__file__), Path(__file__).with_name('readout.py')]
    paths += list(args.fits.glob('*/model.pt'))
    write_json(args.output / 'PROTOCOL.json', dict(stage='frozen_transfer', fits=str(args.fits),
        scope='exposed2manual candidate pairs, adaptive development; zero new fitting or LLM forwards',
        hashes={str(p): sha256(p) for p in paths}))
    controls = json.loads((args.cache / 'controls.json').read_text())
    states = torch.from_numpy(np.load(args.cache / 'states.npy'))
    candidates = torch.from_numpy(np.load(args.cache / 'candidates.npy'))
    permutations = row_permutations(len(states), WINDOW)
    results = []
    score_arrays = []
    for directory in sorted(args.fits.glob('*_seed*')):
        checkpoint = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
        model = OrderedCompatibility()
        if checkpoint['variant'] == 'wide_single':
            model = WideCurrentCompatibility()
        model.load_state_dict(checkpoint['state_dict'])
        standardized = (states - checkpoint['mean']) / checkpoint['scale']
        values = transform_view(standardized, checkpoint['variant'], permutations)
        scores = model(values, candidates).numpy()
        score_arrays.append(scores)
        results.append(dict(run=directory.name,
            cases=[dict(id=row['id'], correct_minus_wrong=float(score[0] - score[1]),
                        correct_ranked_first=bool(score[0] > score[1]))
                   for row, score in zip(controls, scores)]))
    np.save(args.output / 'scores.npy', np.stack(score_arrays))
    write_json(args.output / 'FREEZE.json', dict(scores_sha256=sha256(args.output / 'scores.npy'),
        checkpoint_hashes={str(p): sha256(p) for p in sorted(args.fits.glob('*/model.pt'))},
        cache=str(args.cache), scope='frozen manual-pair transfer; no natural fitting'))
    write_json(args.output / 'RESULTS.json', dict(status='DONE', new_full_forwards=0,
        fits=0, scope='adaptively developed method, exposed2manual pairs; not natural token AUROC', runs=results))
    print(json.dumps([row for row in results if row['run'].startswith('ordered_')], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('capture', 'transfer'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--fits', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'capture':
        capture(args.output)
    else:
        transfer(args)


if __name__ == '__main__':
    main()
