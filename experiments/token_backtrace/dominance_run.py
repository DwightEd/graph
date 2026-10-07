"""Test native source/history dominance on frozen matched natural token positions."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter
from experiments.decision_risk_flow.run import load_model
from .dominance import choice_readout, measure_layer
from .dominance_data import prepare_native
from .grounded_projection import capture_projection_inputs
from .grounded_projection_data import write_json


@torch.no_grad()
def capture(adapter, case):
    prompt, tokens = case['prompt_length'], case['token_ids']
    ids = adapter.input_ids(tokens)
    with capture_projection_inputs(adapter.native, prompt) as records:
        all_hidden = adapter.native.model(ids, use_cache=False).last_hidden_state[0]
    rows = torch.tensor([case['target'], case['control']], device=ids.device)
    positions = rows + prompt - 1
    actual = ids[0, rows + prompt]
    logits = adapter.native.lm_head(all_hidden[positions]).float()
    logits[torch.arange(len(rows), device=ids.device), actual] = -torch.inf
    rival = logits.argmax(-1)
    if 'rival_token' in case:
        rival[0] = case['rival_token']
    baseline = choice_readout(adapter.native, all_hidden[positions], actual, rival)
    all_positions = torch.arange(len(tokens), device=ids.device)
    cosine, sine = adapter.native.model.rotary_emb(adapter.native.model.embed_tokens(ids), all_positions[None])
    return records, rows, actual, rival, baseline, cosine[0], sine[0]


@torch.no_grad()
def native_canary(adapter, case, records, layer, rows, delta, cosine, sine, actual, rival, reference):
    """One actual full-prefix finite patch, no other query position is edited."""
    position = rows[0] + case['prompt_length'] - 1
    def change(message):
        modified = message.clone()
        modified[position] += delta[1, 0]
        return modified
    with adapter.bind('head_readout', layer, change, edit=True):
        hidden = adapter.forward(case['token_ids'])[position:position + 1]
    measured = choice_readout(adapter.native, hidden, actual[:1], rival[:1])
    error = float(abs(measured['margin'][0] - reference['scores']['margin'][1, 0]))
    assert error < 1e-3, f'finite native margin error {error}'
    return error


def save_case(native_directory, case, layers, measurements, baseline, rival, canary):
    directory = native_directory / case['id']
    directory.mkdir()
    arrays = {}
    for layer, measurement in measurements.items():
        for name, values in measurement['scores'].items():
            arrays[f'{layer}:{name}'] = values
        for field in ('masses', 'message_norms'):
            for name, values in measurement[field].items():
                arrays[f'{layer}:{field}:{name}'] = values
        arrays[f'{layer}:self_diagonal'] = measurement['self_diagonal']
    np.savez_compressed(directory / 'observations.npz', **arrays)
    write_json(directory / 'metadata.json', dict(names=next(iter(measurements.values()))['names'],
        layers=layers, baseline={name: values.tolist() for name, values in baseline.items()},
        rival_ids=rival.cpu().tolist(), finite_canary=canary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-directory', default='native')
    parser.add_argument('--all-layers', action='store_true')
    args = parser.parse_args()
    inputs = prepare_native(args.output, args.native_directory)
    selection = json.loads((args.output / 'head_selection.json').read_text())
    if args.all_layers:
        directions = np.load(args.output / 'teacher_tangents.npz')
        magnitude = (abs(directions['seed42']) + abs(directions['seed123'])) / 2
        selection['layers'] = list(range(32))
        selection['heads'] = {str(layer): np.argsort(magnitude.reshape(32, 32)[layer])[::-1][:8].tolist()
                              for layer in selection['layers']}
    directory = args.output / args.native_directory
    files = [Path(__file__), Path(__file__).with_name('dominance.py'),
             Path(__file__).with_name('dominance_data.py'), directory / 'inputs.json',
             args.output / 'head_selection.json']
    write_json(directory / 'protocol.json', dict(candidate='actual token versus fixed native best nonactual; explicit operator rivals in inputs override target only',
        layers=selection['layers'], necessity='positive baseline margin becomes negative under channel removal',
        conditional_scope='current attention key channels; residual and earlier KV remain original',
        hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    snapshot = directory / 'code_snapshot'
    snapshot.mkdir()
    for path in files[:3]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ModelAdapter(load_model(json.loads((directory / 'inputs.json').read_text())['model']))
    started = time.time()
    for index, case in enumerate(inputs):
        records, rows, actual, rival, baseline, cosine, sine = capture(adapter, case)
        measurements, canary = {}, None
        for layer in selection['layers']:
            measured, delta = measure_layer(adapter, records, layer, rows, case['prompt_length'],
                cosine, sine, case['source_mask'], selection['heads'][str(layer)], actual, rival)
            assert np.max(abs(measured['scores']['margin'][0] - baseline['margin'])) < 1e-3
            measurements[layer] = measured
            if index == 0 and layer == selection['layers'][0]:
                canary = native_canary(adapter, case, records, layer, rows, delta, cosine, sine,
                                       actual, rival, measured)
        save_case(directory, case, selection['layers'], measurements, baseline, rival, canary)
        print('NATIVE', case['id'], case['kind'], 'baseline margins', baseline['margin'].tolist(), flush=True)
    write_json(directory / 'execution.json', dict(status='DONE', cases=len(inputs),
        target_positions=2 * len(inputs), full_forwards=len(inputs)+1, seconds=time.time()-started,
        peak_memory=torch.cuda.max_memory_allocated(), gold_used_for_position_selection=True))


if __name__ == '__main__':
    main()
