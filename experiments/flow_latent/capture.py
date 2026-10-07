"""Same-world native full-head vectors, complete rows and value transport."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter
from experiments.decision_risk_flow.run import load_model
from experiments.token_backtrace.grounded_projection_run import capture
from experiments.token_backtrace.grounded_projection import observed_attention
from experiments.token_backtrace.grounded_projection_data import write_json
from .data import prepare, digest_files
from .flow import coordinate_projection, matched_mappings, rooted_paths


def create_arrays(directory, count, length):
    shapes = dict(node=(count, 32, 32, 128), values=(length, 32, 32, 128),
                  attention=(32, count, 32, length))
    return {name: np.lib.format.open_memmap(directory / (name + '.npy'),
            mode='w+', dtype=np.float16, shape=shape) for name, shape in shapes.items()}


def transported_features(attention, values, source_mask, prompt, mappings):
    device = attention.device
    source = torch.zeros(values.shape[0], dtype=torch.bool, device=device)
    source[:prompt] = torch.tensor(source_mask, device=device)
    history = torch.arange(values.shape[0], device=device) >= prompt
    other = ~(source | history)
    projected = values @ torch.tensor(coordinate_projection(), device=device)
    features = {}
    for kind in ('native', 'rewired'):
        weights = attention
        if kind == 'rewired':
            mapping = torch.tensor(np.stack(mappings), device=device)
            weights = weights.gather(2, mapping)
        for name, mask in (('source', source), ('history', history), ('other', other)):
            message = torch.einsum('thk,khd->thd', weights[:, :, mask], projected[mask])
            features[kind + '_' + name] = message.cpu().numpy()
        edges = weights[:, :, prompt:].cpu().numpy()
        features[kind + '_root'] = rooted_paths(features[kind + '_source'], edges)
    return features


@torch.no_grad()
def capture_case(adapter, case, output):
    started = time.time()
    directory = output / case['id']
    directory.mkdir(exist_ok=True)
    prompt = len(case['source']['prompt_with_source'])
    tokens = case['source']['prompt_with_source'] + case['response']['answer_ids']
    records, cosine, sine, target, baseline, _, _, _ = capture(adapter, case)
    count = len(target)
    arrays = create_arrays(directory, count, len(tokens))
    mappings, movable = matched_mappings(tokens, prompt, case['source']['source_mask'])
    positions = torch.arange(prompt - 1, len(tokens), device=target.device)
    collected = {}
    errors, movable_mass, changed_mass = [], [], []
    projection = coordinate_projection()
    for layer, record in enumerate(records):
        attention, values = observed_attention(adapter, layer, record, positions, cosine, sine)
        reconstructed = torch.einsum('thk,khd->thd', attention, values)
        original = record['head'].reshape(count + 1, 32, 128).to(target.device)
        errors.append(float((reconstructed - original).abs().max()))
        arrays['node'][:, layer] = original[1:].cpu().numpy()
        arrays['values'][:, layer] = values.cpu().numpy()
        arrays['attention'][layer] = attention[:-1].cpu().numpy()
        collected.setdefault('node', []).append((original[1:] @ torch.tensor(
            projection, device=target.device)).cpu().numpy())
        features = transported_features(attention[:-1], values,
            case['source']['source_mask'], prompt, mappings)
        for name, feature in features.items():
            collected.setdefault(name, []).append(feature)
        mask = torch.tensor(np.stack(movable), device=target.device)
        movable_mass.append(float((attention[:-1] * mask[:, None]).sum(-1).mean()))
        mapping = torch.tensor(np.stack(mappings), device=target.device)
        changed_mass.append(float((attention[:-1] - attention[:-1].gather(2, mapping)).abs().sum(-1).mean() / 2))
        del attention, values, reconstructed, original
    assert max(errors) < 2e-4, f'{case["id"]}: reconstruction {max(errors)}'
    for array in arrays.values():
        array.flush()
    features = {name: np.stack(parts, axis=1) for name, parts in collected.items()}
    np.savez(directory / 'features.npz', token_id=target.cpu().numpy(),
             nll=-baseline.numpy(), **features)
    write_json(directory / 'capture.json', dict(id=case['id'], seconds=time.time() - started,
        reconstruction_max=max(errors), movable_mass=np.mean(movable_mass),
        changed_mass=np.mean(changed_mass), prompt=prompt, tokens=count,
        original_token_ids=tokens, head_axes=[32, 32, 128], attention_floor_applied=False))
    print(f'CAPTURE {case["id"]} {case["cohort"]} T={count} error={max(errors):.2g} '
          f'moved={np.mean(changed_mass):.3f} seconds={time.time()-started:.1f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = (json.loads((args.output / 'inputs.json').read_text())
              if args.output.exists() else prepare(args.output))
    files = [Path(__file__).with_name(name) for name in ('capture.py', 'data.py', 'flow.py')]
    files += [args.output / 'inputs.json']
    hashes = digest_files(files)
    freeze_path = args.output / 'capture_freeze.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == hashes, 'Use a fresh directory for changed capture code'
    else:
        write_json(freeze_path, hashes)
        snapshot = args.output / 'capture_code'
        snapshot.mkdir()
        for path in files[:-1]:
            (snapshot / path.name).write_bytes(path.read_bytes())
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ModelAdapter(load_model(inputs['model']))
    started = time.time()
    for case in inputs['cases']:
        if not (args.output / case['id'] / 'capture.json').exists():
            capture_case(adapter, case, args.output)
    write_json(args.output / 'execution.json', dict(status='DONE', cases=len(inputs['cases']),
        forwards=len(inputs['cases']), seconds=time.time() - started,
        peak_memory=torch.cuda.max_memory_allocated(), no_natural_label_fit=True))


if __name__ == '__main__':
    main()
