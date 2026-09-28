"""Add native values and complete grouped head writes to existing attention."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers.models.llama.modeling_llama import repeat_kv

from experiments.decision_risk_flow.data import inputs, read_json, write_json
from experiments.decision_risk_flow.native import prefill
from experiments.decision_risk_flow.run import load_model
from experiments.message_js.measure import js_rows
from .messages import group_messages


def get_inputs(record, samples, source_root, source_lookup):
    if record['kind'] == 'observer':
        prompt, response = inputs(record)
        source = read_json(Path(record['root']) / record['source_file'])
        answer = response['answer_ids']
    else:
        with np.load(samples / record['trace']) as trace:
            length = int(trace['prompt_length'])
            prompt = trace['token_ids'][:length].tolist()
            answer = trace['token_ids'][length:].tolist()
        source = read_json(source_root / source_lookup[record['source_id']])
    assert prompt == source['prompt_with_source'], record['key']
    return prompt, answer, source['source_mask']


@torch.no_grad()
def capture_answer(model, record, prompt, answer, source_mask, base, operator, output):
    count, length = len(answer), len(prompt) + len(answer) - 1
    directory = output / record['key']
    directory.mkdir()
    cache, _, final = prefill(model, prompt, answer, checkpoints=(32,))
    expected = np.load(operator / record['key'] / 'operator.npz')['final_state']
    error = float(np.linalg.norm(final.cpu().numpy() - expected) / np.linalg.norm(expected))
    assert error < .0005, (record['key'], error)
    attention = np.load(base / record['key'] / 'attention.npy', mmap_mode='r')
    derivative = np.load(base / record['key'] / 'derivative.npy', mmap_mode='r')
    source = torch.zeros(length, dtype=torch.bool, device=model.device)
    source[:len(prompt)] = torch.tensor(source_mask, device=model.device)
    history = torch.arange(length, device=model.device) >= len(prompt)
    masks = (source, history, ~(source | history))
    values_file = np.lib.format.open_memmap(directory / 'values.npy', mode='w+', dtype='float32', shape=(32, 8, length, 128))
    factors_file = np.lib.format.open_memmap(directory / 'head_vectors.npy', mode='w+', dtype='float32', shape=(32, 3, 32, count, 128))
    gram_file = np.lib.format.open_memmap(directory / 'head_gram.npy', mode='w+', dtype='float32', shape=(32, 3, count, 32, 32))
    masses, divergences = [], []
    source_numpy = source.cpu().numpy()
    for layer, module in enumerate(model.model.layers):
        values = cache.layers[layer].values
        values_file[layer] = values[0].cpu().numpy()
        values = repeat_kv(values, module.self_attn.num_key_value_groups)[0].float()
        weight = module.self_attn.o_proj.weight.reshape(4096, 32, 128).permute(1, 0, 2).float()
        weights = torch.tensor(np.asarray(attention[layer]), device=model.device)
        factors, gram, mass, magnitude = group_messages(weights, values, weight, masks)
        factors_file[layer] = factors.cpu().numpy()
        gram_file[layer] = gram.cpu().numpy()
        masses.append(mass.cpu().numpy())
        payload = magnitude.cpu().numpy()
        fields = []
        for head in range(32):
            branch = source_numpy
            raw = np.asarray(attention[layer, head])[:, branch]
            transformed = payload[head][:, branch]
            used = np.abs(np.asarray(derivative[layer, head])[:, branch])
            alpha = np.full(count, .5)
            fields.append(np.stack((js_rows(raw, transformed, alpha), js_rows(transformed, used, alpha)), -1))
        divergences.append(np.stack(fields))
        del values, weight, weights, factors, gram, mass, magnitude
    for saved in (values_file, factors_file, gram_file):
        saved.flush()
    np.savez_compressed(directory / 'measurements.npz', mass=np.stack(masses),
        source_js=np.stack(divergences), token_ids=answer, prompt_length=len(prompt),
        source_mask=source_mask, replay_error=error)
    print(dict(key=record['key'], tokens=count, replay_error=error), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', type=Path, default=Path('outputs/message_operator_20260928_v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct')
    args = parser.parse_args()
    manifest = read_json(args.operator / 'manifest.json')
    records, base, samples = manifest['records'], Path(manifest['base']), Path(manifest['samples'])
    source_root = Path(next(row['root'] for row in records if row['kind'] == 'observer'))
    source_lookup = {str(row['source_id']): row['source_file'] for row in read_json(source_root / 'manifest.json')['records']}
    args.output.mkdir(exist_ok=False)
    write_json(args.output / 'manifest.json', dict(manifest, operator=str(args.operator.resolve()), model=args.model))
    write_json(args.output / 'protocol.json', dict(groups=['source', 'history', 'other_prompt'],
        source_scope='existing template source boundaries; natural prompt IDs checked identical',
        labels_read=False, axes_retained=['layer','physical_head','query','key','native_head_dimension'],
        factorization='native 128-dimensional head values plus unchanged W_O exactly reconstruct output vectors',
        gram='complete 32x32 head Gram per layer, token and group; no random projection',
        controls='norm route and net route, same original source mask and offline16 window'))
    model = load_model(args.model)
    started = perf_counter()
    for record in records:
        prompt, answer, mask = get_inputs(record, samples, source_root, source_lookup)
        capture_answer(model, record, prompt, answer, mask, base, args.operator, args.output)
    write_json(args.output / 'capture_complete.json', dict(status='complete', answers=len(records),
        seconds=perf_counter()-started, labels_read=False, peak_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
