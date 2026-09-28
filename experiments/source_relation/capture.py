"""Capture unprojected native prompt Q/K; no labels or semantic annotations."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from experiments.route_complement.capture import get_inputs


@torch.no_grad()
def capture_prompt(model, prompt, directory, expected):
    directory.mkdir()
    layers = len(model.model.layers)
    heads = model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    dim = model.config.hidden_size // heads
    query = np.lib.format.open_memmap(directory / 'query.npy', mode='w+',
        dtype='float32', shape=(layers, heads, len(prompt), dim))
    keys = np.lib.format.open_memmap(directory / 'key.npy', mode='w+',
        dtype='float32', shape=(layers, kv_heads, len(prompt), dim))
    begin = 0
    def hook(index):
        def save(module, args, kwargs):
            state = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            shape = (*state.shape[:-1], -1, dim)
            q = module.q_proj(state).view(shape).transpose(1, 2)
            k = module.k_proj(state).view(shape).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
            query[index, :, begin:begin + state.shape[1]] = q[0].float().cpu().numpy()
        return save
    handles = [layer.self_attn.register_forward_pre_hook(hook(i), with_kwargs=True)
               for i, layer in enumerate(model.model.layers)]
    cache = DynamicCache()
    try:
        for begin in range(0, len(prompt), 256):
            ids = torch.tensor([prompt[begin:begin+256]], device=model.device)
            model.model(ids, past_key_values=cache, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    errors = []
    for index in range(layers):
        keys[index] = cache.layers[index].keys[0].float().cpu().numpy()
        value = cache.layers[index].values[0].float().cpu().numpy()
        reference = np.asarray(expected[index, :, :len(prompt)])
        errors.append(float(np.linalg.norm(value-reference) / max(np.linalg.norm(reference), 1e-12)))
    assert max(errors) < .0005, errors
    query.flush()
    keys.flush()
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('outputs/route_complement_20260928'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = read_json(args.base / 'manifest.json')
    args.output.mkdir(exist_ok=False)
    (args.output / 'sources').mkdir()
    manifest = dict(old, route_base=str(args.base.resolve()))
    write_json(args.output / 'manifest.json', manifest)
    root = Path(next(r['root'] for r in old['records'] if r['kind'] == 'observer'))
    lookup = {str(r['source_id']): r['source_file'] for r in read_json(root / 'manifest.json')['records']}
    model = load_model(old['model'])
    started, seen, verification = perf_counter(), {}, {}
    for row in old['records']:
        prompt, answer, mask = get_inputs(row, Path(old['samples']), root, lookup)
        source = str(row['source_id'])
        if source in seen:
            assert prompt == seen[source], source
            continue
        expected = np.load(args.base / row['key'] / 'values.npy', mmap_mode='r')
        directory = args.output / 'sources' / source
        errors = capture_prompt(model, prompt, directory, expected)
        np.savez_compressed(directory / 'input.npz', token_ids=prompt, source_mask=mask)
        seen[source] = prompt
        verification[source] = dict(reference=row['key'], layer_value_relative_errors=errors)
        print(dict(source=source, prompt=len(prompt), max_error=max(errors), completed=len(seen)), flush=True)
    write_json(args.output / 'capture_complete.json', dict(status='complete', sources=len(seen),
        answers=len(old['records']), seconds=perf_counter()-started, labels_read=False,
        peak_bytes=torch.cuda.max_memory_allocated(), verification=verification))


if __name__ == '__main__':
    main()
