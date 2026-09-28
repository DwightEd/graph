"""Measure full physical-head responses without writing dense edge arrays to disk."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from experiments.decision_risk_flow.data import read_json, write_json, inputs
from experiments.decision_risk_flow.native import prefill, replay, confidence
from experiments.decision_risk_flow.run import load_model
from .measure import FIELDS, layer_features, relation_matrices
from experiments.route_complement.messages import projected_norm


def prefill_with_queries(model, prompt, answer):
    """Keep rotated prompt queries and ordinary original-prefix KV states."""
    chunks = [[] for _ in model.model.layers]
    starts = [0 for _ in model.model.layers]

    def capture(index):
        def hook(module, args, kwargs):
            state = kwargs['hidden_states']
            size = min(state.shape[1], max(0, len(prompt)-starts[index]))
            if size:
                shape = (*state.shape[:-1], -1, module.head_dim)
                query = module.q_proj(state).view(shape).transpose(1, 2)
                key = module.k_proj(state).view(shape).transpose(1, 2)
                query, _ = apply_rotary_pos_emb(query, key, *kwargs['position_embeddings'])
                chunks[index].append(query[0, :, :size].cpu())
            starts[index] += state.shape[1]
        return hook

    handles = [layer.self_attn.register_forward_pre_hook(capture(i), with_kwargs=True)
               for i, layer in enumerate(model.model.layers)]
    try:
        cache, states, final = prefill(model, prompt, answer, checkpoints=(32,))
    finally:
        for handle in handles:
            handle.remove()
    return cache, states, final, [torch.cat(parts, dim=1) for parts in chunks]


def source_edges(model, cache, tokens, positions, answer, alternative, expected, indices, norms, prompt_length):
    final, _, capture = replay(model, cache, tokens, positions, checkpoints=(32,))
    logits = model.lm_head(final).float()
    with torch.no_grad():
        error = float((logits-model.lm_head(expected).float()).abs().max())
    if error > .005:
        raise ValueError(f'Native replay mismatch: {error}')
    targets = torch.tensor(answer, device=model.device)
    other = alternative.to(model.device)
    margin = logits.gather(1, targets[:, None])-logits.gather(1, other[:, None])
    gradients = torch.autograd.grad(margin.sum(), capture.writes)
    reading, effects, routes = [], [], []
    for index, gradient in enumerate(gradients):
        attention, past_value, _ = capture.messages[index]
        magnitude = attention[..., :-1]*norms[index][:, None]
        own = attention[..., -1]*norms[index][:, positions]
        total = magnitude.sum((0, 2))+own.sum(0)
        source = magnitude[..., indices].sum((0, 2))
        history = magnitude[..., prompt_length:].sum((0, 2))
        history = history+own.sum(0)*(positions >= prompt_length)
        routes.append((history-source)/total.clamp_min(1e-30))
        attention = attention[..., indices]
        heads, _, dimension = past_value.shape
        weights = model.model.layers[index].self_attn.o_proj.weight
        weights = weights.reshape(-1, heads, dimension).permute(1, 0, 2).float()
        direction = torch.einsum('bd,hdk->hbk', gradient[0].float(), weights)
        sensitivity = direction @ past_value[:, indices].float().transpose(-1, -2)
        reading.append(attention.cpu().numpy())
        effects.append((sensitivity*attention).cpu().numpy())
    return reading, effects, error, torch.stack(routes).mean(0).cpu().numpy()


def capture_answer(model, prompt, answer, source_mask, special_ids, directory, batch):
    directory.mkdir()
    started = perf_counter()
    cache, _, original, queries = prefill_with_queries(model, prompt, answer)
    observed, alternatives = confidence(model, original, answer, len(prompt), special_ids)
    indices = torch.tensor(np.flatnonzero(source_mask), device=model.device)
    layers, heads, count = len(model.model.layers), model.config.num_attention_heads, len(answer)
    # Bound key-query products while retaining every query and physical head.
    batch = min(batch, max(16, (65536//(len(prompt)+count-1)//16)*16))
    shape = (layers, heads, count, len(indices))
    attention = np.empty(shape, dtype=np.float32)
    derivative = np.empty(shape, dtype=np.float32)
    errors = []
    route = np.empty(count, dtype=np.float64)
    norms = []
    with torch.no_grad():
        for index, module in enumerate(model.model.layers):
            values = repeat_kv(cache.layers[index].values, module.self_attn.num_key_value_groups)[0]
            weights = module.self_attn.o_proj.weight.reshape(-1, heads, values.shape[-1]).permute(1, 0, 2).float()
            norms.append(projected_norm(values, weights))
    for start in range(0, count, batch):
        stop = min(start+batch, count)
        positions = torch.arange(len(prompt)-1+start, len(prompt)-1+stop, device=model.device)
        reading, effects, error, batch_route = source_edges(model, cache, prompt+answer[:-1], positions,
            answer[start:stop], alternatives[start:stop], original[start:stop], indices, norms, len(prompt))
        route[start:stop] = batch_route
        attention[:, :, start:stop] = np.stack(reading)
        derivative[:, :, start:stop] = np.stack(effects)
        errors.append(error)
    values = np.empty((layers, heads, count, len(FIELDS)), dtype=np.float32)
    # Process individual heads to bound source-graph memory on long prompts.
    head_batch = 32 if len(indices)<=800 else 8 if len(indices)<=1600 else 4
    for layer, module in enumerate(model.model.layers):
        keys = repeat_kv(cache.layers[layer].keys, module.self_attn.num_key_value_groups)[0]
        for first in range(0, heads, head_batch):
            last = min(first+head_batch, heads)
            query = queries[layer][first:last].to(model.device)[:, indices]
            key = keys[first:last, indices]
            relations = relation_matrices(query, key, indices)
            reading = torch.tensor(attention[layer, first:last], device=model.device)
            effect = torch.tensor(derivative[layer, first:last], device=model.device)
            values[layer, first:last] = layer_features(reading, effect, relations).cpu().numpy()
    np.savez(directory/'responses.npz', values=values, fields=FIELDS,
        token_ids=answer, prompt_length=len(prompt), source_mask=source_mask)
    np.savez_compressed(directory/'readouts.npz', confidence=observed.numpy(),
        alternative=alternatives.numpy(), token_ids=answer, replay_errors=errors, raw_route=route)
    write_json(directory/'complete.json', dict(seconds=perf_counter()-started,
        tokens=count, max_replay_error=max(errors), physical_heads=layers*heads, query_batch=batch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=Path('outputs/context_response_20260928_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    manifest = read_json(args.pilot/'manifest.json')
    rows = [r for r in manifest['records'] if r['role']=='regression'][:args.limit]
    args.output.mkdir(exist_ok=False)
    model = load_model(manifest['model'])
    for row in rows:
        prompt, response = inputs(row)
        source = read_json(Path(row['root'])/row['source_file'])
        directory = args.output/row['key']
        capture_answer(model, prompt, response['answer_ids'], source['source_mask'],
                       response['special_ids'], directory, args.batch)
        expected = np.load(args.pilot/row['key']/'responses.npz')['values']
        actual = np.load(directory/'responses.npz')['values']
        difference = np.abs(actual-expected)
        error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        selected = [0, 1, 8, 9]
        used_error = float(np.linalg.norm(actual[..., selected]-expected[..., selected])/np.linalg.norm(expected[..., selected]))
        if used_error > .0005:
            raise ValueError(f'Streamed scoring coordinates changed: {row["key"]} {used_error}')
        old_route = np.load(Path(manifest['route_base'])/row['key']/'scores.npz')['raw_route']
        new_route = np.load(directory/'readouts.npz')['raw_route']
        route_error = float(np.abs(old_route-new_route).max())
        if route_error > .0005:
            raise ValueError(f'Route changed: {row["key"]} {route_error}')
        result = read_json(directory/'complete.json')
        result.update(relative_error=error, scoring_coordinate_relative_error=used_error,
                      route_max_error=route_error, max_absolute_error=float(difference.max()))
        write_json(directory/'equivalence.json', result)
        print(row['key'], result, flush=True)
    write_json(args.output/'verification.json', dict(status='passed', answers=len(rows),
        peak_bytes=torch.cuda.max_memory_allocated(), batch=args.batch))


if __name__ == '__main__':
    main()
