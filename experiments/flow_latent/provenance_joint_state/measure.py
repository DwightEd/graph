"""Measure source intervention and signed native message adoption without labels."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter

from experiments.decision_risk_flow.run import load_model
from experiments.token_backtrace.grounded_projection import (
    capture_projection_inputs, layer_finish, native_layout, observed_attention,
    replay_queries, rotate, token_logp,
)
from experiments.token_backtrace.grounded_projection_data import write_json
from experiments.flow_latent.data import digest_files


LOCAL_NEIGHBORS = 8


@contextmanager
def block_source_reads(model, source_mask, length):
    """Block source keys at every non-source query; preserve tokens and positions."""
    source = torch.zeros(length, dtype=torch.bool, device=model.device)
    source[:len(source_mask)] = torch.tensor(source_mask, device=model.device)
    allowed = torch.ones(length, length, dtype=torch.bool, device=model.device).tril()
    allowed &= source[:, None] | ~source[None, :]
    assert allowed.any(-1).all(), 'Source mask removes every causal key'
    mask = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, -torch.inf)

    def replace_mask(module, args, kwargs):
        return args, dict(kwargs, attention_mask=mask[None, None])

    handles = [layer.self_attn.register_forward_pre_hook(replace_mask, with_kwargs=True)
               for layer in model.model.layers]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def capture_world(adapter, case):
    prompt = len(case['source']['prompt_with_source'])
    tokens = case['source']['prompt_with_source'] + case['response']['answer_ids']
    ids = adapter.input_ids(tokens)
    with capture_projection_inputs(adapter.native, prompt) as records:
        hidden = adapter.native.model(ids, use_cache=False).last_hidden_state[0]
    target = ids[0, prompt:]
    native_logp = token_logp(adapter.native, hidden[prompt - 1:-1], target)
    positions = torch.arange(len(tokens), device=ids.device)
    cosine, sine = adapter.native.model.rotary_emb(
        adapter.native.model.embed_tokens(ids), positions[None])
    del hidden
    with block_source_reads(adapter.native, case['source']['source_mask'], len(tokens)):
        hidden = adapter.native.model(ids, use_cache=False).last_hidden_state[0]
    blocked_logp = token_logp(adapter.native, hidden[prompt - 1:-1], target)
    return records, cosine[0], sine[0], target, native_logp, blocked_logp


def query_attention(adapter, index, hidden, positions, layout, cosine, sine):
    """Independent queries with recomputed self KV and fixed native earlier KV."""
    layer = adapter.layers[index]
    heads, kv_heads, width = adapter.head_layout(index)
    normalized = layer.input_layernorm(hidden)
    query = layer.self_attn.q_proj(normalized).reshape(-1, heads, width).transpose(0, 1)
    self_key = layer.self_attn.k_proj(normalized).reshape(-1, kv_heads, width).transpose(0, 1)
    query = rotate(query, cosine[positions], sine[positions], adapter.implementation)
    self_key = rotate(self_key, cosine[positions], sine[positions], adapter.implementation)
    self_key = self_key.repeat_interleave(heads // kv_heads, 0)
    key, value = layout
    scores = query @ key.transpose(-1, -2)
    query_rows = torch.arange(len(positions), device=positions.device)
    scores[:, query_rows, positions] = (query * self_key).sum(-1)
    scores = scores * layer.self_attn.scaling
    future = torch.arange(len(value), device=positions.device)[None] > positions[:, None]
    weights = scores.masked_fill(future[None], -torch.inf).softmax(-1).permute(1, 0, 2)
    messages = torch.einsum('bhk,khd->bhd', weights, value)
    self_value = layer.self_attn.v_proj(normalized).reshape(-1, kv_heads, width)
    self_value = self_value.repeat_interleave(heads // kv_heads, 1)
    self_weight = weights[query_rows, :, positions, None]
    return messages + self_weight * (self_value - value[positions]), weights


def query_pullback(adapter, records, layouts, rows, prompt, cosine, sine, actual):
    positions = rows + prompt - 1
    hidden = records[0]['residual'][rows.cpu()].to(actual.device).requires_grad_(True)
    head_messages, attention = [], []
    for index, layer in enumerate(adapter.layers):
        message, weights = query_attention(
            adapter, index, hidden, positions, layouts[index], cosine, sine)
        head_messages.append(message)
        attention.append(weights.detach())
        hidden = layer_finish(layer, hidden, message)
    logits = adapter.native.lm_head(adapter.native.model.norm(hidden))
    batch_rows = torch.arange(len(rows), device=actual.device)
    competing = logits.detach().clone()
    competing[batch_rows, actual] = -torch.inf
    rival = competing.argmax(-1)
    margin = logits[batch_rows, actual] - logits[batch_rows, rival]
    logp = logits[batch_rows, actual] - logits.logsumexp(-1)
    gradients = torch.autograd.grad(margin.sum(), head_messages)
    return attention, gradients, rival.detach(), margin.detach(), logp.detach()


@torch.no_grad()
def message_observations(weights, gradient, values, rows, prompt, source_mask):
    keys = torch.arange(len(values), device=values.device)
    source = torch.zeros(len(values), dtype=torch.bool, device=values.device)
    source[:prompt] = torch.tensor(source_mask, device=values.device)
    masks = (source, keys >= prompt, ~(source | (keys >= prompt)))
    messages = torch.stack([
        torch.einsum('bhk,khd->bhd', weights * mask[None, None], values)
        for mask in masks
    ], dim=1)
    signed = torch.einsum('bchd,bhd->bch', messages, gradient)

    neighbors = rows[:, None] - torch.arange(1, LOCAL_NEIGHBORS + 1, device=rows.device)
    valid = neighbors >= 0
    key_ids = neighbors.clamp_min(0) + prompt
    edge_values = values[key_ids]
    selected_weights = weights.gather(2, key_ids[:, None].expand(-1, weights.shape[1], -1))
    adopted = torch.einsum('blhd,bhd->blh', edge_values, gradient)
    adopted *= selected_weights.transpose(1, 2) * valid[..., None]
    edge_mass = selected_weights.transpose(1, 2) * valid[..., None]
    return messages, signed, adopted, edge_mass


@torch.no_grad()
def finite_effects(adapter, records, layouts, rows, prompt, cosine, sine, target,
                   rivals, native_margin, source_mask, layer=15):
    positions = rows + prompt - 1
    all_positions = torch.arange(prompt - 1, len(layouts[layer][1]), device=target.device)
    weights, values = observed_attention(
        adapter, layer, records[layer], all_positions, cosine, sine)
    source = torch.zeros(len(values), dtype=torch.bool, device=target.device)
    source[:prompt] = torch.tensor(source_mask, device=target.device)
    message = torch.einsum('bhk,khd->bhd', weights[rows][:, :, source], values[source])
    generator = torch.Generator(device=target.device).manual_seed(73)
    random = torch.randn(message.shape, device=target.device, generator=generator)
    random *= message.norm(dim=-1, keepdim=True) / random.norm(dim=-1, keepdim=True)
    effects = {}
    for name, delta in [('identity', message * 0), ('quarter', -.25 * message),
                        ('full', -message), ('random', random)]:
        hidden = replay_queries(adapter, records, layer, rows, positions, delta, cosine, sine)
        logits = adapter.native.lm_head(hidden)
        batch_rows = torch.arange(len(rows), device=target.device)
        margin = logits[batch_rows, target[rows]] - logits[batch_rows, rivals]
        effects[name] = (native_margin - margin).cpu().numpy()
    return effects


def measured_case(adapter, case, directory, max_positions, batch, finite):
    started = time.time()
    prompt = len(case['source']['prompt_with_source'])
    records, cosine, sine, target, native_logp, blocked_logp = capture_world(adapter, case)
    count = len(target)
    rows = np.arange(count)
    if max_positions and count > max_positions:
        rows = np.unique(np.linspace(0, count - 1, max_positions, dtype=int))
    layouts = [native_layout(adapter, i, r, cosine, sine) for i, r in enumerate(records)]
    layers = len(records)
    heads, _, width = adapter.head_layout(0)
    raw = np.lib.format.open_memmap(directory / 'channel_messages.npy', mode='w+',
        dtype=np.float16, shape=(len(rows), layers, 3, heads, width))
    signed = np.zeros((len(rows), layers, 3, heads), np.float32)
    edge_signed = np.zeros((len(rows), LOCAL_NEIGHBORS, layers, heads), np.float32)
    edge_mass = np.zeros_like(edge_signed)
    self_diagonal = np.zeros((count, layers, heads), np.float32)
    all_positions = torch.arange(prompt - 1, prompt + count, device=target.device)
    for layer, record in enumerate(records):
        weights, _ = observed_attention(adapter, layer, record, all_positions, cosine, sine)
        post_rows = torch.arange(count, device=target.device)
        self_diagonal[:, layer] = weights[post_rows + 1, :, all_positions[1:]].cpu()
    rivals, margins, errors = [], [], []
    finite_rows = {}
    for begin in range(0, len(rows), batch):
        selected = torch.tensor(rows[begin:begin + batch], device=target.device)
        weights, gradients, rival, margin, logp = query_pullback(
            adapter, records, layouts, selected, prompt, cosine, sine, target[selected])
        errors.append(float((logp.cpu() - native_logp[selected.cpu()]).abs().max()))
        for layer in range(layers):
            messages, adoption, local_adoption, local_mass = message_observations(
                weights[layer], gradients[layer], layouts[layer][1], selected,
                prompt, case['source']['source_mask'])
            span = slice(begin, begin + len(selected))
            raw[span, layer] = messages.cpu().numpy()
            signed[span, layer] = adoption.cpu().numpy()
            edge_signed[span, :, layer] = local_adoption.cpu().numpy()
            edge_mass[span, :, layer] = local_mass.cpu().numpy()
        if finite:
            effects = finite_effects(adapter, records, layouts, selected, prompt, cosine,
                sine, target, rival, margin, case['source']['source_mask'])
            for name, values in effects.items():
                finite_rows.setdefault(name, []).append(values)
        rivals.append(rival.cpu().numpy())
        margins.append(margin.cpu().numpy())
        print(f'MEASURE {case["id"]} {begin + len(selected)}/{len(rows)}', flush=True)
    raw.flush()
    assert max(errors) < 3e-4, f'{case["id"]}: independent native replay {max(errors)}'
    np.savez_compressed(directory / 'observations.npz', rows=rows,
        token_id=target.cpu().numpy(), nll=-native_logp.numpy(),
        source_gap=(blocked_logp - native_logp).numpy(), signed=signed,
        edge_signed=edge_signed, edge_mass=edge_mass, self_diagonal=self_diagonal,
        rival=np.concatenate(rivals), margin=np.concatenate(margins),
        **{'finite_' + k: np.concatenate(v) for k, v in finite_rows.items()})
    metadata = dict(id=case['id'], tokens=count, measured_positions=len(rows),
        all_tokens_measured=len(rows) == count, replay_logp_max=max(errors),
        source_mask_count=sum(case['source']['source_mask']), seconds=time.time() - started,
        head_axes=[layers, heads, width], cut_for_finite=15 if finite else None,
        no_labels=True, source_world='all-layer same-position non-source-query/source-key block',
        adoption='full independent-current-query pullback; fixed native earlier KV')
    write_json(directory / 'measurement.json', metadata)
    print(f'DONE {case["id"]} {metadata["seconds"]:.1f}s', flush=True)


def freeze_run(output, source_inputs, ids, max_positions, finite):
    inputs = json.loads(source_inputs.read_text())
    if ids:
        inputs['cases'] = [c for c in inputs['cases'] if c['id'] in ids]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'inputs.json', inputs)
    paths = [Path(__file__),
        Path('experiments/token_backtrace/grounded_projection.py'),
        Path('experiments/decision_risk_flow/run.py'),
        Path('experiments/decision_risk_flow/precision.py'), source_inputs]
    freeze = dict(hashes=digest_files(paths), max_positions=max_positions,
        finite=finite, neighbor_count=LOCAL_NEIGHBORS, source_labels_used=False,
        scope='source-disjoint existing exploratory roster; no blind confirmation')
    write_json(output / 'measurement_freeze.json', freeze)
    snapshot = output / 'code_snapshot'
    snapshot.mkdir()
    for path in paths[:-1]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('outputs/flow_latent_20261007_capture/inputs.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--max-positions', type=int, default=0)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--finite', action='store_true')
    args = parser.parse_args()
    inputs = freeze_run(args.output, args.inputs, args.ids, args.max_positions, args.finite)
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ModelAdapter(load_model(inputs['model']))
    started = time.time()
    for case in inputs['cases']:
        directory = args.output / case['id']
        directory.mkdir()
        measured_case(adapter, case, directory, args.max_positions, args.batch, args.finite)
    write_json(args.output / 'execution.json', dict(status='DONE', cases=len(inputs['cases']),
        seconds=time.time() - started, peak_memory=torch.cuda.max_memory_allocated(),
        full_forwards=2 * len(inputs['cases']), no_natural_label_fit=True))


if __name__ == '__main__':
    main()
