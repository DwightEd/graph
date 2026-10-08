"""Capture ordered prediction states and full native source messages."""
from contextlib import contextmanager

import numpy as np
import torch

from experiments.token_backtrace.grounded_projection import capture_projection_inputs, observed_attention

WINDOW = 8
SITES = ('residual_before', 'attention_write', 'mlp_write', 'source_write')


@contextmanager
def capture_window(model, start):
    records = [{} for _ in model.model.layers]
    handles = []

    def remember_input(index):
        def observe(module, args):
            records[index]['residual_before'] = args[0][0, start:].detach().cpu()
        return observe

    def remember_output(index, name):
        def observe(module, args, output):
            records[index][name] = output[0, start:].detach().cpu()
        return observe

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(remember_input(index)))
        handles.append(layer.self_attn.o_proj.register_forward_hook(remember_output(index, 'attention_write')))
        handles.append(layer.mlp.register_forward_hook(remember_output(index, 'mlp_write')))
        handles.append(layer.register_forward_hook(remember_output(index, 'after')))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def collect_control(adapter, control):
    model = adapter.native
    ids = adapter.input_ids(control['token_ids'])
    start = ids.shape[1] - WINDOW
    with capture_window(model, start) as states:
        with capture_projection_inputs(model, start + 1) as projections:
            hidden = model.model(ids, use_cache=False).last_hidden_state[0, -1]
    logits = model.lm_head(hidden)
    pair_logits = logits[control['candidate_ids']].cpu().numpy()
    probabilities = logits.softmax(-1)
    native = dict(pair_logits=pair_logits.tolist(), greedy_token=int(logits.argmax()),
                  greedy_probability=float(probabilities.max()),
                  candidate_probabilities=probabilities[control['candidate_ids']].cpu().tolist())
    values, heads, mass, errors = source_messages(adapter, ids, control['source_mask'], projections)
    for index, message in enumerate(values):
        states[index]['source_write'] = message
    ordered = torch.stack([torch.stack([row[name] for name in SITES], dim=1) for row in states], dim=1)
    equation_error = max(float(((row['residual_before'] + row['attention_write']) +
                                row['mlp_write'] - row['after']).abs().max()) for row in states)
    return ordered.numpy(), heads, mass, native, dict(head_reconstruction=max(errors),
                                                      state_equation=equation_error)


@torch.no_grad()
def source_messages(adapter, ids, source_mask, records):
    model = adapter.native
    all_positions = torch.arange(ids.shape[1], device=model.device)
    positions = all_positions[-WINDOW:]
    cosine, sine = model.model.rotary_emb(model.model.embed_tokens(ids), all_positions[None])
    source = torch.tensor(source_mask, device=model.device)
    writes, heads, mass, errors = [], [], [], []
    for index, record in enumerate(records):
        attention, values = observed_attention(adapter, index, record, positions, cosine[0], sine[0])
        complete = torch.einsum('whk,khd->whd', attention, values).flatten(1)
        errors.append(float((complete.cpu() - record['head']).abs().max()))
        message = torch.einsum('whk,khd->whd', attention[:, :, source], values[source])
        writes.append(adapter.layers[index].self_attn.o_proj(message.flatten(1)).cpu())
        heads.append(message.cpu().numpy())
        mass.append(attention[:, :, source].sum(-1).cpu().numpy())
    return writes, np.stack(heads, axis=1), np.stack(mass, axis=1), errors
