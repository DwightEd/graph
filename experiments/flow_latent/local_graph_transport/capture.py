"""Same-position native/source-blocked states and full-coordinate local edges.

One chosen native layer is archived; this is not an all-layer collector. The
T+1 rows include both the prompt-end predictor and every answer post-token row.
"""
from contextlib import contextmanager, nullcontext

import numpy as np
import torch

from experiments.flow_latent.provenance_joint_state.measure import block_source_reads
from experiments.token_backtrace.grounded_projection import native_layout, rotate
from .messages import partition_senders


SITES = ('residual', 'attention_write', 'mlp_write')
GROUPS = ('source', 'local', 'remote', 'self', 'other_prompt')


class LayerCaptured(Exception):
    """Expected early exit immediately after the selected native block."""


@contextmanager
def capture_selected_layer(model, index, start, stop_after_layer=False):
    """Reuse native module hooks and always remove them after an early exit."""
    layer = model.model.layers[index]
    record = {}
    handles = []

    def save_input(name):
        def observe(module, args):
            record[name] = args[0][0, start:].detach().float().cpu()
        return observe

    def save_output(name, full=False):
        def observe(module, args, output):
            region = output[0] if full else output[0, start:]
            record[name] = region.detach().float().cpu()
        return observe

    def finish(module, args, output):
        record['after'] = output[0, start:].detach().float().cpu()
        if stop_after_layer:
            raise LayerCaptured()

    handles.append(layer.register_forward_pre_hook(save_input('residual')))
    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(save_input('head')))
    handles.append(layer.self_attn.o_proj.register_forward_hook(save_output('attention_write')))
    handles.append(layer.mlp.register_forward_hook(save_output('mlp_write')))
    for name, module in [('query', layer.self_attn.q_proj),
                         ('key', layer.self_attn.k_proj), ('value', layer.self_attn.v_proj)]:
        handles.append(module.register_forward_hook(save_output(name, full=name != 'query')))
    handles.append(layer.register_forward_hook(finish))
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


def allowed_keys(source, positions, blocked):
    """Exact causal mask; blocked source receivers retain their own source reads."""
    senders = torch.arange(len(source), device=positions.device)
    allowed = senders[None] <= positions[:, None]
    if blocked:
        allowed &= source[positions, None] | ~source[None]
    return allowed


def attention_from_projections(adapter, index, record, positions, cosine, sine,
                               layout, source, blocked):
    """Reconstruct actual native RoPE/GQA attention, including the blocked mask."""
    heads, _, width = adapter.head_layout(index)
    query = record['query'].to(positions.device).reshape(-1, heads, width).transpose(0, 1)
    query = rotate(query, cosine[positions], sine[positions], adapter.implementation)
    key, value = layout
    scores = query @ key.transpose(-1, -2)
    scores *= adapter.layers[index].self_attn.scaling
    allowed = allowed_keys(source, positions, blocked)
    weights = scores.masked_fill(~allowed[None], -torch.inf).softmax(-1)
    return weights.permute(1, 0, 2), value, allowed


def local_key_positions(positions, prompt_length, width):
    """Strict earlier answer nodes; self is not a local edge."""
    neighbors = positions[:, None] - torch.arange(1, width + 1, device=positions.device)
    return neighbors, neighbors >= prompt_length


def message_groups(attention, values, source, positions, prompt_length, width):
    """Exact AV sums, retaining head identity and all head coordinates."""
    groups = partition_senders(source, prompt_length, positions, width)
    messages = []
    masses = []
    for name in GROUPS:
        selected = attention * groups[name][:, None]
        messages.append(torch.einsum('rhs,shd->rhd', selected, values))
        masses.append(selected.sum(-1))
    return torch.stack(messages, dim=1), torch.stack(masses, dim=1)


@torch.no_grad()
def prediction_facts(model, hidden, targets):
    """Full-vocabulary uncertainty is about next-token outputs, not sources."""
    actual_logp = []
    entropy = []
    for begin in range(0, len(targets), 16):
        end = begin + 16
        logits = model.lm_head(hidden[begin:end].to(model.device)).float()
        logp = logits.log_softmax(-1)
        token = targets[begin:end].to(model.device)
        actual_logp.append(logp.gather(1, token[:, None])[:, 0].cpu())
        entropy.append(-(logp.exp() * logp).sum(-1).cpu())
    return dict(actual_logp=torch.cat(actual_logp).numpy(), entropy=torch.cat(entropy).numpy())


@torch.no_grad()
def capture_world(adapter, tokens, prompt_length, answer, source_mask, index,
                  blocked, stop_after_layer):
    """Run one world; early exit skips higher layers without changing this layer."""
    model = adapter.native
    ids = adapter.input_ids(tokens)
    block = block_source_reads(model, source_mask, len(tokens)) if blocked else nullcontext()
    hidden = None
    with block:
        with capture_selected_layer(model, index, prompt_length - 1, stop_after_layer) as record:
            try:
                hidden = model.model(ids, use_cache=False).last_hidden_state[0].cpu()
            except LayerCaptured:
                if not stop_after_layer:
                    raise
    facts = {}
    if hidden is not None:
        targets = torch.tensor(answer, dtype=torch.long)
        facts = prediction_facts(model, hidden[prompt_length - 1:-1], targets)
    error = ((record['residual'] + record['attention_write']) + record['mlp_write'] -
             record['after']).abs().max().item()
    return record, facts, dict(state_equation_max_abs=error,
        executed_layers=index + 1 if stop_after_layer else len(adapter.layers),
        full_model_completed=hidden is not None)


@torch.no_grad()
def graph_arrays(adapter, index, record, positions, cosine, sine, source,
                 prompt_length, width, blocked, chunk):
    """Process only one attention chunk; never materialize full edge D writes."""
    heads, kv_heads, head_dim = adapter.head_layout(index)
    layout = native_layout(adapter, index, record, cosine, sine)
    local_positions, local_valid = local_key_positions(positions, prompt_length, width)
    local_attention, group_heads, group_mass = [], [], []
    head_errors, write_errors, group_errors, masked_mass = [], [], [], []
    for begin in range(0, len(positions), chunk):
        end = begin + chunk
        selected = dict(record, query=record['query'][begin:end])
        rows = positions[begin:end]
        attention, values, allowed = attention_from_projections(
            adapter, index, selected, rows, cosine, sine, layout, source, blocked)
        complete = torch.einsum('rhs,shd->rhd', attention, values)
        head_errors.append((complete.cpu().flatten(1) - record['head'][begin:end]).abs().max().item())
        write = adapter.layers[index].self_attn.o_proj(complete.flatten(1))
        write_errors.append((write.cpu() - record['attention_write'][begin:end]).abs().max().item())
        messages, masses = message_groups(attention, values, source, rows, prompt_length, width)
        group_errors.append((messages.sum(1) - complete).abs().max().item())
        keys = local_positions[begin:end].clamp_min(0)
        selected_attention = attention.gather(2, keys[:, None].expand(-1, heads, -1))
        selected_attention *= local_valid[begin:end, None]
        local_attention.append(selected_attention.cpu().numpy().astype(np.float16))
        group_heads.append(messages.cpu().numpy().astype(np.float16))
        group_mass.append(masses.cpu().numpy().astype(np.float32))
        masked_mass.append(attention.masked_select(~allowed[:, None].expand_as(attention)).abs().max().item()
                           if (~allowed).any() else 0.)
    arrays = dict(nodes=np.stack([record[name].numpy() for name in SITES], axis=1).astype(np.float16),
        query=record['query'].reshape(-1, heads, head_dim).numpy().astype(np.float16),
        key=record['key'].reshape(-1, kv_heads, head_dim).numpy().astype(np.float16),
        value=record['value'].reshape(-1, kv_heads, head_dim).numpy().astype(np.float16),
        local_attention=np.concatenate(local_attention), group_heads=np.concatenate(group_heads),
        group_mass=np.concatenate(group_mass))
    audit = dict(head_reconstruction_max_abs=max(head_errors),
        write_reconstruction_max_abs=max(write_errors), group_sum_max_abs=max(group_errors),
        forbidden_attention_max_abs=max(masked_mass))
    return arrays, audit


@torch.no_grad()
def capture_answer(adapter, prompt_ids, answer_ids, source_mask, layer=15,
                   local_width=8, paired=True, query_chunk=64, stop_after_layer=False):
    """Return FP16 coordinate archives and FP32 native canaries; no labels used.

    nodes[:, :-1] align to prechoice y_t; nodes[:, 1:] align to post-token y_t.
    The blocked world preserves IDs/positions and source queries at every layer.
    """
    model = adapter.native
    tokens = prompt_ids + answer_ids
    ids = adapter.input_ids(tokens)
    positions = torch.arange(len(tokens), device=model.device)
    cosine, sine = model.model.rotary_emb(model.model.embed_tokens(ids), positions[None])
    source = torch.tensor(source_mask + [False] * len(answer_ids), device=model.device, dtype=torch.bool)
    selected_positions = positions[len(prompt_ids) - 1:]
    worlds, audits = [], []
    for blocked in ([False, True] if paired else [False]):
        record, facts, state_audit = capture_world(adapter, tokens, len(prompt_ids), answer_ids,
                                                  source_mask, layer, blocked, stop_after_layer)
        arrays, graph_audit = graph_arrays(adapter, layer, record, selected_positions,
            cosine[0], sine[0], source, len(prompt_ids), local_width, blocked, query_chunk)
        worlds.append(dict(arrays, **facts))
        audits.append(dict(blocked=blocked, **state_audit, **graph_audit))
        del record
    arrays = {name: np.stack([world[name] for world in worlds]) for name in worlds[0]}
    local_positions, local_valid = local_key_positions(selected_positions, len(prompt_ids), local_width)
    arrays.update(token_ids=np.asarray(tokens, dtype=np.int64),
        answer_ids=np.asarray(answer_ids, dtype=np.int64), source_mask=source.cpu().numpy(),
        query_positions=selected_positions.cpu().numpy(),
        local_positions=local_positions.cpu().numpy(), local_valid=local_valid.cpu().numpy())
    audit = dict(layer=layer, local_width=local_width, input_tokens=len(tokens),
        prompt_length=len(prompt_ids), answer_tokens=len(answer_ids), node_rows=len(selected_positions),
        sites=SITES, groups=GROUPS, worlds=audits, archival_dtype='float16 full coordinates',
        timing='row[:-1] prechoice P+t-1; row[1:] post-token P+t',
        stop_after_layer=stop_after_layer, full_model_forwards=0 if stop_after_layer else len(worlds))
    return arrays, audit
