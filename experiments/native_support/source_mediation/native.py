"""Position-preserving prompt x history-KV intervention, batched over target queries.

Each query reads donor past K/V and its own recomputed self K/V. It must never
read another query's hybrid state: that would define a different intervention.
"""

from contextlib import ExitStack, contextmanager
from types import MethodType
from unittest.mock import patch

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from state_audit.model.replay import attention_backend


@torch.no_grad()
def donor_world(model, prefix, answer, source_mask, source_on, chunk_size):
    tokens = prefix + answer[:-1]
    allowed = torch.ones((1, len(tokens)), device=model.native.device, dtype=torch.long)
    if not source_on:
        allowed[0, :len(prefix)] = ~torch.as_tensor(source_mask, device=allowed.device)
    cache, values = DynamicCache(), []
    with attention_backend(model, "eager"):
        for start in range(0, len(tokens), chunk_size):
            stop = min(start + chunk_size, len(tokens))
            hidden = model.native.model(input_ids=model.input_ids(tokens[start:stop]),
                attention_mask=allowed[:, :stop], past_key_values=cache,
                use_cache=True, return_dict=True).last_hidden_state[0]
            first = max(start, len(prefix) - 1)
            if first < stop:
                logits = model.native.lm_head(hidden[first-start:]).float().log_softmax(-1)
                ids = model.input_ids(answer[first-len(prefix)+1:stop-len(prefix)+1])[0]
                values.append(logits.gather(-1, ids[:, None])[:, 0].cpu().numpy())
    # CPU donors bound GPU memory to one original cache or one replay layer.
    states = [(layer.keys[0].cpu(), layer.values[0].cpu()) for layer in cache.layers]
    return states, np.concatenate(values)


def donor_keys_values(states, layer, prompt, stop, prompt_world, history_world, device):
    result = []
    for field in (0, 1):
        prompt_values = states[prompt_world][layer][field][:, :min(prompt, stop)]
        history_values = states[history_world][layer][field][:, prompt:stop]
        result.append(torch.cat([prompt_values, history_values], dim=1).to(device))
    return result


def visible_keys(model, layer, positions, count, source_mask, source_on):
    keys = torch.arange(count, device=positions.device)
    visible = keys[None, :] < positions[:, None]
    family = model.native.config.model_type
    if family == "mistral":
        window = model.native.config.sliding_window
    elif family == "qwen2":
        window = model.layers[layer].self_attn.sliding_window
    else:
        window = None
    if window is not None:
        visible &= keys[None, :] > positions[:, None] - window
    if not source_on:
        blocked = torch.as_tensor(source_mask[:count], device=positions.device)
        visible[:, :len(blocked)] &= ~blocked
    return visible


def paired_attention(model, layer, states, prompt, source_mask, worlds, positions, observed):
    prompt_world, history_world = worlds
    implementation = model.implementation

    def forward(module, hidden_states, position_embeddings, attention_mask, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, module.head_dim)
        query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = implementation.apply_rotary_pos_emb(query, key, *position_embeddings)
        past_key, past_value = donor_keys_values(states, layer, prompt, int(positions[-1]),
                                                prompt_world, history_world, query.device)
        past_key = implementation.repeat_kv(past_key[None], module.num_key_value_groups)
        past_value = implementation.repeat_kv(past_value[None], module.num_key_value_groups)
        key = implementation.repeat_kv(key, module.num_key_value_groups)
        value = implementation.repeat_kv(value, module.num_key_value_groups)
        visible = visible_keys(model, layer, positions, past_key.shape[-2], source_mask, prompt_world)
        head_values = independent_queries(query, key, value, past_key, past_value, visible, module.scaling)
        if observed is not None:
            observed[layer] = head_values[0].transpose(0, 1).float().cpu().numpy()
        output = head_values.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return module.o_proj(output), None

    return forward


def independent_queries(query, key, value, past_key, past_value, visible, scaling):
    """[batch,head,query,width]; last score column is each query's own self edge."""
    past_logits = (query @ past_key.transpose(-1, -2)) * scaling
    past_logits = past_logits.masked_fill(~visible, float("-inf"))
    self_logits = (query * key).sum(-1, keepdim=True) * scaling
    weights = torch.cat([past_logits, self_logits], dim=-1).float().softmax(-1).to(query.dtype)
    return weights[..., :-1] @ past_value + weights[..., -1:] * value


@contextmanager
def paired_hooks(model, states, prompt, source_mask, worlds, positions, observed):
    with ExitStack() as stack:
        for layer, block in enumerate(model.layers):
            forward = paired_attention(model, layer, states, prompt, source_mask, worlds, positions, observed)
            stack.enter_context(patch.object(block.self_attn, "forward", MethodType(forward, block.self_attn)))
        yield


@torch.no_grad()
def replay_world(model, states, prefix, answer, source_mask, worlds, query_chunk, save_heads=False):
    tokens = prefix + answer[:-1]
    logp, heads = [], []
    with attention_backend(model, "eager"):
        for start in range(0, len(answer), query_chunk):
            stop = min(start + query_chunk, len(answer))
            positions = torch.arange(len(prefix)-1+start, len(prefix)-1+stop, device=model.native.device)
            observed = {} if save_heads else None
            with paired_hooks(model, states, len(prefix), source_mask, worlds, positions, observed):
                hidden = model.native.model(input_ids=model.input_ids(tokens[int(positions[0]):int(positions[-1])+1]),
                    position_ids=positions[None], use_cache=False, return_dict=True).last_hidden_state[0]
            logits = model.native.lm_head(hidden).float().log_softmax(-1)
            logp.append(logits.gather(-1, model.input_ids(answer[start:stop])[0, :, None])[:, 0].cpu().numpy())
            if save_heads:
                heads.append(np.stack([observed[layer] for layer in range(len(model.layers))], axis=1))
    return np.concatenate(logp), np.concatenate(heads) if save_heads else None


@torch.no_grad()
def measure(model, views, prefill_chunk, query_chunk, save_heads=False):
    prefix, answer = views["prompt_with_source"], views["answer_ids"]
    source_mask = np.zeros(len(prefix), dtype=bool)
    source_mask[views["removed_prompt_positions"]] = True
    if source_mask[0] or source_mask[-1]:
        raise ValueError("Mediation requires a non-source first token and final prompt query")
    if len(prefix) + len(answer) - 1 > model.native.config.max_position_embeddings:
        raise ValueError("Original input exceeds model context; no truncation")
    states, original = [], []
    for source_on in (0, 1):
        donor, logp = donor_world(model, prefix, answer, source_mask, source_on, prefill_chunk)
        states.append(donor)
        original.append(logp)
    worlds = np.empty((2, 2, len(answer)))
    heads = {}
    # Diagonal replay is an identity control, not skipped in the native experiment.
    for prompt_world in (0, 1):
        for history_world in (0, 1):
            values, readout = replay_world(model, states, prefix, answer, source_mask,
                (prompt_world, history_world), query_chunk, save_heads)
            worlds[prompt_world, history_world] = values
            if save_heads:
                heads[f"heads_{prompt_world}{history_world}"] = readout
    error = np.array([np.max(np.abs(worlds[index, index] - original[index])) for index in (0, 1)])
    return dict(worlds=worlds, native_logp=np.stack(original), identity_max_error=error,
                token_id=np.asarray(answer), target=np.arange(len(answer))), heads
