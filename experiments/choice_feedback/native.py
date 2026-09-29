"""One source event, followed by native, sequential KV updates."""
from contextlib import contextmanager
from types import MethodType

import numpy as np
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def event_attention(probe, keys, dose):
    """Support one unpadded sequence, both full prefix and cached decoding."""
    def forward(module, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, module.head_dim)
        query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, module.layer_idx,
                dict(cache_position=cache_position, cos=position_embeddings[0], sin=position_embeddings[1]))
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        logits = query @ key.transpose(-1, -2) * module.scaling
        positions = torch.arange(key.shape[-2], device=query.device)
        logits.masked_fill_(positions[None, :] > cache_position[:, None], -torch.inf)
        rows = torch.where(cache_position == probe['receiver'])[0]
        if len(rows):
            logits[0, probe['head'], rows.item(), keys] += dose
        weights = logits.softmax(-1)
        if len(rows):
            forward.mass = float(weights[0, probe['head'], rows.item(), keys].sum())
        output = (weights @ value).transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return module.o_proj(output), None
    return forward


@contextmanager
def intervene(model, probe, keys, dose):
    module = model.model.layers[probe['layer']].self_attn
    original = module.forward
    patched = event_attention(probe, keys, dose)
    module.forward = MethodType(patched, module)
    try:
        yield patched
    finally:
        module.forward = original


def newest_kv(cache):
    """Return [layer, K/V, KV-head, head-dim], preserving all GQA heads."""
    return torch.stack([torch.stack((layer.keys[0, :, -1], layer.values[0, :, -1]))
                        for layer in cache.layers]).float().cpu().numpy()


@torch.no_grad()
def stream(model, prefix, continuation, candidates, probe, keys, dose,
           greedy=False, first_token=None, eos_ids=()):
    """The prefix ends at receiver; the next prediction is target q, not q+1."""
    inputs = torch.tensor(prefix, device=model.device)[None]
    cache = None
    margins, logps, hidden, kv, emitted = [], [], [], [], []
    with intervene(model, probe, keys, dose) as patch:
        for step, observed in enumerate(continuation):
            output = model.model(input_ids=inputs, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            state = output.last_hidden_state[0, -1]
            logits = model.lm_head(state).float()
            token = int(logits.argmax()) if greedy else observed
            if step == 0 and first_token is not None:
                token = first_token
            margins.append(float(logits[candidates[0]] - logits[candidates[1]]))
            logps.append(float(logits[token] - logits.logsumexp(-1)))
            hidden.append(state.cpu().numpy())
            kv.append(newest_kv(cache))
            emitted.append(token)
            inputs = torch.tensor([[token]], device=model.device)
            if greedy and token in eos_ids:
                break
        mass = patch.mass
    return dict(margin=np.asarray(margins), logp=np.asarray(logps),
                hidden=np.stack(hidden), kv=np.stack(kv), tokens=np.asarray(emitted), mass=mass)


@torch.no_grad()
def full_margin(model, tokens, start, candidates, probe, keys, dose):
    with intervene(model, probe, keys, dose):
        hidden = model.model(input_ids=torch.tensor(tokens, device=model.device)[None],
                             use_cache=False).last_hidden_state[0, start:]
        weight = model.lm_head.weight[candidates].float()
        logits = hidden.float() @ weight.T
    return (logits[:, 0] - logits[:, 1]).cpu().numpy()
