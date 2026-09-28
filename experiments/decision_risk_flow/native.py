"""Native downstream VJPs for independent current queries with fixed past K/V."""

from contextlib import ExitStack
from types import MethodType
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


@torch.no_grad()
def prefill(model, prompt, answer, checkpoints=(16, 24, 32), chunk=256):
    tokens = prompt + answer[:-1]
    saved = {index: [] for index in checkpoints}
    normalized = []
    handles = []
    begin = 0

    def hook(index):
        def save(module, args, output):
            state = output[0] if isinstance(output, tuple) else output
            first = max(0, len(prompt) - 1 - begin)
            saved[index].append(state[0, first:].detach())
        return save

    for index in checkpoints:
        handles.append(model.model.layers[index - 1].register_forward_hook(hook(index)))
    cache = DynamicCache()
    try:
        for begin in range(0, len(tokens), chunk):
            ids = torch.tensor([tokens[begin:begin + chunk]], device=model.device)
            result = model.model(ids, past_key_values=cache, use_cache=True)
            first = max(0, len(prompt) - 1 - begin)
            normalized.append(result.last_hidden_state[0, first:].detach())
    finally:
        for handle in handles:
            handle.remove()
    states = torch.stack([torch.cat(saved[index]) for index in checkpoints], dim=1)
    return cache, states, torch.cat(normalized)


def confidence(model, normalized, answer, prompt_length, special_ids):
    chunks, alternatives = [], []
    counts = {}
    previous_counts = []
    for token in answer:
        previous_counts.append(counts.get(token, 0))
        counts[token] = counts.get(token, 0) + 1
    with torch.no_grad():
        for begin in range(0, len(answer), 32):
            logits = model.lm_head(normalized[begin:begin + 32]).float()
            target = torch.tensor(answer[begin:begin + 32], device=logits.device)
            logp = logits.log_softmax(-1)
            top = logits.topk(2, dim=-1)
            alternative = torch.where(top.indices[:, 0] == target, top.indices[:, 1], top.indices[:, 0])
            margin = logits.gather(1, target[:, None])[:, 0] - logits.gather(1, alternative[:, None])[:, 0]
            entropy = -(logp.exp() * logp).sum(-1)
            surprise = -logp.gather(1, target[:, None])[:, 0]
            maximum = logp.max(-1).values.exp()
            chunks.append(torch.stack((entropy, surprise, margin, maximum), -1).cpu())
            alternatives.append(alternative.cpu())
    continuous = torch.cat(chunks)
    positions = torch.arange(len(answer)).float().log1p()
    length = torch.full_like(positions, float(prompt_length)).log1p()
    repetition = torch.tensor(previous_counts).float().log1p()
    special = torch.tensor([token in special_ids for token in answer]).float()
    return torch.cat((continuous, torch.stack((positions, length, repetition, special), -1)), -1), torch.cat(alternatives)


class QueryReplay:
    """Each query reads original past states and its own recomputed self state."""

    def __init__(self, model, cache, positions, checkpoints=(16, 24, 32), gate=None):
        self.model, self.cache, self.positions = model, cache, positions
        self.checkpoints, self.gate = checkpoints, gate
        self.writes, self.messages, self.states, self.residuals = [], [], {}, {}
        self.mlp_writes = []
        self.stack = ExitStack()

    def __enter__(self):
        for index, layer in enumerate(self.model.model.layers):
            self.stack.enter_context(patch.object(layer.self_attn, 'forward',
                MethodType(self.attention(index), layer.self_attn)))
            handle = layer.register_forward_pre_hook(self.save_residual(index))
            self.stack.callback(handle.remove)
            handle = layer.mlp.register_forward_hook(self.save_mlp)
            self.stack.callback(handle.remove)
            if index + 1 in self.checkpoints:
                handle = layer.register_forward_hook(self.save_state(index + 1))
                self.stack.callback(handle.remove)
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def save_residual(self, index):
        def hook(module, args):
            self.residuals[index] = args[0].detach()[0]
        return hook

    def save_state(self, index):
        def hook(module, args, output):
            state = output[0] if isinstance(output, tuple) else output
            self.states[index] = state[0]
        return hook

    def save_mlp(self, module, args, output):
        if self.gate is not None and 'scales' in self.gate:
            output = output * self.gate['scales'][2]
        self.mlp_writes.append(output)
        return output

    def attention(self, index):
        def forward(module, hidden_states, position_embeddings, attention_mask, **kwargs):
            shape = (*hidden_states.shape[:-1], -1, module.head_dim)
            query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
            key = module.k_proj(hidden_states).view(shape).transpose(1, 2)
            value = module.v_proj(hidden_states).view(shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            key, value = [repeat_kv(x, module.num_key_value_groups) for x in (key, value)]
            past_key = repeat_kv(self.cache.layers[index].keys, module.num_key_value_groups)
            past_value = repeat_kv(self.cache.layers[index].values, module.num_key_value_groups)
            visible = torch.arange(past_key.shape[-2], device=query.device)[None] < self.positions[:, None]
            # Match SDPA's FP32 accumulation rather than rounding logits and
            # probabilities to BF16 at every attention operation.
            compute_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
            query_compute = query.to(compute_dtype)
            scores = (query_compute @ past_key.to(compute_dtype).transpose(-1, -2)) * module.scaling
            scores = scores.masked_fill(~visible, -torch.inf)
            self_scores = (query_compute * key.to(compute_dtype)).sum(-1, keepdim=True) * module.scaling
            weights = torch.cat((scores, self_scores), -1).softmax(-1)
            if self.gate is not None and 'scales' in self.gate:
                scales = self.gate['scales']
                prompt = self.gate['prompt_length']
                key_positions = torch.arange(past_key.shape[-2], device=query.device)
                multiplier = torch.where(key_positions < prompt, scales[0], scales[1])
                multiplier = multiplier[None].expand(len(self.positions), -1)
                own = torch.where(self.positions < prompt, scales[0], scales[1])
                multiplier = torch.cat((multiplier, own[:, None]), -1)
                weights = weights * multiplier[None, None]
            elif self.gate is not None and index == self.gate['layer']:
                multiplier = torch.ones_like(weights)
                multiplier[:, self.gate['head'], :, self.gate['key']] = self.gate['scale']
                weights = weights * multiplier
            head = weights[..., :-1] @ past_value.to(compute_dtype) + weights[..., -1:] * value.to(compute_dtype)
            write = module.o_proj(head.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).to(hidden_states.dtype))
            self.writes.append(write)
            self.messages.append((weights.detach()[0], past_value.detach()[0], value.detach()[0]))
            return write, None
        return forward


def replay(model, cache, tokens, positions, checkpoints=(16, 24, 32), gate=None):
    embedded = model.model.embed_tokens(torch.tensor(tokens, device=model.device)[positions])
    embedded = embedded.detach().requires_grad_(True)[None]
    with QueryReplay(model, cache, positions, checkpoints, gate) as capture:
        final = model.model(inputs_embeds=embedded, position_ids=positions[None], use_cache=False).last_hidden_state[0]
    states = torch.stack([capture.states[index] for index in checkpoints], dim=1)
    return final, states, capture
