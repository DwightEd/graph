"""Source-conditioned full-vector history transport and independent-query replay.

Source projection is an experimental grounding operator, not a certified semantic
binding correction. No entity annotations or hallucination labels enter it.
"""
from contextlib import contextmanager

import torch
from torch.nn.functional import normalize


@contextmanager
def capture_projection_inputs(model, prompt):
    """CPU native tensors; source/past KV stays fixed during each query replay."""
    records = [{} for _ in model.model.layers]
    handles = []

    def remember(layer, name, region):
        def observe(module, inputs, output=None):
            value = inputs[0] if output is None else output
            records[layer][name] = value[0, region].detach().float().cpu()
        return observe

    for index, layer in enumerate(model.model.layers):
        region = slice(prompt - 1, None)
        for name, module in (("query", layer.self_attn.q_proj), ("key", layer.self_attn.k_proj),
                             ("value", layer.self_attn.v_proj)):
            kept = region if name == "query" else slice(None)
            handles.append(module.register_forward_hook(remember(index, name, kept)))
        handles.append(layer.register_forward_pre_hook(remember(index, "residual", region)))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(remember(index, "head", region)))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def rotate(values, cosine, sine, implementation):
    """Values [heads, positions, D]; native RoPE with original absolute positions."""
    return values * cosine[None] + implementation.rotate_half(values) * sine[None]


def native_layout(adapter, index, record, cosine, sine):
    heads, kv_heads, width = adapter.head_layout(index)
    device = adapter.native.device
    key = record["key"].to(device).reshape(-1, kv_heads, width).transpose(0, 1)
    key = rotate(key, cosine, sine, adapter.implementation).repeat_interleave(heads // kv_heads, 0)
    value = record["value"].to(device).reshape(-1, kv_heads, width)
    value = value.repeat_interleave(heads // kv_heads, 1)
    return key, value


def observed_attention(adapter, index, record, positions, cosine, sine):
    heads, _, width = adapter.head_layout(index)
    query = record["query"].to(adapter.native.device).reshape(-1, heads, width).transpose(0, 1)
    query = rotate(query, cosine[positions], sine[positions], adapter.implementation)
    key, value = native_layout(adapter, index, record, cosine, sine)
    scores = query @ key.transpose(-1, -2) * adapter.layers[index].self_attn.scaling
    future = torch.arange(value.shape[0], device=positions.device)[None] > positions[:, None]
    scores.masked_fill_(future[None], -torch.inf)
    attention = scores.softmax(-1).permute(1, 0, 2)
    return attention, value


def content_candidates(model, tokens, prompt, source_mask, nearest=16):
    """Exact token matches when available; otherwise static-embedding neighbors.

    This controls surface content without imposing externally labelled roles.
    No future answer token is consulted for a current token's candidate source.
    """
    device = model.device
    source = torch.where(torch.tensor(source_mask, device=device))[0]
    ids = torch.tensor(tokens, device=device)
    answer_ids = ids[prompt:]
    embedding = model.model.embed_tokens.weight
    source_embedding = normalize(embedding[ids[source]].float(), dim=-1)
    answer_embedding = normalize(embedding[answer_ids].float(), dim=-1)
    similarity = answer_embedding @ source_embedding.T
    exact = answer_ids[:, None] == ids[source][None]
    top = similarity.topk(min(nearest, len(source)), dim=-1).indices
    selected = torch.zeros_like(exact).scatter_(1, top, True)
    selected = torch.where(exact.any(-1)[:, None], exact, selected)
    kernel = ((similarity - similarity.max(-1, keepdim=True).values) / .1).exp()
    kernel *= selected
    return source, kernel, similarity.max(-1).values


def projected_carriers(attention, values, source, kernel, prompt):
    """Project post-token V into source V; graph/flat retain all H×128 coordinates."""
    source_attention = attention[1:, :, source]
    weights = source_attention * kernel[:, None]
    weights /= weights.sum(-1, keepdim=True).clamp_min(1e-30)
    projected = torch.einsum("ths,shd->thd", weights, values[source])
    flat_weights = kernel / kernel.sum(-1, keepdim=True).clamp_min(1e-30)
    flat = torch.einsum("ts,shd->thd", flat_weights, values[source])
    original = values[prompt:]
    # Equal V norm avoids a count-dependent shrinking of the historical message.
    norm = original.norm(dim=-1, keepdim=True)
    projected = normalize(projected, dim=-1) * norm
    flat = normalize(flat, dim=-1) * norm
    confidence = source_attention.sum(-1).clamp(0, 1)[..., None]
    return confidence * (projected - original), confidence * (flat - original), projected


def history_transport(attention, displacement, prompt, rewired=False):
    """Predictor q=P+t−1 reads only previously input answer nodes k<t."""
    weights = attention[:-1, :, prompt:]
    count = displacement.shape[0]
    if not rewired:
        return torch.einsum("thk,khd->thd", weights, displacement)
    transported = []
    generator = torch.Generator(device=weights.device).manual_seed(73)
    for target in range(count):
        mapping = torch.arange(count, device=weights.device)
        distance = target - torch.arange(target, device=weights.device)
        for lower, upper in ((2, 4), (4, 8), (8, 16), (16, 32), (32, count + 1)):
            indices = torch.where((distance >= lower) & (distance < upper))[0]
            mapping[indices] = indices[torch.randperm(len(indices), generator=generator, device=weights.device)]
        transported.append(torch.einsum("hk,khd->hd", weights[target], displacement[mapping]))
    return torch.stack(transported)


def layer_finish(layer, residual, messages):
    hidden = residual + layer.self_attn.o_proj(messages.flatten(1))
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


def replay_attention(adapter, index, record, hidden, positions, cosine, sine):
    """Recompute current Q/K/V; replace only self key/value, keep every earlier KV."""
    layer = adapter.layers[index]
    heads, kv_heads, width = adapter.head_layout(index)
    normalized = layer.input_layernorm(hidden)
    query = layer.self_attn.q_proj(normalized).reshape(-1, heads, width).transpose(0, 1)
    self_key = layer.self_attn.k_proj(normalized).reshape(-1, kv_heads, width).transpose(0, 1)
    query = rotate(query, cosine[positions], sine[positions], adapter.implementation)
    self_key = rotate(self_key, cosine[positions], sine[positions], adapter.implementation)
    self_key = self_key.repeat_interleave(heads // kv_heads, 0)
    key, value = native_layout(adapter, index, record, cosine, sine)
    scores = query @ key.transpose(-1, -2)
    rows = torch.arange(len(positions), device=positions.device)
    scores[:, rows, positions] = (query * self_key).sum(-1)
    scores *= layer.self_attn.scaling
    future = torch.arange(value.shape[0], device=positions.device)[None] > positions[:, None]
    scores.masked_fill_(future[None], -torch.inf)
    weights = scores.softmax(-1)
    messages = torch.einsum("htk,khd->thd", weights, value)
    self_value = layer.self_attn.v_proj(normalized).reshape(-1, kv_heads, width)
    self_value = self_value.repeat_interleave(heads // kv_heads, 1)
    self_weight = weights[:, rows, positions].T[..., None]
    return messages + self_weight * (self_value - value[positions])


@torch.no_grad()
def replay_queries(adapter, records, start_layer, rows, positions, delta, cosine, sine, dose=1.):
    """Batch independent causal query worlds; there is no shared edited history."""
    record = records[start_layer]
    device = adapter.native.device
    heads, _, width = adapter.head_layout(start_layer)
    residual = record["residual"][rows.cpu()].to(device)
    native_head = record["head"][rows.cpu()].to(device).reshape(-1, heads, width)
    hidden = layer_finish(adapter.layers[start_layer], residual, native_head + dose * delta)
    for index in range(start_layer + 1, len(records)):
        messages = replay_attention(adapter, index, records[index], hidden, positions, cosine, sine)
        hidden = layer_finish(adapter.layers[index], hidden, messages)
    return adapter.native.model.norm(hidden)


@torch.no_grad()
def token_logp(model, hidden, token_ids):
    values = []
    for begin in range(0, len(hidden), 16):
        logits = model.lm_head(hidden[begin:begin + 16]).float()
        selected = token_ids[begin:begin + 16, None]
        values.append((logits.gather(1, selected)[:, 0] - logits.logsumexp(-1)).cpu())
    return torch.cat(values)
