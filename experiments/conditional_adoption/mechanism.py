"""Signed native candidate responses with fixed, previously computed past KV.

Every physical head remains separate. Source, local history, remote history,
self and other-prompt directions partition the current attention message.
These are influence measurements, not factual-support or hallucination scores.
"""
from contextlib import ExitStack, contextmanager

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from experiments.token_backtrace.messages import capture_messages, patch_message


GROUPS = ('source', 'local_history', 'remote_history', 'self', 'other_prompt')
OBJECTIVES = ('actual_logp', 'actual_minus_native_rival')


@torch.no_grad()
def prefill_past(model, tokens, chunk_size=128):
    """Compute strictly past KV without retaining past autograd or large logits."""
    cache = None
    for start in range(0, len(tokens), chunk_size):
        ids = torch.tensor([tokens[start:start + chunk_size]], device=model.device)
        output = model.model(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
    return cache


def copy_past(cache):
    """A query gets its own cache object; appending never changes another query."""
    return DynamicCache((layer.keys.detach(), layer.values.detach()) for layer in cache.layers)


@contextmanager
def capture_query(model):
    """Keep native current-query head/FFN tensors live and detached Q for readout."""
    queries = [None] * len(model.model.layers)
    rotary = [None] * len(model.model.layers)
    ffn_writes = [None] * len(model.model.layers)
    handles = []

    def query_hook(index):
        def observe(module, inputs, output):
            queries[index] = output.detach()
        return observe

    def rotary_hook(index):
        def observe(module, inputs, kwargs):
            rotary[index] = kwargs['position_embeddings']
        return observe

    def ffn_hook(index):
        def observe(module, inputs, output):
            ffn_writes[index] = output
        return observe

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(query_hook(index)))
        handles.append(layer.self_attn.register_forward_pre_hook(rotary_hook(index), with_kwargs=True))
        handles.append(layer.mlp.register_forward_hook(ffn_hook(index)))
    try:
        with capture_messages(model) as captured:
            yield queries, rotary, ffn_writes, captured
    finally:
        for handle in handles:
            handle.remove()


def channel_masks(source_mask, prompt_length, query_position, local_window, device):
    """Local is a diagnostic distance band, never an inferred semantic boundary."""
    positions = torch.arange(query_position + 1, device=device)
    source = torch.zeros(query_position + 1, device=device, dtype=torch.bool)
    source[:prompt_length] = torch.tensor(source_mask, device=device)
    self_key = positions == query_position
    history = (positions >= prompt_length) & ~self_key
    local = history & (positions >= query_position - local_window)
    remote = history & ~local
    source &= ~self_key
    other = (positions < prompt_length) & ~source & ~self_key
    return torch.stack((source, local, remote, self_key, other))


@torch.no_grad()
def output_grams(model):
    """One model-wide CPU cache of physical-head WO Gram matrices."""
    matrices = []
    for layer in model.model.layers:
        attention = layer.self_attn
        heads = attention.q_proj.out_features // attention.head_dim
        weight = attention.o_proj.weight.detach().float().reshape(-1, heads, attention.head_dim)
        weight = weight.permute(1, 0, 2)
        matrices.append((weight.transpose(1, 2) @ weight).cpu())
    return torch.stack(matrices)


@torch.no_grad()
def reconstruct_directions(model, queries, rotary, cache, masks, native_messages, gram_cache=None):
    """Return real AV [L,G,H,d], projected norms and reconstruction error."""
    factors, norms, masses, grams, errors = [], [], [], [], []
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        heads = attention.q_proj.out_features // attention.head_dim
        query = queries[index].view(1, 1, heads, attention.head_dim).transpose(1, 2)
        cosine, sine = rotary[index]
        query, _ = apply_rotary_pos_emb(query, query, cosine, sine)
        keys = repeat_kv(cache.layers[index].keys, attention.num_key_value_groups)
        values = repeat_kv(cache.layers[index].values, attention.num_key_value_groups)[0].float()
        weights = (query[0].float() @ keys[0].float().transpose(-1, -2) * attention.scaling).softmax(-1)[:, 0]
        grouped = torch.einsum('hk,gk,hkd->ghd', weights, masks.float(), values)
        if gram_cache is None:
            weight = attention.o_proj.weight.detach().float().reshape(-1, heads, attention.head_dim)
            weight = weight.permute(1, 0, 2)
            gram = weight.transpose(1, 2) @ weight
        else:
            gram = gram_cache[index].to(model.device)
        squared = torch.einsum('ghd,hde,ghe->gh', grouped, gram, grouped)
        expected = native_messages[index][0, 0].float().reshape(heads, attention.head_dim)
        errors.append((grouped.sum(0) - expected).abs().max().cpu())
        factors.append(grouped.cpu())
        norms.append(squared.clamp_min(0).sqrt().cpu())
        masses.append(torch.einsum('hk,gk->gh', weights, masks.float()).cpu())
        if gram_cache is None:
            grams.append(gram.cpu())
    return dict(messages=torch.stack(factors), projected_norm=torch.stack(norms),
                attention_mass=torch.stack(masses),
                output_gram=torch.stack(grams) if gram_cache is None else gram_cache,
                reconstruction_error=torch.stack(errors))


def candidate_objectives(logits, actual_id):
    """The competitor is native best nonactual, without a gold repair candidate."""
    competitors = logits.detach().clone()
    competitors[actual_id] = -torch.inf
    rival = int(competitors.argmax())
    return (logits.log_softmax(-1)[actual_id], logits[actual_id] - logits[rival]), rival


def signed_responses(gradients, messages):
    """[objective,L,H,d] dot [L,G,H,d] -> [objective,L,G,H]."""
    return torch.einsum('olhd,lghd->olgh', gradients, messages)


def measure_query(model, past, query_token, actual_id, source_mask, prompt_length, local_window=16,
                  gram_cache=None):
    """Suffix gradients include later-layer current Q/K, RMS and native FFNs.

    Past KV is fixed and detached. This cannot identify which past source event
    created a historical carrier, and is not a full history mediation estimate.
    Q/K forming the edited head itself precede the injection and stay fixed.
    """
    cache = copy_past(past)
    position = cache.get_seq_length()
    ids = torch.tensor([[query_token]], device=model.device)
    with torch.enable_grad(), capture_query(model) as captured:
        embedding = model.model.embed_tokens(ids).detach().requires_grad_(True)
        output = model.model(inputs_embeds=embedding, past_key_values=cache, use_cache=True)
        logits = model.lm_head(output.last_hidden_state[0, 0]).float()
        queries, rotary, ffn_writes, (messages, _) = captured
        objectives, rival = candidate_objectives(logits, actual_id)
        masks = channel_masks(source_mask, prompt_length, position, local_window, model.device)
        directions = reconstruct_directions(model, queries, rotary, cache, masks, messages, gram_cache)
        total_gradients, residual_gradients = [], []
        for objective in objectives:
            gradients = torch.autograd.grad(objective, tuple(messages) + tuple(ffn_writes), retain_graph=True)
            total_gradients.append(torch.stack([value[0, 0].float().reshape(-1, model.config.head_dim).cpu()
                                               for value in gradients[:len(messages)]]))
            direct = [value[0, 0].float() @ layer.self_attn.o_proj.weight.detach().float()
                      for value, layer in zip(gradients[len(messages):], model.model.layers)]
            residual_gradients.append(torch.stack([value.reshape(-1, model.config.head_dim).detach().cpu()
                                                  for value in direct]))
    total = torch.stack(total_gradients)
    residual = torch.stack(residual_gradients)
    logp = logits.detach().log_softmax(-1).cpu()
    return dict(**directions, gradient=total, gradient_residual=residual,
                response=signed_responses(total, directions['messages']),
                response_residual=signed_responses(residual, directions['messages']),
                response_ffn=signed_responses(total - residual, directions['messages']),
                logp_full=logp, actual_id=actual_id, rival_id=rival, query_position=position,
                actual_logp=float(logp[actual_id]), margin=float(objectives[1].detach()),
                entropy=float(-(logp.exp() * logp).sum()),
                gradients_scope='current query with fixed detached native past KV')


def norm_preserving_control(messages, gram, seed=37):
    """Random coordinate signs, rescaled to preserve each physical WO norm."""
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, messages.shape, generator=generator).mul(2).sub(1)
    changed = messages * signs
    original_norm = torch.einsum('lghd,lhde,lghe->lgh', messages, gram, messages).clamp_min(0).sqrt()
    changed_norm = torch.einsum('lghd,lhde,lghe->lgh', changed, gram, changed).clamp_min(0).sqrt()
    return changed * (original_norm / changed_norm.clamp_min(1e-30))[..., None]


@torch.no_grad()
def finite_query(model, past, query_token, actual_id, rival_id, layer, head, direction, dose):
    """Add a small signed vector at one whole head; recompute the native suffix."""
    ids = torch.tensor([[query_token]], device=model.device)
    with patch_message(model, layer, head, receiver=0, delta=direction, alpha=dose):
        output = model.model(input_ids=ids, past_key_values=copy_past(past), use_cache=True)
        logits = model.lm_head(output.last_hidden_state[0, 0]).float()
    return torch.stack((logits.log_softmax(-1)[actual_id], logits[actual_id] - logits[rival_id])).cpu()


@contextmanager
def patch_all_heads(model, directions, dose):
    """Joint frozen-direction intervention across current physical layer/heads."""
    def add_direction(direction):
        def replace(module, inputs):
            changed = inputs[0] + dose * direction.flatten().to(inputs[0])[None, None]
            return (changed, *inputs[1:])
        return replace

    with ExitStack() as stack:
        for layer, direction in zip(model.model.layers, directions):
            handle = layer.self_attn.o_proj.register_forward_pre_hook(add_direction(direction))
            stack.callback(handle.remove)
        yield


@torch.no_grad()
def finite_query_group(model, past, query_token, actual_id, rival_id, directions, dose):
    """Finite counterpart to sum of current native site directional responses."""
    ids = torch.tensor([[query_token]], device=model.device)
    with patch_all_heads(model, directions, dose):
        output = model.model(input_ids=ids, past_key_values=copy_past(past), use_cache=True)
        logits = model.lm_head(output.last_hidden_state[0, 0]).float()
    return torch.stack((logits.log_softmax(-1)[actual_id], logits[actual_id] - logits[rival_id])).cpu()
