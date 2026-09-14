"""Read-only, bounded-memory Llama capture of actual local attention endpoints.

One teacher-forced backbone forward per response; no generation, intervention,
JVP, or labels. Q/K/V are observed through hooks. Reconstruct only response-query
blocks using the native RoPE and full causal softmax, then discard full rows.
SDPA runs the backbone; no all-layer/full-attention tensor is retained.
"""
import inspect
import numpy as np
import torch


@torch.inference_mode()
def attention_blocks(q, k, v, cosine, sine, context, *, start, source_mask,
                     response_special, window=16, chunk=16, scaling=None):
    """q/k/v: [1,S,heads,d]; context: native pre-WO concatenation [1,S,H*d]."""
    if q.shape[0] != 1 or q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('one unpadded sequence of Llama Q/K/V required')
    _, length, heads, dim = q.shape
    if k.shape[1] != length or k.shape[-1] != dim or heads % k.shape[2]:
        raise ValueError('GQA dimensions differ')
    if min(window, chunk) < 1 or not 0 <= start < length:
        raise ValueError('invalid query range or chunk')
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    def rotate(x):
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)
    q, k = q * cosine + rotate(q) * sine, k * cosine + rotate(k) * sine
    k = k.repeat_interleave(heads // k.shape[1], dim=1)
    v = v.repeat_interleave(heads // v.shape[1], dim=1)
    n, prompt = length - start, start + 1
    device = q.device
    source = torch.as_tensor(source_mask, device=device, dtype=torch.bool)
    if source.shape != (prompt,):
        raise ValueError('prompt-aligned source mask required')
    special = np.asarray(response_special, bool)
    if special.shape != (n,):
        raise ValueError('one special flag per output response token')
    local = np.zeros((n, heads, window), np.float32)
    masses = {name: np.zeros((n, heads), np.float32) for name in ('source_mass', 'history_mass', 'remote_mass')}
    pos = torch.arange(length, device=device)
    src = torch.zeros(length, dtype=torch.bool, device=device); src[:prompt] = source
    hist = pos >= prompt
    # ids[:-1] was input; the last response token has no key in this replay.
    if n > 1:
        hist[prompt:] &= ~torch.as_tensor(special[:-1], device=device)
    norm_error, norm_native, mass_error = 0., 0., 0.
    scale = dim ** -.5 if scaling is None else scaling
    for a in range(start, length, chunk):
        b = min(a + chunk, length)
        qs = torch.arange(a, b, device=device)
        logits = (q[:, :, a:b] @ k.transpose(-1, -2)) * scale
        logits = logits.masked_fill(pos[None, :] > qs[:, None], -torch.inf)
        weights = torch.softmax(logits.float(), dim=-1)[0]
        mass_error = max(mass_error, float((weights.sum(-1) - 1).abs().max()))
        target = slice(a - start, b - start)
        masses['source_mass'][target] = weights[:, :, src].sum(-1).T.cpu().numpy()
        masses['history_mass'][target] = weights[:, :, hist].sum(-1).T.cpu().numpy()
        remote = hist[None, :] & (pos[None, :] <= qs[:, None] - window)
        masses['remote_mass'][target] = (weights * remote).sum(-1).T.cpu().numpy()
        t = qs - start
        previous = t[:, None] - torch.arange(1, window + 1, device=device)[None, :]
        keys = (prompt + previous).clamp(0, length - 1)
        valid = previous >= 0
        gathered = weights.gather(2, keys[None, :, :].expand(heads, -1, -1))
        # lag1 is the query's OWN key (the preceding response token), not y_t.
        valid &= hist[keys]
        local[target] = (gathered * valid[None]).permute(1, 0, 2).cpu().numpy()
        reconstructed = (weights[None].to(v.dtype) @ v).transpose(1, 2).reshape(b - a, -1).float()
        native = context[0, a:b].float()
        norm_error += float((reconstructed - native).square().sum())
        norm_native += float(native.square().sum())
    return dict(local_attention=local, **masses,
                attention_replay_relative_error=np.sqrt(norm_error / max(norm_native, 1e-20)),
                softmax_mass_error=mass_error)


@torch.inference_mode()
def capture_response(model, row, *, window=16, chunk=16, logit_chunk=32, special_ids=()):
    config = model.config
    if config.model_type != 'llama' or getattr(config, 'pretraining_tp', 1) != 1:
        raise ValueError('supported observer: ordinary dense Llama/Llama3 GQA, tensor-parallel=1')
    if getattr(config, 'sliding_window', None) or min(chunk, logit_chunk, window) < 1:
        raise ValueError('sliding-window models are not supported by this capture')
    ids, prompt = row['token_ids'], row['prompt_length']
    n = len(ids) - prompt
    if n < 1 or len(row['offsets']) != n or prompt < 1:
        raise ValueError('original prompt/response token coordinates differ')
    layers, heads = len(model.model.layers), config.num_attention_heads
    result = dict(local_attention=np.empty((n, layers, heads, window), np.float32),
                  source_mass=np.empty((n, layers, heads), np.float32),
                  history_mass=np.empty((n, layers, heads), np.float32),
                  remote_mass=np.empty((n, layers, heads), np.float32))
    errors = np.empty(layers); handles = []; seen = set()
    device = model.get_input_embeddings().weight.device
    response_special = np.isin(ids[prompt:], list(special_ids))
    def register(index, layer):
        storage = {}
        def projection_hook(name):
            def read(module, args, output):
                storage[name] = output.detach()
            return read
        def context_hook(module, args):
            storage['context'] = args[0].detach()
        def read_attention(module, args, kwargs, output):
            bound = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
            position = bound.get('position_embeddings', kwargs.get('position_embeddings'))
            qraw, kraw, vraw = (storage[name] for name in ('q', 'k', 'v'))
            dim = getattr(module, 'head_dim', config.hidden_size // heads)
            shape = (1, len(ids) - 1, -1, dim)
            if position is None:
                pos_ids = bound.get('position_ids', kwargs.get('position_ids'))
                if not hasattr(module, 'rotary_emb') or pos_ids is None:
                    raise ValueError('Llama attention must expose RoPE cos/sin or legacy rotary_emb + position_ids')
                position = module.rotary_emb(vraw.reshape(shape).transpose(1, 2), pos_ids)
            if getattr(module, 'q_norm', None) is not None or getattr(module, 'k_norm', None) is not None:
                raise ValueError('Q/K-normalized variants need a separate verified observer')
            part = attention_blocks(qraw.reshape(shape), kraw.reshape(shape), vraw.reshape(shape),
                                    position[0], position[1], storage['context'], start=prompt - 1,
                                    source_mask=row['source_mask'], response_special=response_special,
                                    window=window, chunk=chunk, scaling=getattr(module, 'scaling', None))
            for key in result:
                result[key][:, index] = part[key]
            errors[index] = part['attention_replay_relative_error']
            if part['softmax_mass_error'] > 1e-5 or errors[index] > .02:
                raise ValueError(f'QK replay disagrees with native attention at layer {index}: relative error {errors[index]}')
            seen.add(index); storage.clear()
            # No return value: native output is never modified.
        for name in ('q', 'k', 'v'):
            handles.append(getattr(layer.self_attn, name + '_proj').register_forward_hook(projection_hook(name)))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(context_hook))
        handles.append(layer.self_attn.register_forward_hook(read_attention, with_kwargs=True))
    try:
        for index, layer in enumerate(model.model.layers):
            register(index, layer)
        output = model.model(torch.tensor([ids[:-1]], device=device), use_cache=False,
                             output_attentions=False, output_hidden_states=False, return_dict=True)
        hidden = output.last_hidden_state[0, prompt - 1:]
        if len(seen) != layers:
            raise RuntimeError('native attention hooks were bypassed')
        values = {k: [] for k in ('entropy', 'negative_margin', 'saved_nll')}
        for a in range(0, n, logit_chunk):
            logits = model.lm_head(hidden[a:a + logit_chunk]).float()
            lp = logits.log_softmax(-1)
            top = logits.topk(2, dim=-1).values
            target = torch.as_tensor(ids[prompt + a:prompt + a + len(lp)], device=device)
            values['entropy'].append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
            values['negative_margin'].append((top[:, 1] - top[:, 0]).cpu().numpy())
            values['saved_nll'].append((-lp[torch.arange(len(lp), device=device), target]).cpu().numpy())
        result.update({k: np.concatenate(v) for k, v in values.items()},
                      offsets=np.asarray(row['offsets'], np.int32),
                      attention_replay_relative_error=errors,
                      token_ids=np.asarray(ids, np.int32))
        return result
    finally:
        for handle in handles:
            handle.remove()
