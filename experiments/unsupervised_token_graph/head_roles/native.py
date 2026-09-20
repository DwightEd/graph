"""Frozen-layer Q/K swaps in Llama; use the checkpoint's actual RoPE tensors."""

import numpy as np
import torch

from .blocks import permutation
from .metrics import swap_scores


def rotate(values, cosine, sine):
    # HF Llama pairs the FIRST and SECOND halves, not adjacent coordinates.
    half = values.shape[-1] // 2
    perpendicular = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
    return values * cosine + perpendicular * sine


def attention_row(query, keys, cosine, sine, position, scale, ordinary):
    rotated_query = rotate(query, cosine[position], sine[position])
    rotated_keys = rotate(keys, cosine[:position + 1], sine[:position + 1])
    logits = torch.einsum("hd,hsd->hs", rotated_query.float(), rotated_keys.float()) * scale
    logits[:, ~ordinary] = -torch.inf
    return logits.softmax(-1)


def block_means(attention, pair):
    return torch.stack([attention[:, start:end].mean(-1) for start, end in pair], -1)


def probe_query(query, keys, cosine, sine, position, pairs, ordinary, scale, temperature):
    base = attention_row(query, keys, cosine, sine, position, scale, ordinary)
    before, after, masses = [], [], []
    for pair in pairs:
        order = torch.as_tensor(permutation(position + 1, pair), device=keys.device)
        changed = attention_row(query, keys[:, order], cosine, sine, position, scale, ordinary)
        before.append(block_means(base, pair).cpu().numpy())
        after.append(block_means(changed, pair).cpu().numpy())
        masses.append(sum(base[:, start:end].sum(-1) for start, end in pair).cpu().numpy())
    before, after = np.asarray(before), np.asarray(after)
    scores = swap_scores(before, after, temperature)
    scores["swapped_mass"] = np.mean(masses, axis=0)
    scores["ordinary_entropy"] = (-(base * base.clamp_min(1e-30).log()).sum(-1)).cpu().numpy()
    return scores, before, after, base


def collect_layer(module, hidden, position_embeddings, positions, swaps, ordinary, temperature):
    width = module.head_dim
    query = module.q_proj(hidden)[0].reshape(len(hidden[0]), -1, width).transpose(0, 1)
    keys = module.k_proj(hidden)[0].reshape(len(hidden[0]), -1, width).transpose(0, 1)
    keys = keys.repeat_interleave(query.shape[0] // keys.shape[0], dim=0)
    cosine, sine = (value[0] for value in position_embeddings)
    rows = []
    for position, pairs in zip(positions, swaps):
        valid = torch.as_tensor(ordinary[:position + 1], device=hidden.device)
        scores, before, after, base = probe_query(query[:, position], keys[:, :position + 1],
            cosine, sine, position, pairs, valid, module.scaling, temperature)
        rows.append(dict(scores=scores, before=before, after=after, base=base))
    return rows


def capture(model, token_ids, positions, swaps, excluded, temperature):
    """One native forward, with separate local counterfactual calculations per layer.

    No changed key or attention is fed into the real forward. Checkpoint output
    is unchanged; this identifies local routing behavior, not causal importance.
    """
    if model.config.model_type != "llama":
        raise ValueError("This native probe currently supports Llama checkpoints")
    ordinary = ~np.isin(token_ids, excluded)
    records, handles = {}, []
    for layer_index, layer in enumerate(model.model.layers):
        def before(module, inputs, kwargs, index=layer_index):
            records[index] = collect_layer(module, kwargs["hidden_states"],
                kwargs["position_embeddings"], positions, swaps, ordinary, temperature)
        def after(module, inputs, output, index=layer_index):
            for position, row in zip(positions, records[index]):
                valid = torch.as_tensor(ordinary[:position + 1], device=output[1].device)
                native = output[1][0, :, position, :position + 1].float()
                native = native * valid
                native = native / native.sum(-1, keepdim=True)
                if not torch.isfinite(native).all():
                    raise ValueError("Native ordinary-key mass underflowed; inspect this run in float32")
                row["scores"]["reconstruction_error"] = (native - row.pop("base")).abs().amax(-1).cpu().numpy()
        handles.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(after))
    ids = torch.as_tensor(np.asarray(token_ids)[None], device=next(model.parameters()).device)
    try:
        with torch.inference_mode():
            model.model(input_ids=ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return records
