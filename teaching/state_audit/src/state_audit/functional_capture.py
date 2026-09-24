"""Whole-phrase native sensitivity with residual and FFN transfer channels.

Each objective sums log probabilities over one saved original-answer interval.
Its gradients may include later targets inside that interval. They are neither
independent token gradients nor hallucination scores. Q/K and RMS stay native.
"""

import numpy as np
import torch

from .attribution import frozen_parameters
from .capture import numpy
from .functional_hooks import functional_hooks
from .model.replay import attention_backend


def log_probability(logits, targets):
    return logits.float().log_softmax(-1).gather(-1, targets[:, None])[:, 0]


def rms_readout(model, mid, write, targets, native_logits):
    """Float32 endpoint identity; native low-precision discrepancy is separate."""
    norm = model.native.model.norm
    before, message = mid.float(), write.float()
    after = before + message
    scale_before = (before.square().mean(-1) + norm.variance_epsilon).rsqrt()
    scale_after = (after.square().mean(-1) + norm.variance_epsilon).rsqrt()
    weight = model.native.lm_head.weight
    logits_before = model.native.lm_head(norm(mid)).float()
    rivals = native_logits.detach().clone()
    rivals.scatter_(1, targets[:, None], -torch.inf)
    foil = rivals.argmax(-1)
    direction = (weight[targets].float() - weight[foil].float()) * norm.weight.float()
    previous = (direction * before).sum(-1)
    direct = (direction * message).sum(-1) * scale_after
    rescale = previous * (scale_after - scale_before)
    delta = (direction * after).sum(-1) * scale_after - previous * scale_before
    logp = logits_before.log_softmax(-1)
    return dict(pre_ffn_log_probability=numpy(log_probability(logits_before, targets)),
        pre_ffn_entropy=numpy(-(logp.exp() * logp).sum(-1)),
        rms_direct_margin=numpy(direct), rms_rescale_margin=numpy(rescale),
        rms_margin_delta=numpy(delta), rms_identity_error=numpy(delta - direct - rescale),
        rms_native_rounding_error=numpy(
            native_logits.gather(1, targets[:, None])[:, 0]
            - native_logits.gather(1, foil[:, None])[:, 0]
            - (direction * after).sum(-1) * scale_after),
        rms_foil_ids=numpy(foil), rms_scale_before=numpy(scale_before),
        rms_scale_after=numpy(scale_after))


def attention_rows(model, layer, record, rotary, queries):
    heads, kv_heads, width = model.head_layout(layer)
    count = record["query"].shape[1]
    query = record["query"].view(1, count, heads, width).transpose(1, 2)
    key = record["key"].view(1, count, kv_heads, width).transpose(1, 2)
    query, key = model.implementation.apply_rotary_pos_emb(query, key, *rotary)
    key = key.repeat_interleave(heads // kv_heads, 1)
    scores = query[0, :, queries].float() @ key[0].float().transpose(-1, -2)
    scores *= model.layers[layer].self_attn.scaling
    causal = torch.arange(count, device=queries.device)[None] > queries[:, None]
    scores.masked_fill_(causal[None], -torch.inf)
    if record["mask"] is not None:
        mask = record["mask"][0, 0, queries, :count]
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask[None], -torch.inf)
        else:
            scores += mask[None]
    return scores.softmax(-1).transpose(0, 1)


def group_messages(effects, masks):
    return torch.einsum("qhk,kg->qhg", effects, masks)


def layer_effects(model, layer, record, gradients, rotary, queries, masks):
    head_gradient, ffn_gradient = gradients
    heads, kv_heads, width = model.head_layout(layer)
    head_gradient = head_gradient[0, queries].float().reshape(-1, heads, width)
    downstream = ffn_gradient[0, queries].float()
    projection = model.layers[layer].self_attn.o_proj.weight.float()
    direct_gradient = (downstream @ projection).reshape(-1, heads, width)
    values = record["value"][0].float().reshape(-1, kv_heads, width)
    values = values.repeat_interleave(heads // kv_heads, 1).transpose(0, 1)
    attention = attention_rows(model, layer, record, rotary, queries)
    total = torch.einsum("qhd,hkd->qhk", head_gradient, values) * attention
    direct = torch.einsum("qhd,hkd->qhk", direct_gradient, values) * attention
    mediated = total - direct
    result = {"head_attention": numpy(group_messages(attention, masks))}
    reconstructed = torch.einsum("qhk,hkd->qhd", attention, values)
    native_head = record["head"][0, queries].float().reshape(-1, heads, width)
    result["head_reconstruction_error"] = numpy((reconstructed - native_head).abs().amax(-1))
    for name, value in (("total", total), ("residual", direct), ("ffn_mediated", mediated)):
        result[f"head_{name}"] = numpy(group_messages(value, masks))
        result[f"head_{name}_positive"] = numpy(group_messages(value.clamp_min(0), masks))
        result[f"head_{name}_negative"] = numpy(group_messages((-value).clamp_min(0), masks))
    return result


def capture_observed(model, prefix_ids, target_ids, groups, group_count):
    """One native forward/backward for an exact interval of the original answer."""
    tokens = prefix_ids + target_ids
    prefix_length = len(prefix_ids)
    device = model.native.device
    queries = torch.arange(prefix_length - 1, len(tokens) - 1, device=device)
    targets = torch.tensor(target_ids, device=device)
    key_groups = torch.tensor(groups + [group_count] * (len(target_ids) - 1), device=device)
    masks = torch.nn.functional.one_hot(key_groups, group_count + 1).float()
    with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
        embeddings = model.native.model.embed_tokens(model.input_ids(tokens[:-1])).detach().requires_grad_(True)
        with attention_backend(model, "sdpa"), functional_hooks(model) as records:
            output = model.native.model(inputs_embeds=embeddings, use_cache=False, return_dict=True)
            logits = model.native.lm_head(output.last_hidden_state[0, queries]).float()
            logp = log_probability(logits, targets)
            sites = [embeddings]
            for record in records.values():
                sites.extend((record["head"], record["ffn"]))
            gradients = torch.autograd.grad(logp.sum(), sites)
            with torch.no_grad():
                return collect_observed(model, embeddings, records, gradients, logits, logp,
                                        queries, targets, masks, prefix_length, key_groups)


def collect_observed(model, embeddings, records, gradients, logits, logp,
                     queries, targets, masks, prefix_length, key_groups):
    positions = torch.arange(embeddings.shape[1], device=embeddings.device)[None]
    rotary = model.native.model.rotary_emb(embeddings, positions)
    roots = (embeddings.float() * gradients[0].float()).sum(-1)[0]
    distribution = logits.log_softmax(-1)
    result = dict(log_probability=numpy(logp), target_ids=numpy(targets), query=numpy(queries),
                  key_group_ids=numpy(key_groups),
                  entropy=numpy(-(distribution.exp() * distribution).sum(-1)),
                  prefix_root_sensitivity=numpy(roots[:prefix_length]),
                  within_unit_root_sensitivity=numpy(roots[prefix_length:]))
    layers = [layer_effects(model, layer, record, gradients[1 + 2 * layer:3 + 2 * layer],
                           rotary, queries, masks) for layer, record in records.items()]
    result.update({name: np.stack([row[name] for row in layers], axis=1) for name in layers[0]})
    result["ffn_write_sensitivity"] = np.stack([
        numpy((gradients[2 + 2 * layer][0, queries].float()
               * record["ffn"][0, queries].float()).sum(-1))
        for layer, record in records.items()], axis=1)
    last = records[len(model.layers) - 1]
    result.update(rms_readout(model, last["mid"][0, queries], last["ffn"][0, queries], targets, logits))
    return result
