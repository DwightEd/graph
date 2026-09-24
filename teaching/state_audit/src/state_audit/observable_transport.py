"""Native cumulative Jacobian actions in the output categorical Fisher metric.

No dense Jacobian, target-token contrast, detached value rules, or model edits.
Sketch coordinates have no factual polarity. Probe rank is an accuracy budget,
not a learned bottleneck or an estimate of the full Jacobian's rank.
"""

import math

import numpy as np
import torch

from .attribution import frozen_parameters
from .capture import numpy
from .functional_capture import attention_rows
from .functional_hooks import functional_hooks
from .model.replay import attention_backend


def output_probes(vocabulary, rank, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(2, (rank, vocabulary), generator=generator)
    return ((2 * signs.float() - 1) / math.sqrt(rank)).to(device)


def fisher_seeds(probability, probes):
    """Rows xi satisfy E[sum xi xi^T] = diag(p) - p p^T."""
    weighted = probes * probability.sqrt()
    return weighted - weighted.sum(-1, keepdim=True) * probability


def pullbacks(hidden, sites, query, seeds):
    """Independent VJPs for one query; discard all other positions immediately."""
    rows = []
    for seed in seeds:
        gradients = torch.autograd.grad(
            hidden[0, query], sites, grad_outputs=seed,
            retain_graph=True,
        )
        rows.append(torch.stack([gradient[0, query].detach() for gradient in gradients]))
        del gradients
    return torch.stack(rows).float()  # probe, (layer x [mid, after]), residual


def operator_geometry(before, after):
    """Output-conditioned adjoint geometry; NOT full-J Henrici or effective rank."""
    change = before - after
    before_energy = before.square().sum()
    after_energy = after.square().sum()
    denominator = (before_energy * after_energy).sqrt().clamp_min(1e-30)
    return dict(
        pullback_gram=numpy(before @ before.T),
        skip_gram=numpy(after @ after.T),
        pullback_skip_cross=numpy(before @ after.T),
        ffn_pullback_energy=numpy(change.square().sum()),
        pullback_energy=numpy(before_energy),
        skip_energy=numpy(after_energy),
        pullback_alignment=numpy((before * after).sum() / denominator),
    )


def message_groups(model, layer, record, rotary, query, groups, group_count):
    """Keep physical heads and automatic source blocks, including predictor self."""
    device = record["value"].device
    queries = torch.tensor([query], device=device)
    attention = attention_rows(model, layer, record, rotary, queries)[0, :, :query + 1]
    heads, kv_heads, width = model.head_layout(layer)
    values = record["value"][0, :query + 1].float().reshape(-1, kv_heads, width)
    values = values.repeat_interleave(heads // kv_heads, 1).transpose(0, 1)
    masks = torch.nn.functional.one_hot(groups[:query + 1], group_count).float()
    messages = torch.einsum("hj,hjd,jg->hgd", attention, values, masks)
    mass = attention @ masks
    projection = model.layers[layer].self_attn.o_proj.weight.float()
    matrices = projection.reshape(projection.shape[0], heads, width).permute(1, 2, 0)
    gram = matrices @ matrices.transpose(-1, -2)
    energy = torch.einsum("hjd,hde,hje->hj", values, gram, values).clamp_min(0)
    magnitude = attention * energy.sqrt()
    return messages, mass, magnitude, attention, projection


def layer_observations(model, layer, record, rotary, query, groups, source_count, prompt, gradients):
    messages, mass, magnitude, attention, projection = message_groups(
        model, layer, record, rotary, query, groups, source_count + 4,
    )
    heads, _, width = model.head_layout(layer)
    before, after = gradients[:, 0], gradients[:, 1]
    total_axis = (before @ projection).reshape(-1, heads, width)
    direct_axis = (after @ projection).reshape(-1, heads, width)
    total = torch.einsum("rhd,hgd->hgr", total_axis, messages)
    direct = torch.einsum("rhd,hgd->hgr", direct_axis, messages)
    source = groups[:query + 1] < source_count
    history = torch.arange(query + 1, device=groups.device) >= prompt
    difference = history.float() - source.float()
    native_head = record["head"][0, query].float().reshape(heads, width)
    return dict(
        response_total=numpy(total), response_residual=numpy(direct),
        response_ffn=numpy(total - direct), read_mass=numpy(mass),
        raw_route=numpy((magnitude * difference).sum() / magnitude.sum().clamp_min(1e-30)),
        raw_attention=numpy((attention * difference).sum(-1).mean()),
        head_reconstruction_error=numpy((messages.sum(1) - native_head).abs().max()),
        **operator_geometry(before, after),
    )


def observe_query(model, records, hidden, rotary, query, prompt, groups, source_count, probes):
    with torch.no_grad():
        logits = model.native.lm_head(hidden[0, query]).float()
        logp = logits.log_softmax(-1)
        seeds = fisher_seeds(logp.exp(), probes)
        weight = model.native.lm_head.weight
        seeds = seeds.to(weight.dtype) @ weight
    sites = [record[name] for record in records.values() for name in ("mid", "ffn")]
    gradients = pullbacks(hidden, sites, query, seeds)
    grouped = groups.clone()
    grouped[query] = source_count + 3
    with torch.no_grad():
        layers = [layer_observations(
            model, layer, record, rotary, query, grouped, source_count, prompt,
            gradients[:, 2 * layer:2 * layer + 2],
        ) for layer, record in records.items()]
    result = {key: np.stack([row[key] for row in layers]) for key in layers[0]}
    result["entropy"] = numpy(-(logp.exp() * logp).sum())
    result["query"] = np.asarray(query)
    return result


def iter_observable_transport(model, token_ids, prompt, targets, groups, source_count, rank=8, seed=37):
    """One native forward per caller's chunk, rank reverse sweeps per target.

    Later input positions are present only for batching; causal attention keeps
    each independent query objective prefix-only. No sampling derivative is used.
    """
    device = model.native.device
    ids = token_ids[:prompt + max(targets)]
    probes = output_probes(model.native.config.vocab_size, rank, seed, device)
    groups = torch.tensor(groups[:len(ids)], device=device)
    with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
        embeddings = model.native.model.embed_tokens(model.input_ids(ids)).detach().requires_grad_(True)
        positions = torch.arange(len(ids), device=device)[None]
        rotary = model.native.model.rotary_emb(embeddings, positions)
        with attention_backend(model, "sdpa"), functional_hooks(model) as records:
            hidden = model.native.model(
                inputs_embeds=embeddings, use_cache=False, return_dict=True,
            ).last_hidden_state
            for target in targets:
                result = observe_query(model, records, hidden, rotary, prompt + target - 1,
                                       prompt, groups, source_count, probes)
                result["target"] = np.asarray(target)
                result["token_id"] = np.asarray(token_ids[prompt + target])
                yield result
