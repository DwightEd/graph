"""Native multi-direction responses of individual head/source messages.

One query graph, batched VJPs, detached historical KV. Derivatives include native
downstream attention, FFN and normalization, not just the output projection.
"""

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from .attribution import _projection_gram, frozen_parameters
from .capture import numpy
from .model.replay import attention_backend, detach_cache, extend_cache
from .native_trace import native_hooks

RESPONSE_SITES = ("residual_before", "head_readout", "mlp_write", "attention")


def state_basis(width, rank, seed, device):
    """Fixed common coordinates; no target-dependent rotations or test fitting."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(width, rank, generator=generator)
    basis = torch.linalg.qr(matrix, mode="reduced").Q.T
    return basis.to(device)


def response_axes(model, hidden, observed, basis, choices):
    logits = model.native.lm_head(hidden).float()
    alternatives = logits.detach().clone()
    alternatives[observed] = -torch.inf
    candidates = torch.cat((logits.new_tensor([observed], dtype=torch.long),
                            alternatives.topk(choices - 1).indices))
    selected = logits[candidates]
    axes = torch.cat((basis @ hidden.float(), selected - selected.mean()))
    logp = logits.log_softmax(-1)
    metadata = {"state": numpy(axes[:len(basis)]),
                "candidate_ids": candidates.detach().cpu().numpy(),
                "candidate_logits": numpy(selected), "entropy": numpy(-(logp.exp() * logp).sum()),
                "surprisal": numpy(-logp[observed]),
                "candidate_tail_mass": numpy(1 - logp[candidates].exp().sum())}
    return axes, metadata


def grouped_readouts(model, records, cache, groups, group_count, grams):
    masks = torch.nn.functional.one_hot(groups, group_count).float()
    grouped, attention, energy = [], [], []
    for layer, record in records.items():
        heads, kv_heads, _ = model.head_layout(layer)
        values = cache.layers[layer].values[0].detach().float()
        values = values.repeat_interleave(heads // kv_heads, 0)
        mass = record["attention"][:, -1].detach().float()
        grouped.append(torch.einsum("hj,hjd,jb->hbd", mass, values, masks))
        squared = ((values @ grams[layer]) * values).sum(-1).clamp_min(0)
        attention.append(numpy(mass))
        energy.append(numpy(mass.square() * squared))
    return grouped, np.stack(attention), np.stack(energy)


def directional_responses(model, axes, records, cache, grouped, rank, batch_size):
    """Batch output directions, never sum losses across targets or head identities."""
    sites = [record[name] for record in records.values()
             for name in ("head_readout", "mlp_write")]
    head_chunks, ffn_chunks = [], []
    edge_energy = [np.zeros_like(numpy(r["attention"][:, -1])) for r in records.values()]
    identity = torch.eye(len(axes), device=axes.device)
    for start in range(0, len(axes), batch_size):
        stop = min(start + batch_size, len(axes))
        gradients = torch.autograd.grad(
            axes, sites, grad_outputs=identity[start:stop], is_grads_batched=True,
            retain_graph=stop < len(axes),
        )
        heads, ffns = [], []
        for layer, record in records.items():
            head_gradient = gradients[2 * layer][:, -1].float()
            heads.append(numpy(torch.einsum("chd,hbd->hbc", head_gradient, grouped[layer])))
            state_directions = max(0, min(stop, rank) - start)
            if state_directions:
                heads_count, kv_heads, _ = model.head_layout(layer)
                values = cache.layers[layer].values[0].detach().float()
                values = values.repeat_interleave(heads_count // kv_heads, 0)
                effect = torch.einsum("chd,hjd->hjc", head_gradient[:state_directions], values)
                mass = record["attention"][:, -1].detach().float()
                edge_energy[layer] += numpy((effect * mass[..., None]).square().sum(-1))
            ffn_gradient = gradients[2 * layer + 1][:, -1].float()
            ffns.append(numpy((ffn_gradient * record["mlp_write"][-1].float()).sum(-1)))
        head_chunks.append(np.stack(heads))
        ffn_chunks.append(np.stack(ffns))
    return np.concatenate(head_chunks, -1), np.concatenate(ffn_chunks, -1), np.stack(edge_energy)


def measure_responses(model, records, cache, hidden, observed, basis, groups,
                      group_count, grams, choices, gradient_batch):
    axes, arrays = response_axes(model, hidden, observed, basis, choices)
    grouped, attention, energy = grouped_readouts(model, records, cache, groups, group_count, grams)
    effects, ffns, response_energy = directional_responses(
        model, axes, records, cache, grouped, len(basis), gradient_batch
    )
    group_ids = groups.detach().cpu().numpy()
    arrays.update(attention=attention, edge_value_energy=energy,
                  edge_response_energy=response_energy,
                  group_effect=effects, ffn_effect=ffns,
                  group_attention=attention @ np.eye(group_count, dtype=np.float32)[group_ids],
                  group_ids=group_ids)
    return arrays


def iter_response_traces(model, tokens, prompt, targets, prompt_groups, source_count,
                         special_ids, *, rank=8, choices=4, seed=37,
                         gradient_batch=4, prefill_chunk_size=256):
    """Group order: evidence blocks, strict history, self, other, special.

    Current-query edge derivatives are exact for this native graph (up to kernel
    precision). They are not derivatives through the creation of past KV states.
    """
    cache = DynamicCache()
    basis = state_basis(model.native.config.hidden_size, rank, seed, model.native.device)
    grams = {layer: _projection_gram(model, layer) for layer in range(len(model.layers))}
    for target in targets:
        query = prompt + target - 1
        groups = np.full(query + 1, source_count, dtype=np.int64)
        groups[:prompt] = prompt_groups
        groups[query] = source_count + 1
        groups[np.isin(tokens[:query + 1], special_ids)] = source_count + 3
        groups = torch.as_tensor(groups, device=model.native.device)
        with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
            extend_cache(model, cache, tokens[:query], prefill_chunk_size)
            with attention_backend(model, "eager"), native_hooks(
                model, representations=RESPONSE_SITES
            ) as records:
                output = model.native.model(input_ids=model.input_ids([tokens[query]]),
                                            past_key_values=cache, use_cache=True, return_dict=True)
                arrays = measure_responses(
                    model, records, cache, output.last_hidden_state[0, -1], tokens[query + 1],
                    basis, groups, source_count + 4, grams, choices, gradient_batch,
                )
            detach_cache(cache)
        arrays.update(target=np.asarray(target), query=np.asarray(query),
                      token_id=np.asarray(tokens[query + 1]))
        yield arrays
