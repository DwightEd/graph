"""Full-prefix value-path provenance and local head writes for each target.

One teacher-forced forward per answer; separate reverse objectives per token and
competitor. SDPA avoids retaining dense full-prefix attention for every layer.
"""

import numpy as np
import torch

from .attribution import _projection_gram, frozen_parameters
from .capture import numpy
from .model.replay import attention_backend
from .value_paths import value_path_hooks


def target_choices(model, hidden, targets, choices):
    logits = model.native.lm_head(hidden).float()
    other = logits.detach().clone()
    other.scatter_(1, targets[:, None], -torch.inf)
    candidates = torch.cat((targets[:, None], other.topk(choices - 1).indices), dim=-1)
    selected = logits.gather(1, candidates)
    logp = logits.log_softmax(-1)
    return selected[:, :1] - selected[:, 1:], {
        "candidate_ids": candidates.detach().cpu().numpy(), "candidate_logits": numpy(selected),
        "entropy": numpy(-(logp.exp() * logp).sum(-1)),
        "surprisal": numpy(-logp.gather(1, targets[:, None])[:, 0]),
        "candidate_tail_mass": numpy(1 - logp.gather(1, candidates).exp().sum(-1)),
    }


def attention_row(model, layer, record, rotary, query, row):
    """Reconstruct only this native row, including GQA, RoPE and window mask."""
    cosine, sine = rotary
    queries, keys = model.implementation.apply_rotary_pos_emb(
        record["query"][None], record["key"][None], cosine, sine,
    )
    heads, kv_heads, _ = model.head_layout(layer)
    keys = keys[0, :, :query + 1].repeat_interleave(heads // kv_heads, 0)
    logits = (queries[0, :, query:query + 1] @ keys.transpose(-1, -2))[:, 0]
    logits *= model.layers[layer].self_attn.scaling
    if record["mask"] is not None:
        mask = record["mask"][row, :query + 1]
        if mask.dtype == torch.bool:
            logits.masked_fill_(~mask, -torch.inf)
        else:
            logits += mask
    return logits.float().softmax(-1).to(queries.dtype).float()


def group_parts(effects, masks):
    """Keep opposing token effects before summing them into source blocks."""
    return {"positive": masks.T @ effects.clamp_min(0),
            "negative": masks.T @ (-effects).clamp_min(0)}


def read_layer(model, layer, record, gradients, rotary, masks, query, row, gram):
    head_gradient, ffn_gradient = gradients
    attention = attention_row(model, layer, record, rotary, query, row)
    heads, kv_heads, _ = model.head_layout(layer)
    values = record["value"][:, :query + 1].float().repeat_interleave(heads // kv_heads, 0)
    effects = torch.einsum("chd,hjd->hjc", head_gradient[:, row].float(), values)
    effects *= attention[..., None]
    positive = torch.einsum("hjc,jg->hgc", effects.clamp_min(0), masks)
    negative = torch.einsum("hjc,jg->hgc", (-effects).clamp_min(0), masks)
    energy = ((values @ gram) * values).sum(-1).clamp_min(0)
    ffn = (ffn_gradient[:, row].float() * record["mlp_write"][row].float()).sum(-1)
    return {"head_positive": numpy(positive), "head_negative": numpy(negative),
            "ffn_choice": numpy(ffn), "attention": numpy(attention),
            "edge_value_energy": numpy(attention.square() * energy)}


def capture_row(model, embeddings, records, rotary, gradients, margins, metadata,
                groups, group_count, grams, query, target, row):
    masks = torch.nn.functional.one_hot(groups[:query + 1], group_count).float()
    effects = (gradients[0][:, 0, :query + 1].float() * embeddings[0, :query + 1].float()).sum(-1).T
    roots = group_parts(effects, masks)
    result = {f"root_{name}": numpy(value) for name, value in roots.items()}
    result.update(root_token_choice=numpy(effects), group_ids=groups[:query + 1].cpu().numpy(),
                  margin=numpy(margins[row]), target=np.asarray(target), query=np.asarray(query),
                  token_id=metadata["candidate_ids"][row, 0],
                  ledger_error=numpy(effects.sum(0) - margins[row]))
    result.update({name: value[row] for name, value in metadata.items()})
    layers = [read_layer(model, layer, record, gradients[1 + 2 * layer:3 + 2 * layer],
                        rotary, masks, query, row, grams[layer])
              for layer, record in records.items()]
    result.update({name: np.stack([values[name] for values in layers]) for name in layers[0]})
    return result


def objective_gradients(margins, row, sites, gradient_batch, last_row):
    count = margins.shape[1]
    chunks = [[] for _ in sites]
    for start in range(0, count, gradient_batch):
        stop = min(start + gradient_batch, count)
        retain = not (last_row and stop == count)
        if stop - start == 1:
            gradients = torch.autograd.grad(margins[row, start], sites, retain_graph=retain)
            gradients = [value.unsqueeze(0) for value in gradients]
        else:
            vectors = torch.zeros((stop - start, *margins.shape), device=margins.device)
            vectors[torch.arange(stop - start), row, torch.arange(start, stop)] = 1
            gradients = torch.autograd.grad(margins, sites, grad_outputs=vectors,
                is_grads_batched=True, retain_graph=retain)
        for values, gradient in zip(chunks, gradients):
            values.append(gradient)
    return [torch.cat(values) for values in chunks]


def iter_value_paths(model, token_ids, prompt, targets, groups, group_count,
                     *, choices=4, gradient_batch=1):
    """Source groups label input embeddings, including historical answer roots.

    Future positions are masked by the native decoder. No gradient through token
    sampling is implied. Parameters, hooks and backend are restored on close.
    """
    targets = list(targets)
    if not targets:
        return
    device = model.native.device
    queries = torch.tensor([prompt + target - 1 for target in targets], device=device)
    prefix = token_ids[:int(queries[-1]) + 1]
    groups = torch.as_tensor(groups[:len(prefix)], device=device, dtype=torch.long)
    observed = torch.tensor([token_ids[prompt + target] for target in targets], device=device)
    with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
        embeddings = model.native.model.embed_tokens(model.input_ids(prefix)).detach()
        embeddings.requires_grad_(True)
        positions = torch.arange(len(prefix), device=device)[None]
        with torch.no_grad():
            rotary = model.native.model.rotary_emb(embeddings, positions)
            grams = {layer: _projection_gram(model, layer) for layer in range(len(model.layers))}
        with attention_backend(model, "sdpa"), value_path_hooks(model, queries) as records:
            output = model.native.model(inputs_embeds=embeddings, use_cache=False, return_dict=True)
            selected = output.last_hidden_state[0, queries]
            margins, metadata = target_choices(model, selected, observed, choices)
            sites = [embeddings]
            for record in records.values():
                sites.extend((record["head_readout"], record["mlp_write"]))
            for row, target in enumerate(targets):
                gradients = objective_gradients(
                    margins, row, sites, gradient_batch, row == len(targets) - 1,
                )
                with torch.no_grad():
                    result = capture_row(
                        model, embeddings, records, rotary, gradients, margins, metadata,
                        groups, group_count, grams, int(queries[row]), target, row,
                    )
                del gradients
                yield result
