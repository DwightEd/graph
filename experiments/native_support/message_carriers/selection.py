"""Rank gates on unit-mean log probability, retaining physical head and key identity."""

import numpy as np
import torch

from state_audit.attribution import frozen_parameters
from state_audit.capture import numpy
from state_audit.functional_capture import attention_rows, log_probability
from state_audit.functional_hooks import functional_hooks
from state_audit.model.replay import attention_backend


def layer_candidates(model, layer, record, gradient, rotary, queries, prompt, start):
    heads, kv_heads, width = model.head_layout(layer)
    attention = attention_rows(model, layer, record, rotary, queries)
    # Match native eager probability precision; SDPA can differ by roundoff.
    attention = attention.to(record["value"].dtype).float()
    values = record["value"][0].float().reshape(-1, kv_heads, width)
    values = values.repeat_interleave(heads // kv_heads, 1).transpose(0, 1)
    derivative = gradient[0, queries].float().reshape(-1, heads, width)
    keys = slice(prompt, prompt + start)
    effects = torch.einsum("qhd,hjd->qhj", derivative, values[:, keys])
    effects = (effects * attention[:, :, keys]).sum(0)
    eligible = attention[:, :, keys].gt(0).any(0)
    reconstructed = torch.einsum("qhj,hjd->qhd", attention, values)
    observed = record["head"][0, queries].float().reshape(-1, heads, width)
    error = (reconstructed - observed).abs().max()
    return numpy(effects), numpy(eligible), float(error)


def measure_candidates(model, prompt_ids, answer, unit):
    """One full native backward; gradients include later targets INSIDE this unit."""
    prompt, start, stop = len(prompt_ids), unit["start"], unit["stop"]
    tokens = prompt_ids + answer[:stop - 1]
    device = model.native.device
    queries = torch.arange(prompt + start - 1, prompt + stop - 1, device=device)
    targets = torch.tensor(answer[start:stop], device=device)
    with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
        embedding = model.native.model.embed_tokens(model.input_ids(tokens)).detach().requires_grad_(True)
        with attention_backend(model, "sdpa"), functional_hooks(model) as records:
            hidden = model.native.model(inputs_embeds=embedding, use_cache=False,
                                        return_dict=True).last_hidden_state
            logp = log_probability(model.native.lm_head(hidden[0, queries]), targets)
            gradients = torch.autograd.grad(logp.mean(), [record["head"] for record in records.values()])
        with torch.no_grad():
            rotary = model.native.model.rotary_emb(embedding, torch.arange(len(tokens), device=device)[None])
            measured = [layer_candidates(model, layer, record, gradients[layer], rotary,
                        queries, prompt, start) for layer, record in records.items()]
    return dict(logp=numpy(logp), gradient=np.stack([row[0] for row in measured]),
                eligible=np.stack([row[1] for row in measured]),
                reconstruction_error=np.asarray([row[2] for row in measured]))


def select_carriers(conditions, top_k, seed):
    """One strongest history key per head; top K heads across both source worlds.

    The random control matches head and count, not energy or lag. With only one
    eligible key it coincides with the selected edge; this overlap is saved.
    """
    gradients = np.stack([condition["gradient"] for condition in conditions])
    if not np.isfinite(gradients).all():
        raise ValueError("Nonfinite message-gate gradients")
    eligible = np.logical_or.reduce([condition["eligible"] for condition in conditions])
    importance = np.abs(gradients).max(0)
    ranked = []
    for layer, head in np.argwhere(eligible.any(-1)):
        keys = np.flatnonzero(eligible[layer, head])
        key = int(keys[np.argmax(importance[layer, head, keys])])
        ranked.append((int(layer), int(head), key))
    ranked.sort(key=lambda edge: (-importance[edge], *edge))
    chosen = np.asarray(ranked[:top_k], dtype=np.int64).reshape(-1, 3)
    generator, sham = np.random.default_rng(seed), []
    for layer, head, key in chosen:
        alternatives = np.flatnonzero(eligible[layer, head])
        alternatives = alternatives[alternatives != key]
        random_key = int(generator.choice(alternatives)) if len(alternatives) else int(key)
        sham.append((layer, head, random_key))
    approximation = np.asarray([gradients[:, layer, head, key] for layer, head, key in chosen])
    return dict(edges=chosen, sham_edges=np.asarray(sham, dtype=np.int64).reshape(-1, 3),
                approximation=approximation.reshape(-1, 2), eligible_counts=eligible.sum(-1))
