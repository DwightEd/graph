"""One output choice, native gradients at all answer receivers, bounded CPU snapshots."""

import numpy as np
import torch

from state_audit.attribution import frozen_parameters
from state_audit.capture import numpy
from state_audit.functional_hooks import functional_hooks
from state_audit.message_gates import edge_gates
from state_audit.model.replay import attention_backend


def choose_foil(logits, target):
    alternatives = logits.detach().clone()
    alternatives[target] = -torch.inf
    return int(alternatives.argmax())


def output_values(logits, target, foil):
    logits = logits.float()
    logp = logits[target] - torch.logsumexp(logits, dim=-1)
    margin = logits[target] - logits[foil]
    return torch.stack((logp, margin))


def snapshot_layer(record, gradient):
    """Keep native Q/K/V and output derivatives, not a model backward graph."""
    snapshot = {name: record[name].detach().cpu() for name in ("query", "key", "value", "head")}
    snapshot["gradient"] = gradient.detach().cpu()
    mask = record["mask"]
    snapshot["mask"] = None if mask is None else mask.detach().cpu()
    return snapshot


def observe_token(model, prompt, answer, target, foil=None):
    """Independent margin objective for one target; never read answer[target+1:]."""
    tokens = prompt + answer[:target]
    with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
        embedding = model.native.model.embed_tokens(model.input_ids(tokens)).detach().requires_grad_(True)
        with attention_backend(model, "sdpa"), functional_hooks(model) as records:
            hidden = model.native.model(inputs_embeds=embedding, use_cache=False, return_dict=True).last_hidden_state
            logits = model.native.lm_head(hidden[0, -1]).float()
            foil = choose_foil(logits, answer[target]) if foil is None else foil
            values = output_values(logits, answer[target], foil)
            gradients = torch.autograd.grad(values[1], [record["head"] for record in records.values()])
        with torch.no_grad():
            positions = torch.arange(len(tokens), device=embedding.device)[None]
            rotary = tuple(value.detach().cpu() for value in model.native.model.rotary_emb(embedding, positions))
            layers = [snapshot_layer(record, gradients[layer]) for layer, record in records.items()]
    return dict(values=numpy(values), foil=foil, layers=layers, rotary=rotary, prompt=len(prompt))


@torch.no_grad()
def deleted_values(model, prompt, answer, target, foil, edges=(), strength=1.0):
    absolute = [(int(layer), int(head), len(prompt) + int(query), len(prompt) + int(key))
                for layer, head, query, key in edges]
    with attention_backend(model, "sdpa"), edge_gates(model, absolute, strength):
        hidden = model.forward(prompt + answer[:target])
        logits = model.native.lm_head(hidden[-1])
    return numpy(output_values(logits, answer[target], foil))
