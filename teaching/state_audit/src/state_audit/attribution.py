"""Target-specific signed message sensitivity from one native prefix replay."""

from contextlib import ExitStack, contextmanager
from functools import partial

import numpy as np
import torch

from .capture import CaptureSpec, numpy


def candidate_margin(logits, target_id):
    """Log odds of the actual token against all other vocabulary entries."""
    alternatives = torch.cat((logits[:target_id], logits[target_id + 1 :]))
    return logits[target_id] - torch.logsumexp(alternatives, dim=-1)


@contextmanager
def frozen_parameters(model):
    """Keep parameter gradients untouched; restore every requires_grad flag."""
    parameters = tuple(model.native.parameters())
    flags = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def _observe(record, name, value, *, start_gradient=False):
    if start_gradient:
        value = value.detach().requires_grad_(True)
    if name == "attention":
        # Copy one row so the observer does not retain the full attention storage.
        record[name] = value[:, -1].detach().clone()
    elif name == "value":
        record[name] = value.detach()
    else:
        # This is the actual o_proj input, not a returned attention view.
        record[name] = value
    return value


@contextmanager
def _target_hooks(model, layers):
    records = {layer: {} for layer in layers}
    first_layer = min(layers)
    with ExitStack() as stack:
        for layer, record in records.items():
            for name in ("attention", "value", "head_readout"):
                # Only descendants of measured sites need a gradient graph.
                start_gradient = layer == first_layer and name == "head_readout"
                observer = partial(_observe, record, name, start_gradient=start_gradient)
                stack.enter_context(model.bind(name, layer, observer))
        yield records


def _layer_edges(model, layer, record, gradient):
    """[head, key] sensitivities and squared norms of A * V * W_O messages."""
    heads, kv_heads, width = model.head_layout(layer)
    values = record["value"].float().repeat_interleave(heads // kv_heads, dim=0)
    attention = record["attention"].float()
    query_gradient = gradient[-1].float()
    sensitivity = (values * query_gradient[:, None, :]).sum(-1)

    projection = model.layers[layer].self_attn.o_proj.weight.detach().float()
    projection = projection.reshape(projection.shape[0], heads, width).permute(1, 2, 0)
    gram = projection @ projection.transpose(-1, -2)
    # The Gram matrix avoids materializing [head, key, residual_width] writes.
    squared_norm = ((values @ gram) * values).sum(-1).clamp_min(0)
    return dict(
        attention=numpy(attention),
        contribution=numpy(attention * sensitivity),
        value_energy=numpy(attention.square() * squared_norm),
        head_readout=numpy(record["head_readout"][-1]),
        head_gradient=numpy(query_gradient),
    )


def _target_readout(model, prefix, target_id, records):
    hidden = model.forward(prefix)
    logits = model.native.lm_head(hidden[-1]).float()
    margin = candidate_margin(logits, target_id)
    readouts = tuple(record["head_readout"] for record in records.values())
    gradients = torch.autograd.grad(margin, readouts)
    log_probs = logits.log_softmax(-1)
    result = dict(
        logits=numpy(logits),
        target_logp=numpy(log_probs[target_id]),
        logit_entropy=numpy(-(log_probs.exp() * log_probs).sum()),
        margin=numpy(margin),
    )
    edges = [
        _layer_edges(model, layer, record, gradient)
        for (layer, record), gradient in zip(records.items(), gradients)
    ]
    result.update({name: np.stack([edge[name] for edge in edges]) for name in edges[0]})
    return result


def capture_target_attribution(
    model, token_ids, prompt_length, target, *, layers=None, special_token_ids=()
):
    """Replay prefix [:P+t], score token [P+t], and return signed edge sensitivity.

    Edge arrays have axes [layer, head, prefix_key]. They retain native special-token
    entries; ordinary_keys excludes those entries in aggregate_sources. No attention
    is renormalized or edited. Each call differentiates one target, never a sum of
    token losses. value_energy is ||A_j * V_j * W_O,h||_2 squared, not route mass.
    Contributions are local multiplicative-gate derivatives at the current query;
    layers are separate measurement sites, not additive pieces of one total effect.
    Autograd starts at the earliest selected full-prefix head readout; upstream
    activations need no gradient graph because input/root attribution is not requested.
    """
    prefix = token_ids[: prompt_length + target]
    target_id = int(token_ids[prompt_length + target])
    selected = CaptureSpec(layers=layers).selected_layers(model)
    with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
        with _target_hooks(model, selected) as records:
            result = _target_readout(model, prefix, target_id, records)
    result.update(
        layers=np.asarray(selected, dtype=np.int64),
        query=np.asarray(len(prefix) - 1, dtype=np.int64),
        target=np.asarray(target, dtype=np.int64),
        target_id=np.asarray(target_id, dtype=np.int64),
        key_positions=np.arange(len(prefix)),
        ordinary_keys=~np.isin(prefix, special_token_ids),
    )
    return result


def aggregate_sources(attribution, source_groups, group_count):
    """Preserve layer/head/group axes and sum ordinary keys without normalization.

    source_groups assigns each prefix key an integer in [0, group_count); -1 excludes
    a key. Negative contribution is reported as a nonnegative suppression magnitude.
    Summed value_energy is a sum of edge squared norms, not a group-write norm.
    """
    source_groups = np.asarray(source_groups)
    masks = source_groups[:, None] == np.arange(group_count)[None, :]
    masks &= attribution["ordinary_keys"][:, None]
    contribution = attribution["contribution"]
    values = dict(
        route_mass=attribution["attention"],
        contribution_positive=np.maximum(contribution, 0),
        contribution_negative=np.maximum(-contribution, 0),
        value_energy=attribution["value_energy"],
    )
    return {name: value @ masks for name, value in values.items()}
