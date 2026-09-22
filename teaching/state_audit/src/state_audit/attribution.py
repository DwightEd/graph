"""Target-specific signed message sensitivity from one native prefix replay."""

from contextlib import ExitStack, contextmanager
from functools import partial

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from .capture import CaptureSpec, numpy
from .model.replay import attention_backend, detach_cache, extend_cache, prefill_cache


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
def _target_hooks(model, layers, *, capture_values=True):
    records = {layer: {} for layer in layers}
    first_layer = min(layers)
    names = ("attention", "head_readout")
    if capture_values:
        names += ("value",)
    with ExitStack() as stack:
        for layer, record in records.items():
            for name in names:
                # Only descendants of measured sites need a gradient graph.
                start_gradient = layer == first_layer and name == "head_readout"
                observer = partial(_observe, record, name, start_gradient=start_gradient)
                stack.enter_context(model.bind(name, layer, observer))
        yield records


def _projection_gram(model, layer):
    """Fixed W_O,h^T W_O,h; reuse across targets, preserving each physical head."""
    heads, _, width = model.head_layout(layer)
    projection = model.layers[layer].self_attn.o_proj.weight.detach().float()
    projection = projection.reshape(projection.shape[0], heads, width).permute(1, 2, 0)
    return projection @ projection.transpose(-1, -2)


def _layer_edges(model, layer, record, gradient, gram):
    """[head, key] sensitivities and squared norms of A * V * W_O messages."""
    heads, kv_heads, _ = model.head_layout(layer)
    values = record["value"].float().repeat_interleave(heads // kv_heads, dim=0)
    attention = record["attention"].float()
    query_gradient = gradient[-1].float()
    sensitivity = (values * query_gradient[:, None, :]).sum(-1)

    # The Gram matrix avoids materializing [head, key, residual_width] writes.
    squared_norm = ((values @ gram) * values).sum(-1).clamp_min(0)
    return dict(
        attention=numpy(attention),
        contribution=numpy(attention * sensitivity),
        value_energy=numpy(attention.square() * squared_norm),
        head_readout=numpy(record["head_readout"][-1]),
        head_gradient=numpy(query_gradient),
    )


def _target_readout(model, hidden, target_id, records, grams):
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
        _layer_edges(model, layer, record, gradient, grams[layer])
        for (layer, record), gradient in zip(records.items(), gradients)
    ]
    result.update({name: np.stack([edge[name] for edge in edges]) for name in edges[0]})
    return result


def _cached_query(model, query_id, target_id, layers, cache, grams):
    with (
        attention_backend(model, "eager"),
        _target_hooks(model, layers, capture_values=False) as records,
    ):
        output = model.native.model(
            input_ids=model.input_ids([query_id]),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        for layer, record in records.items():
            record["value"] = cache.layers[layer].values[0].detach()
        return _target_readout(model, output.last_hidden_state[0], target_id, records, grams)


def _target_metadata(result, prefix, target, target_id, layers, special_token_ids):
    result.update(
        layers=np.asarray(layers, dtype=np.int64),
        query=np.asarray(len(prefix) - 1, dtype=np.int64),
        target=np.asarray(target, dtype=np.int64),
        target_id=np.asarray(target_id, dtype=np.int64),
        key_positions=np.arange(len(prefix)),
        ordinary_keys=~np.isin(prefix, special_token_ids),
    )
    return result


def _capture_target(model, token_ids, prompt_length, target, layers, special_token_ids, chunk_size):
    prefix = token_ids[: prompt_length + target]
    target_id = int(token_ids[prompt_length + target])
    selected = CaptureSpec(layers=layers).selected_layers(model)
    with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
        grams = {layer: _projection_gram(model, layer) for layer in selected}
        if chunk_size is None:
            with _target_hooks(model, selected) as records:
                hidden = model.forward(prefix)
                result = _target_readout(model, hidden, target_id, records, grams)
        else:
            cache = prefill_cache(model, prefix[:-1], chunk_size)
            result = _cached_query(model, prefix[-1], target_id, selected, cache, grams)
    return _target_metadata(result, prefix, target, target_id, selected, special_token_ids)


def iter_target_attributions(
    model,
    token_ids,
    prompt_length,
    targets,
    *,
    layers=None,
    special_token_ids=(),
    prefill_chunk_size=256,
):
    """Reuse one answer's causal K/V across strictly increasing target indices.

    Each target has its own derivative. Detach its cached K/V before advancing,
    so a later target never keeps an earlier target's backward graph alive.
    Gaps (including already saved targets on resume) are filled without gradients.
    Yields contain only CPU arrays, with parameter flags and hooks restored.
    """
    targets = list(targets)
    if not targets:
        return
    if prompt_length < 1 or targets[0] < 0 or targets[-1] >= len(token_ids) - prompt_length:
        raise ValueError("Targets must have a nonempty causal prefix and an observed token")
    if any(right <= left for left, right in zip(targets, targets[1:])):
        raise ValueError("Targets must be strictly increasing for causal cache reuse")
    selected = CaptureSpec(layers=layers).selected_layers(model)
    cache = DynamicCache()
    with torch.inference_mode(False):
        grams = {layer: _projection_gram(model, layer) for layer in selected}
    for target in targets:
        prefix = token_ids[: prompt_length + target]
        target_id = int(token_ids[prompt_length + target])
        with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
            extend_cache(model, cache, prefix[:-1], prefill_chunk_size)
            result = _cached_query(model, prefix[-1], target_id, selected, cache, grams)
            detach_cache(cache)
        yield _target_metadata(result, prefix, target, target_id, selected, special_token_ids)


def capture_target_attribution(
    model,
    token_ids,
    prompt_length,
    target,
    *,
    layers=None,
    special_token_ids=(),
    prefill_chunk_size=256,
):
    """Cache prefix [:P+t-1], replay its last query, and score token [P+t].

    Edge arrays have axes [layer, head, prefix_key]. They retain native special-token
    entries; ordinary_keys excludes those entries in aggregate_sources. No attention
    is renormalized or edited. Each call differentiates one target, never a sum of
    token losses. value_energy is ||A_j * V_j * W_O,h||_2 squared, not route mass.
    Contributions are local multiplicative-gate derivatives at the current query;
    layers are separate measurement sites, not additive pieces of one total effect.
    History K/V are constants: causality prevents any current-query edge perturbation
    from changing earlier positions. Only the final query needs a gradient graph,
    starting at its earliest selected head readout. The full prefix, native masks,
    and downstream current-query Q/K/V dependence are preserved. Chunked SDPA
    prefill avoids storing full attention matrices; the final query uses eager
    attention to expose its one row. Kernel rounding can differ from full replay.
    """
    return _capture_target(
        model, token_ids, prompt_length, target, layers, special_token_ids, prefill_chunk_size
    )


def capture_target_attribution_full(
    model, token_ids, prompt_length, target, *, layers=None, special_token_ids=()
):
    """Quadratic-memory full replay reference for small-model equivalence checks."""
    return _capture_target(model, token_ids, prompt_length, target, layers, special_token_ids, None)


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
