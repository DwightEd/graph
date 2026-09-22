"""Head-resolved native trajectories using ModelAdapter and bounded causal KV replay."""

from contextlib import ExitStack, contextmanager
from functools import partial

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from .attribution import _projection_gram, frozen_parameters
from .capture import CaptureSpec, numpy
from .model.replay import attention_backend, detach_cache, extend_cache
from .native_dependencies import capture_dependencies
from .native_ledger import finish_ledger, projected_messages, readout_direction

TRACE_SITES = (
    "residual_before",
    "residual_mid",
    "residual_after",
    "head_readout",
    "attention_write",
    "mlp_write",
    "mlp_activation",
    "attention",
)


def observe_site(record, name, value, *, start_gradient=False):
    if start_gradient:
        value = value.detach().requires_grad_(True)
    record[name] = value
    return value


@contextmanager
def native_hooks(model, *, with_grad=True, representations=TRACE_SITES):
    spec = CaptureSpec(representations=representations)
    records = {layer: {} for layer in spec.selected_layers(model)}
    with ExitStack() as stack:
        for layer, record in records.items():
            for name in spec.representations:
                observer = partial(
                    observe_site,
                    record,
                    name,
                    start_gradient=with_grad and layer == 0 and name == "residual_before",
                )
                stack.enter_context(model.bind(name, layer, observer))
        yield records


def source_masks(group_ids, count, device):
    return torch.as_tensor(
        np.asarray(group_ids)[:, None] == np.arange(count), dtype=torch.float32, device=device
    )


def layer_messages(model, records, cache, direction, grams, masks):
    results, biases = [], []
    for layer, record in records.items():
        results.append(
            projected_messages(
                model, layer, record, cache.layers[layer].values[0], direction, grams[layer], masks
            )
        )
        bias = model.layers[layer].self_attn.o_proj.bias
        biases.append(0.0 if bias is None else float(bias.float() @ direction))
    arrays = {name: np.stack([item[name] for item in results]) for name in results[0]}
    arrays["attention_bias_score"] = np.asarray(biases)
    return arrays


def margin_sensitivity(model, records, cache, margin):
    sites = [
        (layer, name, record[name])
        for layer, record in records.items()
        for name in ("head_readout", "mlp_write")
    ]
    gradients = torch.autograd.grad(margin, [item[2] for item in sites])
    edges, mlps = [], []
    for (layer, name, value), gradient in zip(sites, gradients):
        if name == "mlp_write":
            mlps.append(numpy((gradient[-1] * value[-1]).sum()))
            continue
        heads, kv_heads, _ = model.head_layout(layer)
        values = cache.layers[layer].values[0].detach().float()
        values = values.repeat_interleave(heads // kv_heads, dim=0)
        attention = records[layer]["attention"][:, -1].detach().float()
        edges.append(numpy(attention * (values * gradient[-1].float()[:, None]).sum(-1)))
    return dict(edge_margin_sensitivity=np.stack(edges), mlp_margin_sensitivity=np.stack(mlps))


def measure_query(model, cache, plan, records, hidden, grams):
    candidates = plan["candidate_ids"]
    logits = model.native.lm_head(hidden).float()
    residual = records[len(records) - 1]["residual_after"][-1]
    direction, difference, scale = readout_direction(model, residual, candidates)
    masks = source_masks(plan["group_ids"], len(plan["group_names"]), hidden.device)
    with torch.no_grad():
        arrays = layer_messages(model, records, cache, direction, grams, masks)
        finish_ledger(
            model, records, arrays, hidden, logits, candidates, direction, difference, scale
        )
    if plan["dependencies"]:
        arrays.update(
            capture_dependencies(
                records,
                arrays,
                masks,
                plan["positive_groups"],
                plan["negative_groups"],
                plan["receiver_budget"],
            )
        )
    arrays.update(
        margin_sensitivity(model, records, cache, logits[candidates[0]] - logits[candidates[1]])
    )
    logp = logits.detach().log_softmax(-1)
    observed = plan["observed_id"]
    observed_logp = np.asarray(np.nan) if observed is None else numpy(logp[observed])
    arrays.update(
        candidate_logp=numpy(logp[list(candidates)]),
        entropy=numpy(-(logp.exp() * logp).sum()),
        observed_logp=observed_logp,
        query=np.asarray(plan["query"]),
        candidate_ids=np.asarray(candidates),
        group_ids=np.asarray(plan["group_ids"]),
        group_names=np.asarray(plan["group_names"]),
        observed_id=np.asarray(-1 if observed is None else observed),
    )
    return arrays


def iter_native_traces(model, token_ids, plans, *, prefill_chunk_size=256):
    """Plans use absolute increasing query positions; earlier K/V remain constants.

    Gradients follow all downstream computations at this query. They do not measure
    how changing a previous token would change its cached state or future queries.
    All layers/heads are retained, and source groups partition every visible key.
    """
    queries = [plan["query"] for plan in plans]
    if any(a >= b for a, b in zip(queries, queries[1:])):
        raise ValueError("Query positions must be strictly increasing")
    cache = DynamicCache()
    with torch.inference_mode(False):
        grams = {layer: _projection_gram(model, layer) for layer in range(len(model.layers))}
    for plan in plans:
        query = plan["query"]
        with torch.inference_mode(False), torch.enable_grad(), frozen_parameters(model):
            extend_cache(model, cache, token_ids[:query], prefill_chunk_size)
            with attention_backend(model, "eager"), native_hooks(model) as records:
                output = model.native.model(
                    input_ids=model.input_ids([token_ids[query]]),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                arrays = measure_query(
                    model, cache, plan, records, output.last_hidden_state[0, -1], grams
                )
            detach_cache(cache)
        yield arrays
