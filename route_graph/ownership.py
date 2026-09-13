"""Factorial interventions on source node values and attention connections.

The graph's actual high-dimensional V attributes and A edges are manipulated
separately. This is a mechanism experiment, not a trained graph classifier.
"""

import numpy as np
import torch


def source_context_delta(
    base_attention, base_values, donor_attention, donor_values, factor
):
    """Return a source-message change [query, head, head_dim], before O.

    A is [head, query, source_key]; V is [source_key, KV_head, head_dim].
    E interventions keep each receiving head's total source attention mass.
    """
    if factor not in {"x", "e", "xe"}:
        raise ValueError("factor must be x, e, or xe")
    a0, a1, v0, v1 = [
        x.float() for x in (base_attention, donor_attention, base_values, donor_values)
    ]
    if a0.ndim != 3 or v0.ndim != 3 or a0.shape != a1.shape or v0.shape != v1.shape:
        raise ValueError("incompatible attention/value shapes")
    heads = a0.shape[0]
    if a0.shape[-1] != v0.shape[0] or heads % v0.shape[1]:
        raise ValueError("incompatible source keys or GQA")
    if (
        any(not torch.isfinite(x).all() for x in (a0, a1, v0, v1))
        or (a0 < 0).any()
        or (a1 < 0).any()
    ):
        raise ValueError("invalid source attributes")
    mass0, mass1 = a0.sum(-1, keepdim=True), a1.sum(-1, keepdim=True)
    if factor != "x" and ((mass0 > 0) & (mass1 == 0)).any():
        raise ValueError("donor cannot redistribute nonzero source mass")
    weights = a0
    if factor != "x":
        scale = mass0 / torch.where(mass1 > 0, mass1, torch.ones_like(mass1))
        weights = a1 * scale
    values = v0 if factor == "e" else v1
    repeats = heads // v0.shape[1]
    base = torch.einsum("hqk,khd->qhd", a0, v0.repeat_interleave(repeats, dim=1))
    changed = torch.einsum(
        "hqk,khd->qhd", weights, values.repeat_interleave(repeats, dim=1)
    )
    delta = changed - base
    return delta, {
        "source_mass_max_error": float(
            (weights.sum(-1, keepdim=True) - mass0).abs().max()
        ),
        "delta_l2": float(delta.norm()),
        "base_source_context_l2": float(base.norm()),
    }


@torch.no_grad()
def capture_source_factors(model, ids, queries, source_positions, layers):
    """Capture A/V from one full forward, with all chosen queries and source keys."""
    captured, handles = {}, []
    source_positions, queries = list(source_positions), list(queries)
    kv_heads = model.config.num_key_value_heads

    def value_hook(index):
        def hook(module, args, output):
            value = output[0, source_positions].reshape(
                len(source_positions), kv_heads, -1
            )
            captured[index]["values"] = value.float().cpu()

        return hook

    def attention_hook(index):
        def hook(module, args, output):
            captured[index]["attention"] = (
                output[1][0, :, queries][:, :, source_positions].float().cpu()
            )

        return hook

    def mlp_hook(index):
        def hook(module, args, output):
            captured[index]["mlp_update"] = output[0, queries].float().cpu()

        return hook

    try:
        for index in layers:
            captured[index] = {}
            attention = model.model.layers[index].self_attn
            handles.extend(
                [
                    attention.v_proj.register_forward_hook(value_hook(index)),
                    attention.register_forward_hook(attention_hook(index)),
                    model.model.layers[index].mlp.register_forward_hook(
                        mlp_hook(index)
                    ),
                ]
            )
        output = model(
            torch.as_tensor(ids, device=model.device)[None],
            use_cache=False,
            output_attentions=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    return output.logits[0, queries].float().cpu(), captured


@torch.no_grad()
def intervene_source_factors(model, ids, read_queries, source_positions, patches):
    """Recompute actual suffix after one/multiple layers' source-only exchanges.

    Each patch gives layer, query positions, factor, donor A/V for those queries.
    The O projection is rerun at its original full shape, making same-world
    exchange an exact sham even under bf16 matrix multiplication.
    endpoint_swap exchanges two CURRENT source edge endpoints, with V fixed.
    """
    handles, diagnostics = [], []
    if len({p["layer"] for p in patches}) != len(patches):
        raise ValueError("one patch specification per layer required")
    sources = list(source_positions)
    kv_heads = model.config.num_key_value_heads
    try:
        for spec in patches:
            attention = model.model.layers[spec["layer"]].self_attn
            query = list(spec["queries"])
            if any(k > min(query) for k in sources):
                raise ValueError(
                    "source keys must already be visible at every patched query"
                )
            cache = {}
            if spec["factor"] == "endpoint_swap":
                pair = spec["source_indices"]
                if (
                    len(pair) != 2
                    or len(set(pair)) != 2
                    or any(i < 0 or i >= len(sources) for i in pair)
                ):
                    raise ValueError("expected two distinct source endpoint indices")

            if spec["factor"] == "mlp":

                def mlp_exchange(module, args, output, spec=spec, query=query):
                    changed = output.clone()
                    donor = spec["mlp_update"].to(output.device, output.dtype)
                    if donor.shape != output[0, query].shape:
                        raise ValueError("donor MLP query shape differs")
                    diagnostics.append(
                        {
                            "layer": spec["layer"],
                            "delta_l2": float(
                                (donor.float() - output[0, query].float()).norm()
                            ),
                            "base_mlp_l2": float(output[0, query].float().norm()),
                        }
                    )
                    changed[0, query] = donor
                    return changed

                handles.append(
                    model.model.layers[spec["layer"]].mlp.register_forward_hook(
                        mlp_exchange
                    )
                )
                continue

            def value_hook(module, args, output, cache=cache):
                cache["values"] = output[0, sources].reshape(len(sources), kv_heads, -1)

            def context_hook(module, args, cache=cache):
                cache["context"] = args[0]

            def exchange(module, args, output, spec=spec, query=query, cache=cache):
                weights = output[1][0, :, query][:, :, sources]
                if spec["factor"] == "endpoint_swap":
                    pair = list(spec["source_indices"])
                    donor_attention = weights.clone()
                    donor_attention[:, :, pair] = weights[:, :, pair[::-1]]
                    donor_values, factor = cache["values"], "e"
                else:
                    donor_attention = spec["attention"].to(weights.device)
                    donor_values = spec["values"].to(weights.device)
                    factor = spec["factor"]
                delta, stats = source_context_delta(
                    weights,
                    cache["values"],
                    donor_attention,
                    donor_values,
                    factor,
                )
                context = cache["context"].clone()
                context[0, query] = (
                    context[0, query].float() + delta.reshape(len(query), -1)
                ).to(context.dtype)
                projected = module.o_proj(context)
                diagnostics.append({"layer": spec["layer"], **stats})
                return projected, output[1]

            handles.extend(
                [
                    attention.v_proj.register_forward_hook(value_hook),
                    attention.o_proj.register_forward_pre_hook(context_hook),
                    attention.register_forward_hook(exchange),
                ]
            )
        output = model(torch.as_tensor(ids, device=model.device)[None], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return output.logits[0, list(read_queries)].float().cpu(), diagnostics


def factorial_effects(margins):
    """2x2 mean main effects and difference-in-differences, no label fitting."""
    values = np.asarray(margins, dtype=float)
    if values.shape != (2, 2) or not np.isfinite(values).all():
        raise ValueError("expected finite [source_world, history_world] margins")
    return {
        "source_main_effect": float((values[1] - values[0]).mean()),
        "history_main_effect": float((values[:, 1] - values[:, 0]).mean()),
        "interaction": float(values[1, 1] - values[1, 0] - values[0, 1] + values[0, 0]),
    }
