"""Value-path attribution rules inspired by DecompX and product relevance.

The native forward values are unchanged. Attention probabilities and RMS scales
are constants for attribution; SwiGLU splits product relevance equally between
its two inputs. This is a specified decomposition, not the native Jacobian.
"""

from contextlib import ExitStack, contextmanager
from functools import partial
from types import MethodType
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint


def rms_value_path(module, hidden):
    values = hidden.float()
    scale = torch.rsqrt(values.square().mean(-1, keepdim=True) + module.variance_epsilon)
    normalized = (values * scale.detach()).to(hidden.dtype)
    return module.weight * normalized


def swiglu_rule(module, hidden):
    gate = module.gate_proj(hidden)
    up = module.up_proj(hidden)
    activated = module.act_fn(gate)
    secant = gate * gate.sigmoid().detach()
    activated = activated.detach() + (secant - secant.detach())
    product = activated * up
    half = .5 * product
    # Native product in the forward; half of the relevance to each factor.
    shared = product.detach() + (half - half.detach())
    return module.down_proj(shared)


def swiglu_value_path(module, hidden):
    # Recompute only the MLP during reverse passes; do not retain every wide
    # gate/up activation across a long answer and all decoder layers.
    return checkpoint(swiglu_rule, module, hidden, use_reentrant=False)


def observe_projection(record, name, value):
    record[name] = value.detach()
    return value.detach() if name in ("query", "key") else value


def observe_rows(record, name, rows, value):
    selected = value.index_select(0, rows)
    record[name] = selected
    # The measured view must be on the graph, not a disconnected observation.
    return value.index_copy(0, rows, selected)


def observe_mask(record, rows, module, inputs, kwargs):
    mask = kwargs["attention_mask"]
    record["mask"] = None if mask is None else mask[0, 0].index_select(0, rows).detach()


def check_value_layout(model):
    if model.native.config.hidden_act != "silu":
        raise ValueError("Value-path capture requires the adapter's SiLU-gated MLP")
    modules = [model.native.lm_head]
    for layer in model.layers:
        modules.extend((layer.self_attn.v_proj, layer.self_attn.o_proj,
                        layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj))
    if any(module.bias is not None for module in modules):
        raise ValueError("Value-path ledger requires bias-free value/MLP/output projections")


@contextmanager
def value_path_hooks(model, rows):
    """Retain selected head/FFN rows; never detach historical value paths."""
    check_value_layout(model)
    records = {layer: {} for layer in range(len(model.layers))}
    with ExitStack() as stack:
        norm = model.native.model.norm
        stack.enter_context(patch.object(norm, "forward", MethodType(rms_value_path, norm)))
        for index, layer in enumerate(model.layers):
            record = records[index]
            handle = layer.self_attn.register_forward_pre_hook(
                partial(observe_mask, record, rows), with_kwargs=True,
            )
            stack.callback(handle.remove)
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                stack.enter_context(patch.object(norm, "forward", MethodType(rms_value_path, norm)))
            mlp_forward = MethodType(swiglu_value_path, layer.mlp)
            stack.enter_context(patch.object(layer.mlp, "forward", mlp_forward))
            for name in ("query", "key", "value"):
                observer = partial(observe_projection, record, name)
                stack.enter_context(model.bind(name, index, observer))
            for name in ("head_readout", "mlp_write"):
                observer = partial(observe_rows, record, name, rows)
                stack.enter_context(model.bind(name, index, observer))
        yield records
