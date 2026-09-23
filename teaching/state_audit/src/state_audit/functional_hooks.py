"""Native-gradient observations with scoped FFN recomputation, never value rules."""

from contextlib import ExitStack, contextmanager
from functools import partial
from unittest.mock import patch

from torch.utils.checkpoint import checkpoint


def observe_input(record, name, module, inputs):
    record[name] = inputs[0]


def observe_output(record, name, module, inputs, output):
    record[name] = output


def observe_detached(record, name, module, inputs, output):
    record[name] = output.detach()


def observe_mask(record, module, inputs, kwargs):
    mask = kwargs["attention_mask"]
    record["mask"] = None if mask is None else mask.detach()


def checkpoint_forward(forward, hidden):
    return checkpoint(forward, hidden, use_reentrant=False)


@contextmanager
def functional_hooks(model):
    records = {index: {} for index in range(len(model.layers))}
    with ExitStack() as stack:
        for index, layer in enumerate(model.layers):
            record = records[index]
            for name, module in (("head", layer.self_attn.o_proj),
                                 ("mid", layer.post_attention_layernorm)):
                handle = module.register_forward_pre_hook(partial(observe_input, record, name))
                stack.callback(handle.remove)
            handle = layer.mlp.register_forward_hook(partial(observe_output, record, "ffn"))
            stack.callback(handle.remove)
            for name, module in (("query", layer.self_attn.q_proj),
                                 ("key", layer.self_attn.k_proj), ("value", layer.self_attn.v_proj)):
                handle = module.register_forward_hook(partial(observe_detached, record, name))
                stack.callback(handle.remove)
            handle = layer.self_attn.register_forward_pre_hook(
                partial(observe_mask, record), with_kwargs=True)
            stack.callback(handle.remove)
            stack.enter_context(patch.object(layer.mlp, "forward",
                partial(checkpoint_forward, layer.mlp.forward)))
        yield records
