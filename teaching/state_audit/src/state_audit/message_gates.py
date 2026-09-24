"""Sparse A[q,j] V[j] gates before W_O, with native downstream recomputation.

Only selected rows are reconstructed; the full native attention uses SDPA.
Subtraction has kernel/dtype roundoff relative to eager A @ V deletion.
No attention renormalization, frozen donor, token removal, or detached replay.
"""

from contextlib import ExitStack, contextmanager
from functools import partial

from .functional_capture import attention_rows
from .functional_hooks import observe_detached


def observe_attention(record, module, inputs, kwargs):
    record["mask"] = kwargs["attention_mask"]
    record["rotary"] = kwargs["position_embeddings"]


def subtract_messages(model, layer, record, queries, edges, strength, module, inputs):
    """Each edge is (head, absolute key); gate applies at every supplied query."""
    heads, kv_heads, width = model.head_layout(layer)
    original = inputs[0]
    changed = original.clone().view(1, original.shape[1], heads, width)
    attention = attention_rows(model, layer, record, record["rotary"], queries)
    attention = attention.to(original.dtype)
    values = record["value"][0].view(-1, kv_heads, width)
    for head, key in edges:
        message = attention[:, head, key, None] * values[key, head // (heads // kv_heads)]
        changed[0, queries, head] -= strength * message
    return (changed.reshape_as(original), *inputs[1:])


@contextmanager
def message_gates(model, queries, edges, strength=1.0):
    """Edges use (layer, head, absolute key), preserving native Q/K/V at each layer."""
    with ExitStack() as stack:
        for layer in sorted({edge[0] for edge in edges}):
            record = {}
            attention = model.layers[layer].self_attn
            for name in ("query", "key", "value"):
                projection = {"query": attention.q_proj, "key": attention.k_proj,
                              "value": attention.v_proj}[name]
                handle = projection.register_forward_hook(partial(observe_detached, record, name))
                stack.callback(handle.remove)
            handle = attention.register_forward_pre_hook(
                partial(observe_attention, record), with_kwargs=True)
            stack.callback(handle.remove)
            selected = [(head, key) for index, head, key in edges if index == layer]
            handle = attention.o_proj.register_forward_pre_hook(partial(
                subtract_messages, model, layer, record, queries, selected, strength))
            stack.callback(handle.remove)
        yield
