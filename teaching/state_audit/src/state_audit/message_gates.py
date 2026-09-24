"""Sparse A[q,j] V[j] gates before W_O, with native downstream recomputation.

Only selected rows are reconstructed; the full native attention uses SDPA.
Subtraction has kernel/dtype roundoff relative to eager A @ V deletion.
No attention renormalization, frozen donor, token removal, or detached replay.
"""

from contextlib import ExitStack, contextmanager
from functools import partial

import torch

from .functional_capture import attention_rows
from .functional_hooks import observe_detached


def observe_attention(record, module, inputs, kwargs):
    record["mask"] = kwargs["attention_mask"]
    record["rotary"] = kwargs["position_embeddings"]


def subtract_messages(model, layer, record, edges, strength, module, inputs):
    """Each edge is (head, absolute query, absolute key); no Cartesian expansion."""
    heads, kv_heads, width = model.head_layout(layer)
    original = inputs[0]
    positions = sorted({query for _, query, _ in edges})
    queries = torch.tensor(positions, device=original.device)
    changed = original.clone().view(1, original.shape[1], heads, width)
    attention = attention_rows(model, layer, record, record["rotary"], queries)
    attention = attention.to(original.dtype)
    values = record["value"][0].view(-1, kv_heads, width)
    for head, query, key in edges:
        row = positions.index(query)
        message = attention[row, head, key] * values[key, head // (heads // kv_heads)]
        changed[0, query, head] -= strength * message
    return (changed.reshape_as(original), *inputs[1:])


@contextmanager
def edge_gates(model, edges, strength=1.0):
    """Native (layer, head, absolute query, absolute key) interventions."""
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
            selected = [(head, query, key) for index, head, query, key in edges if index == layer]
            handle = attention.o_proj.register_forward_pre_hook(partial(
                subtract_messages, model, layer, record, selected, strength))
            stack.callback(handle.remove)
        yield


@contextmanager
def message_gates(model, queries, edges, strength=1.0):
    """V1 bundle: apply each (layer,head,key) at all the supplied queries."""
    expanded = [(layer, head, query, key) for layer, head, key in edges for query in queries.tolist()]
    with edge_gates(model, expanded, strength):
        yield
