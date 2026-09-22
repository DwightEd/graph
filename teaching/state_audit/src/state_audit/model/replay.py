"""Bounded-memory native prefix replay for a single causal query."""

from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache


@contextmanager
def attention_backend(model, implementation):
    """Temporarily choose the native attention kernel and restore it on failure."""
    previous = model.native.config._attn_implementation
    model.native.set_attn_implementation(implementation)
    try:
        yield
    finally:
        model.native.set_attn_implementation(previous)


def extend_cache(model, cache, token_ids, chunk_size):
    """Append only the unseen part of a prefix, without a backward graph.

    An unbounded cache keeps absolute source positions for sliding-window models;
    their native mask still enforces the window. no_grad (not inference_mode) keeps
    these constants usable by the subsequent differentiable query.
    """
    with torch.no_grad(), attention_backend(model, "sdpa"):
        for start in range(cache.get_seq_length(), len(token_ids), chunk_size):
            model.native.model(
                input_ids=model.input_ids(token_ids[start : start + chunk_size]),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
    return cache


def prefill_cache(model, token_ids, chunk_size):
    return extend_cache(model, DynamicCache(), token_ids, chunk_size)


def detach_cache(cache):
    """The next target treats all earlier K/V as constants, including the last query."""
    for layer in cache.layers:
        layer.keys = layer.keys.detach()
        layer.values = layer.values.detach()
