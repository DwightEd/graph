"""Chunked native teacher forcing; no gradients, edits, or labelled rivals."""

from time import perf_counter

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from .attribution import _projection_gram
from .forward_ledger import measure_block
from .model.replay import attention_backend, extend_cache
from .native_ledger import ledger_total
from .native_trace import native_hooks, source_masks

GROUPS = ("prompt", "history", "special")
FORWARD_SITES = (
    "residual_before", "residual_mid", "residual_after",
    "attention_write", "mlp_write", "attention",
)
EDGE_FIELDS = ("attention", "edge_logit_write", "edge_value_energy")


def partition_prefix(token_ids, prompt_length, special_ids):
    groups = np.zeros(len(token_ids), dtype=np.int64)
    groups[prompt_length:] = 1
    groups[np.isin(token_ids, special_ids)] = 2
    return groups


def target_blocks(targets, size):
    """Batch consecutive missing rows; replay gaps without capturing them."""
    block = []
    for target in targets:
        if block and (len(block) == size or target != block[-1] + 1):
            yield block
            block = []
        block.append(target)
    if block:
        yield block


def capture_block(model, cache, tokens, query, count, groups, grams, energies):
    with attention_backend(model, "eager"), native_hooks(
        model, with_grad=False, representations=FORWARD_SITES
    ) as records:
        output = model.native.model(
            input_ids=model.input_ids(tokens[query:query + count]),
            past_key_values=cache, use_cache=True, return_dict=True,
        )
        observed = model.input_ids(tokens[query + 1:query + count + 1])[0]
        masks = source_masks(groups, len(GROUPS), output.last_hidden_state.device)
        return measure_block(
            model, records, cache, output.last_hidden_state[0], observed, masks, grams, energies
        )


def split_block(batch, block, prompt_length, groups, seconds):
    """Discard causally masked future keys before writing the existing token schema."""
    for row, target in enumerate(block):
        query = prompt_length + target - 1
        arrays = {name: value[row] for name, value in batch.items()}
        for name in EDGE_FIELDS:
            arrays[name] = arrays[name][..., :query + 1]
        arrays.update(
            query=np.asarray(query), target=np.asarray(target),
            group_ids=groups[:query + 1], group_names=np.asarray(GROUPS),
            initial_score=np.asarray(arrays["residual_scores"][0, 0]),
            capture_seconds=np.asarray(seconds / len(block)),
            query_chunk_size=np.asarray(len(block)),
        )
        arrays["ledger_total"] = np.asarray(ledger_total(arrays))
        arrays["ledger_error"] = arrays["ledger_total"] - arrays["logit_gap"]
        yield arrays


def iter_forward_traces(
    model, token_ids, prompt_length, targets, special_ids, *,
    prefill_chunk_size=256, query_chunk_size=8,
):
    """Row t reads only keys <= P+t-1 under the native causal attention mask.

    Offline teacher forcing batches known inputs; no row can attend to later rows.
    Increasing targets may skip completed cache files. Scoring still visits every
    answer token in order to construct its causal support graph.
    """
    targets = list(targets)
    if any(a >= b for a, b in zip(targets, targets[1:])):
        raise ValueError("Targets must be strictly increasing")
    if query_chunk_size < 1:
        raise ValueError("Query chunk size must be positive")
    cache, energies = DynamicCache(), {}
    started = perf_counter()
    with torch.no_grad():
        grams = {layer: _projection_gram(model, layer) for layer in range(len(model.layers))}
    for block in target_blocks(targets, query_chunk_size):
        query = prompt_length + block[0] - 1
        with torch.no_grad():
            extend_cache(model, cache, token_ids[:query], prefill_chunk_size)
            groups = partition_prefix(token_ids[:query + len(block)], prompt_length, special_ids)
            batch = capture_block(
                model, cache, token_ids, query, len(block), groups, grams, energies
            )
        seconds = perf_counter() - started
        yield from split_block(batch, block, prompt_length, groups, seconds)
        started = perf_counter()
