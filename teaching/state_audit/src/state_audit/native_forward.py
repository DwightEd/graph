"""One native forward per observed token; no gradients, edits, or labelled rivals."""

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from .attribution import _projection_gram
from .capture import numpy
from .model.replay import attention_backend, extend_cache
from .native_ledger import finish_ledger, readout_direction
from .native_trace import layer_messages, native_hooks, source_masks

GROUPS = ("prompt", "history", "special")
VECTOR_FIELDS = (
    "residual_before", "residual_mid", "residual_after", "attention_write",
    "mlp_write", "mlp_activation", "head_readout", "head_writes",
    "group_readouts", "direction",
)


def partition_prefix(token_ids, prompt_length, special_ids):
    groups = np.zeros(len(token_ids), dtype=np.int64)
    groups[prompt_length:] = 1
    groups[np.isin(token_ids, special_ids)] = 2
    return groups


def native_candidates(logits, observed_id):
    """The comparator is the highest-logit other token in this same native run."""
    top = logits.topk(2).indices.tolist()
    competitor = top[1] if top[0] == observed_id else top[0]
    return observed_id, competitor


def measure_forward(model, records, cache, hidden, observed_id, groups, grams):
    logits = model.native.lm_head(hidden).float()
    candidates = native_candidates(logits, observed_id)
    residual = records[len(records) - 1]["residual_after"][-1]
    direction, difference, scale = readout_direction(model, residual, candidates)
    masks = source_masks(groups, len(GROUPS), hidden.device)
    arrays = layer_messages(model, records, cache, direction, grams, masks)
    finish_ledger(
        model, records, arrays, hidden, logits, candidates, direction, difference, scale
    )
    logp = logits.log_softmax(-1)
    arrays.update(
        entropy=numpy(-(logp.exp() * logp).sum()),
        surprisal=numpy(-logp[observed_id]),
        observed_id=np.asarray(observed_id),
        competitor_id=np.asarray(candidates[1]),
        group_ids=groups,
        group_names=np.asarray(GROUPS),
    )
    # Preserve head/source scalars and the exact ledger, not multi-GB raw vectors.
    for name in VECTOR_FIELDS:
        del arrays[name]
    return arrays


def iter_forward_traces(
    model, token_ids, prompt_length, targets, special_ids, *, prefill_chunk_size=256
):
    """Target t is scored at q=P+t-1, before its token is ever fed into the model.

    Increasing targets may skip completed cache files. A scoring pass must still
    consume every answer token in order to construct its causal support graph.
    """
    targets = list(targets)
    if any(a >= b for a, b in zip(targets, targets[1:])):
        raise ValueError("Targets must be strictly increasing")
    cache = DynamicCache()
    with torch.no_grad():
        grams = {layer: _projection_gram(model, layer) for layer in range(len(model.layers))}
    for target in targets:
        query = prompt_length + target - 1
        with torch.no_grad():
            extend_cache(model, cache, token_ids[:query], prefill_chunk_size)
            with attention_backend(model, "eager"), native_hooks(model, with_grad=False) as records:
                output = model.native.model(
                    input_ids=model.input_ids([token_ids[query]]),
                    past_key_values=cache, use_cache=True, return_dict=True,
                )
                groups = partition_prefix(token_ids[:query + 1], prompt_length, special_ids)
                arrays = measure_forward(
                    model, records, cache, output.last_hidden_state[0, -1],
                    token_ids[query + 1], groups, grams,
                )
            arrays.update(query=np.asarray(query), target=np.asarray(target))
        yield arrays
