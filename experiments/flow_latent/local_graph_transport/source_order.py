"""Content-preserving document permutations and forced-prefix output measurements.

These outputs describe order sensitivity. They are not factual probabilities,
source-owner certification or a new hallucination detector.
"""
import re
import time

import numpy as np
import torch


CONDITIONS = ('native', 'cycle_fwd', 'cycle_rev')
ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def document_blocks(prompt):
    """Headers, complete bodies and trailing separators move together."""
    matches = list(re.finditer(r'^passage ([123]):', prompt, flags=re.MULTILINE))
    if [match.group(1) for match in matches] != ['1', '2', '3']:
        raise ValueError('Expected exactly the three canonical passage headers')
    endings = list(re.finditer(r'^In case the passages', prompt, flags=re.MULTILINE))
    if len(endings) != 1 or endings[0].start() <= matches[-1].start():
        raise ValueError('Expected one unchanged trailing instruction')
    cuts = [match.start() for match in matches] + [endings[0].start()]
    blocks = [prompt[cuts[index]:cuts[index + 1]] for index in range(3)]
    if not all(block.endswith('\n\n') for block in blocks):
        raise ValueError('Each original document must include its separator tail')
    return dict(prefix=prompt[:cuts[0]], blocks=blocks, suffix=prompt[cuts[-1]:],
                character_cuts=cuts)


def token_cut(offsets, character):
    """Reject cuts through a merged BPE token rather than inventing fragments."""
    offsets = np.asarray(offsets, dtype=np.int64)
    if np.any((offsets[:, 0] < character) & (offsets[:, 1] > character)):
        raise ValueError('Document boundary crosses a tokenizer token')
    after = np.flatnonzero(offsets[:, 0] >= character)
    if not len(after) or offsets[after[0], 0] != character:
        raise ValueError('Document cut does not coincide with a token start')
    return int(after[0])


def compile_orders(tokenizer, prompt, expected_ids, expected_source_mask):
    """Prove text, BPE, original document identity and token-map bijection."""
    parts = document_blocks(prompt)
    template = lambda text: tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
    original_text = template(prompt)
    if original_text.count(prompt) != 1:
        raise ValueError('Chat template did not preserve the exact prompt once')
    start = original_text.index(prompt)
    encoded = tokenizer(original_text, add_special_tokens=False, return_offsets_mapping=True)
    original_ids = np.asarray(encoded['input_ids'], dtype=np.int64)
    if not np.array_equal(original_ids, expected_ids):
        raise ValueError('Canonical tokenizer IDs differ from the historical cache')
    offsets = np.asarray(encoded['offset_mapping'], dtype=np.int64)
    # Reproduce the historical whole-passage mask, including its last-token overlap.
    source_start = start + parts['character_cuts'][0]
    source_stop = start + prompt.index('\nIn case the passages', parts['character_cuts'][0])
    source_mask = (offsets[:, 0] < source_stop) & (offsets[:, 1] > source_start)
    if not np.array_equal(source_mask, expected_source_mask):
        raise ValueError('Canonical source mask differs from the historical cache')
    cuts = [token_cut(offsets, start + cut) for cut in parts['character_cuts']]
    prefix = np.arange(cuts[0], dtype=np.int64)
    chunks = [np.arange(cuts[index], cuts[index + 1], dtype=np.int64) for index in range(3)]
    suffix = np.arange(cuts[-1], len(original_ids), dtype=np.int64)
    original_doc = np.full(len(original_ids), -1, dtype=np.int64)
    internal_offset = np.full(len(original_ids), -1, dtype=np.int64)
    for identity, chunk in enumerate(chunks, start=1):
        original_doc[chunk] = identity
        internal_offset[chunk] = np.arange(len(chunk))
    variants = {}
    for name, order in zip(CONDITIONS, ORDERS):
        raw = parts['prefix'] + ''.join(parts['blocks'][index] for index in order) + parts['suffix']
        text = template(raw)
        new_to_old = np.concatenate([prefix, *[chunks[index] for index in order], suffix])
        old_to_new = np.argsort(new_to_old)
        ids = np.asarray(tokenizer(text, add_special_tokens=False)['input_ids'], dtype=np.int64)
        if not np.array_equal(np.sort(new_to_old), np.arange(len(original_ids))):
            raise ValueError('Document map is not bijective')
        if not np.array_equal(ids, original_ids[new_to_old]):
            raise ValueError('Retokenized text differs from the original token-chunk permutation')
        if not np.array_equal(new_to_old[old_to_new], np.arange(len(ids))):
            raise ValueError('Inverse token map does not reproduce identity')
        if not np.array_equal(ids[:len(prefix)], original_ids[:len(prefix)]) or not np.array_equal(
                ids[len(ids) - len(suffix):], original_ids[len(original_ids) - len(suffix):]):
            raise ValueError('Chat/instruction prefix or suffix changed')
        variants[name] = dict(prompt=raw, chat_text=text, prompt_ids=ids.tolist(),
            new_to_old=new_to_old.tolist(), old_to_new=old_to_new.tolist(),
            source_mask=source_mask[new_to_old].tolist(),
            original_doc_id=original_doc[new_to_old].tolist(),
            original_doc_token_offset=internal_offset[new_to_old].tolist())
    if variants['native']['prompt'] != prompt or variants['native']['chat_text'] != original_text:
        raise ValueError('Identity text reconstruction differs')
    return dict(prompt_length=len(original_ids), character_cuts=parts['character_cuts'],
        token_cuts=cuts, original_blocks=parts['blocks'], prefix=parts['prefix'], suffix=parts['suffix'],
        variants=variants)


def logit_readout(logits, targets, rivals=None):
    """Use original native rivals for every changed condition, never reselect."""
    if logits.ndim != 2 or len(targets) != len(logits):
        raise ValueError('Logits and target rows differ')
    targets = torch.as_tensor(targets, dtype=torch.long, device=logits.device)
    if rivals is None:
        competitors = logits.detach().clone()
        competitors.scatter_(1, targets[:, None], -torch.inf)
        rivals = competitors.argmax(-1)
    else:
        rivals = torch.as_tensor(rivals, dtype=torch.long, device=logits.device)
    if rivals.shape != targets.shape or torch.any(rivals == targets):
        raise ValueError('Each target requires a distinct frozen native competitor')
    actual = logits.gather(1, targets[:, None])[:, 0]
    margin = actual - logits.gather(1, rivals[:, None])[:, 0]
    logp = logits.log_softmax(-1).gather(1, targets[:, None])[:, 0]
    return dict(actual_id=targets.detach().cpu().numpy(), rival_id=rivals.detach().cpu().numpy(),
                actual_logp=logp.detach().cpu().numpy(), margin=margin.detach().cpu().numpy())


@torch.no_grad()
def complete_readout(model, prompt_ids, answer_ids, rivals=None):
    """Fresh full prefill; output row P+t-1 is causal for original target t."""
    if not prompt_ids or not answer_ids:
        raise ValueError('Prompt and whole answer must be nonempty')
    if rivals is not None and len(rivals) != len(answer_ids):
        raise ValueError('Frozen competitor count differs from whole answer')
    tokens = prompt_ids + answer_ids
    inputs = torch.tensor([tokens], dtype=torch.long, device=model.device)
    visits = [0] * len(model.model.layers)
    handles = []
    for index, layer in enumerate(model.model.layers):
        def observe(module, args, output, index=index):
            visits[index] += 1
        handles.append(layer.register_forward_hook(observe))
    started = time.perf_counter()
    try:
        hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[0]
    finally:
        for handle in handles:
            handle.remove()
    if visits != [1] * len(visits):
        raise ValueError('A complete native layer pass was not observed')
    predictions = hidden[len(prompt_ids) - 1:-1]
    if len(predictions) != len(answer_ids):
        raise ValueError('Target/prediction alignment differs')
    rows = []
    for start in range(0, len(answer_ids), 16):
        stop = min(start + 16, len(answer_ids))
        logits = model.lm_head(predictions[start:stop]).float()
        rows.append(logit_readout(logits, answer_ids[start:stop],
            None if rivals is None else rivals[start:stop]))
    arrays = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    arrays['query_position'] = np.arange(len(prompt_ids) - 1, len(tokens) - 1, dtype=np.int64)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite native output')
    audit = dict(full_model_forwards=1, layer_visits=visits, lm_head_projection_calls=len(rows),
        input_tokens=len(tokens), prompt_length=len(prompt_ids), answer_tokens=len(answer_ids),
        seconds=time.perf_counter() - started, target_timing='P+t-1; complete forced answer prefix',
        fresh_full_prefill=True, cached_KV_reused=False)
    return arrays, audit


def verify_readout(actual, reference, tolerance):
    """Identity/rival checks are exact; only numeric outputs have tolerance."""
    for key in ('actual_id', 'rival_id', 'query_position'):
        if not np.array_equal(actual[key], reference[key]):
            raise ValueError(f'Native identity differs: {key}')
    errors = {}
    for key in ('actual_logp', 'margin'):
        first, second = np.asarray(actual[key]), np.asarray(reference[key])
        if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError(f'Invalid native output: {key}')
        errors[key] = float(np.abs(first - second).max())
        if errors[key] > tolerance:
            raise ValueError(f'Native numeric gate failed: {key} {errors[key]} > {tolerance}')
    return dict(passed=True, tolerance=tolerance, max_absolute_errors=errors,
                identity_and_rivals_exact=True)


def order_fields(logp, margin):
    """Frozen descriptive readouts, all conditions and all individual tokens."""
    logp, margin = np.asarray(logp), np.asarray(margin)
    if logp.ndim != 2 or logp.shape[0] != 3 or margin.shape != logp.shape:
        raise ValueError('Exactly three aligned native/cyclic conditions required')
    if not np.isfinite(logp).all() or not np.isfinite(margin).all():
        raise ValueError('Nonfinite order responses')
    return dict(margin_range=np.ptp(margin, axis=0), actual_logp_range=np.ptp(logp, axis=0),
        signed_margin=margin[1:].mean(0) - margin[0],
        signed_actual_logp=logp[1:].mean(0) - logp[0],
        delta_margin=margin[1:] - margin[0], delta_actual_logp=logp[1:] - logp[0])
