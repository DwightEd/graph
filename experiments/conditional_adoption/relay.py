"""Four-condition native source amplification and strict-past KV relay.

Source AV is increased at one fixed layer. All later operations, including
past keys and values in the donor, run natively. Effects are not truth labels.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from experiments.decision_risk_flow.run import load_model
from .mechanism import prefill_past


LAYER = 15
AMPLITUDE = .25
CONDITIONS = ('D', 'A', 'B', 'C')
INPUTS = Path('outputs/conditional_adoption_20261008/complete_roster/INPUTS.json')
NATIVE = Path('outputs/conditional_adoption_20261008/complete_capture')
MEMORY_CAP_BYTES = 22 * 1024 ** 3


def strict_past(cache, position):
    """Return independent mutable cache metadata, with only keys strictly before q."""
    return DynamicCache((layer.keys[:, :, :position].detach(),
                         layer.values[:, :, :position].detach()) for layer in cache.layers)


def group_mask(source_mask, prompt_length, query_positions, key_count, device, group):
    positions = torch.arange(key_count, device=device)
    source = torch.zeros(key_count, device=device, dtype=torch.bool)
    source_count = min(prompt_length, key_count)
    source[:source_count] = torch.tensor(source_mask[:source_count], device=device)
    selected = source if group == 'source' else (positions < prompt_length) & ~source
    return selected[None, :] & (positions[None, :] < query_positions[:, None])


def native_group_message(attention, projected_query, rotary, cache, positions,
                         source_mask, prompt_length, attention_mask, group='source'):
    """Rebuild native all-key attention; preserve each physical head's source AV."""
    batch, length, _ = projected_query.shape
    heads = attention.q_proj.out_features // attention.head_dim
    query = projected_query.view(batch, length, heads, attention.head_dim).transpose(1, 2)
    query, _ = apply_rotary_pos_emb(query, query, *rotary)
    keys = repeat_kv(cache.layers[attention.layer_idx].keys, attention.num_key_value_groups)
    values = repeat_kv(cache.layers[attention.layer_idx].values, attention.num_key_value_groups)
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    scores = query.to(dtype) @ keys.to(dtype).transpose(-1, -2) * attention.scaling
    future = torch.arange(keys.shape[-2], device=positions.device)[None, :] > positions[:, None]
    scores = scores.masked_fill(future[None, None], -torch.inf)
    if attention_mask is not None:
        scores = scores + attention_mask[..., :keys.shape[-2]].to(dtype)
    weights = scores.softmax(-1)
    selected = group_mask(source_mask, prompt_length, positions, keys.shape[-2], keys.device, group)
    grouped = (weights * selected[None, None]) @ values.to(dtype)
    complete = weights @ values.to(dtype)
    return (grouped.transpose(1, 2).reshape(batch, length, -1),
            complete.transpose(1, 2).reshape(batch, length, -1))


@contextmanager
def amplify_source(model, source_mask, prompt_length, amplitude=AMPLITUDE, layer=LAYER,
                   active_positions=None, group='source', keep_message=False):
    """Add source AV before WO, then let native FFN and later blocks recompute."""
    assert group in ('source', 'other_prompt')
    attention = model.model.layers[layer].self_attn
    state = dict(reconstruction_max_error=0., modified_queries=0)

    def observe_attention(module, inputs, kwargs):
        state['kwargs'] = kwargs

    def observe_query(module, inputs, output):
        state['query'] = output

    def inject_message(module, inputs):
        kwargs = state['kwargs']
        positions = kwargs['cache_position']
        message, complete = native_group_message(attention, state['query'],
            kwargs['position_embeddings'], kwargs['past_key_values'], positions,
            source_mask, prompt_length, kwargs['attention_mask'], group)
        error = float((complete - inputs[0]).abs().max())
        state['reconstruction_max_error'] = max(state['reconstruction_max_error'], error)
        active = positions >= prompt_length - 1
        if active_positions is not None:
            active &= torch.isin(positions, torch.tensor(active_positions, device=positions.device))
        state['modified_queries'] += int(active.sum()) if amplitude else 0
        if keep_message:
            state['message'] = message.detach().cpu()
        added = amplitude * message * active[None, :, None]
        return (inputs[0] + added.to(inputs[0].dtype),)

    handles = [attention.register_forward_pre_hook(observe_attention, with_kwargs=True),
               attention.q_proj.register_forward_hook(observe_query),
               attention.o_proj.register_forward_pre_hook(inject_message)]
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def prefill_amplified(model, tokens, source_mask, prompt_length, amplitude=AMPLITUDE,
                      layer=LAYER, chunk_size=128, active_positions=None):
    cache = None
    calls = 0
    with amplify_source(model, source_mask, prompt_length, amplitude, layer,
                        active_positions=active_positions) as audit:
        for start in range(0, len(tokens), chunk_size):
            ids = torch.tensor([tokens[start:start + chunk_size]], device=model.device)
            output = model.model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            calls += 1
    return cache, dict(reconstruction_max_error=audit['reconstruction_max_error'],
                       modified_queries=audit['modified_queries'], forwards=calls)


@torch.no_grad()
def query_response(model, cache, position, query_token, actual_id, rival_id,
                   source_mask, prompt_length, amplitude=0., layer=LAYER, group='source'):
    past = strict_past(cache, position)
    ids = torch.tensor([[query_token]], device=model.device)
    with amplify_source(model, source_mask, prompt_length, amplitude, layer,
                        group=group, keep_message=True) as audit:
        output = model.model(input_ids=ids, past_key_values=past, use_cache=True)
        logits = model.lm_head(output.last_hidden_state[0, 0]).float()
    response = torch.stack((logits.log_softmax(-1)[actual_id],
                            logits[actual_id] - logits[rival_id])).cpu()
    diagnostics = dict(reconstruction_max_error=audit['reconstruction_max_error'],
        message=audit['message'][0, 0].reshape(-1, model.config.head_dim), forwards=1)
    return response, diagnostics


def four_conditions(model, native_cache, donor_cache, position, query_token,
                    actual_id, rival_id, source_mask, prompt_length, amplitude=AMPLITUDE,
                    layer=LAYER, native_response=None):
    """D/A use native past; B/C use all affected strictly past donor KV."""
    responses = {}
    diagnostics = {}
    for name, cache, dose in (('D', native_cache, 0.), ('A', native_cache, amplitude),
                              ('B', donor_cache, 0.), ('C', donor_cache, amplitude)):
        if name == 'D' and native_response is not None:
            responses[name] = native_response
            continue
        responses[name], diagnostics[name] = query_response(model, cache, position,
            query_token, actual_id, rival_id, source_mask, prompt_length, dose, layer)
    regret = torch.stack([responses['D'] - responses[name] for name in ('A', 'B', 'C')])
    return dict(response=torch.stack([responses[name] for name in CONDITIONS]),
                regret=regret, interaction=regret[2] - regret[0] - regret[1],
                diagnostics=diagnostics)


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def freeze_protocol(inputs_path, native, output, profile_tokens):
    document = json.loads(inputs_path.read_text())
    records = document['records'][:1] if profile_tokens else document['records']
    if not profile_tokens:
        complete = json.loads((native / 'MANIFEST.json').read_text())
        assert complete['status'] == 'DONE' and complete['full_answer_capture']
        assert {r['id'] for r in complete['records']} == {r['id'] for r in records}
    hashes = {str(inputs_path.resolve()): file_hash(inputs_path)}
    for record in records:
        path = native / record['id'] / 'arrays.npz'
        hashes[str(path.resolve())] = file_hash(path)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'code_snapshot'
    snapshot.mkdir()
    files = [Path(__file__), Path(__file__).with_name('mechanism.py'),
             Path('experiments/decision_risk_flow/run.py'), Path('experiments/decision_risk_flow/precision.py')]
    for path in files:
        shutil.copy2(path, snapshot / path.name)
    protocol = dict(input_path=str(inputs_path.resolve()), native_capture=str(native.resolve()),
        input_sha256=file_hash(inputs_path), input_hashes=hashes,
        layer=LAYER, amplitude=AMPLITUDE, conditions=CONDITIONS,
        actual_candidate='original response token', rival='frozen native best nonactual',
        intervention='native source-group AV before L15 WO; source mask only; predictor >= P-1',
        donor='all affected strictly past KV; current and downstream operations native',
        readout='D logp minus A/B/C logp; interaction=C-A-B', entropy_in_score=False,
        token_or_span_averaging=False, new_fits=0, historical_case_labels_exposed=True,
        source_amplification_proves_truth=False, profile_tokens=profile_tokens,
        planned_answer_ids=[r['id'] for r in records],
        code_sha256={str(path): file_hash(path) for path in files})
    write_json(output / 'PROTOCOL.json', protocol)
    return document, records


def native_arrays(native, record, count):
    with np.load(native / record['id'] / 'arrays.npz') as archived:
        arrays = {name: archived[name].copy() for name in
                  ('actual_id', 'rival_id', 'actual_logp', 'margin', 'query_position')}
    assert len(arrays['actual_id']) >= count
    assert np.array_equal(arrays['actual_id'][:count], record['response']['answer_ids'][:count])
    expected = np.arange(record['prompt_length'] - 1, record['prompt_length'] - 1 + count)
    assert np.array_equal(arrays['query_position'][:count], expected)
    return arrays


def result_arrays(rows, frozen, count):
    """Expose scalar logp regret separately from the fixed-rival margin response."""
    arrays = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
    arrays['interaction_response'] = arrays['interaction']
    arrays['interaction'] = arrays['interaction_response'][:, 0]
    for index, name in enumerate(('current', 'past', 'both')):
        arrays['regret_' + name] = arrays['regret'][:, index, 0]
    arrays['condition_actual_logp'] = arrays['response'][:, :, 0]
    arrays['condition_margin'] = arrays['response'][:, :, 1]
    arrays.update({name: values[:count] for name, values in frozen.items()})
    return arrays


def capture_answer(model, record, native, output, profile_tokens):
    directory = output / record['id']
    directory.mkdir()
    tokens = record['prompt'] + record['response']['answer_ids']
    count = min(profile_tokens, record['token_count']) if profile_tokens else record['token_count']
    frozen = native_arrays(native, record, count)
    native_cache = prefill_past(model, tokens[:-1])
    donor_cache, donor_audit = prefill_amplified(model, tokens[:-1], record['source_mask'],
                                               record['prompt_length'])
    assert donor_audit['reconstruction_max_error'] <= 1e-3
    first_position = record['prompt_length'] - 1
    guard, guard_audit = query_response(model, native_cache, first_position, tokens[first_position],
        int(frozen['actual_id'][0]), int(frozen['rival_id'][0]), record['source_mask'],
        record['prompt_length'])
    archived_first = torch.tensor([frozen['actual_logp'][0], frozen['margin'][0]], dtype=torch.float32)
    identity_error = float((guard - archived_first).abs().max())
    assert identity_error <= 1e-3, f'{record["id"]}: native identity {identity_error}'
    rows = []
    errors = []
    started = time.time()
    for target in range(count):
        position = record['prompt_length'] + target - 1
        baseline = torch.tensor([frozen['actual_logp'][target], frozen['margin'][target]], dtype=torch.float32)
        result = four_conditions(model, native_cache, donor_cache, position, tokens[position],
            int(frozen['actual_id'][target]), int(frozen['rival_id'][target]), record['source_mask'],
            record['prompt_length'], native_response=baseline)
        maximum = max(d['reconstruction_max_error'] for d in result['diagnostics'].values())
        assert maximum <= 1e-3, f'{record["id"]}: source AV reconstruction {maximum}'
        errors.append(dict(target=target, identity=identity_error if target == 0 else None,
                           identity_checked=target == 0,
                           reconstruction=maximum))
        rows.append(dict(response=result['response'].numpy(), regret=result['regret'].numpy(),
            interaction=result['interaction'].numpy(),
            source_message=result['diagnostics']['A']['message'].numpy()))
        assert torch.cuda.max_memory_allocated() <= MEMORY_CAP_BYTES
        if (target + 1) % 32 == 0 or target + 1 == count:
            print(json.dumps(dict(id=record['id'], measured=target + 1, total=count,
                                  seconds=time.time() - started)), flush=True)
    arrays = result_arrays(rows, frozen, count)
    np.savez_compressed(directory / 'arrays.npz', **arrays)
    forwards = 2 * ((len(tokens) - 2) // 128 + 1) + 3 * count + 1
    audit = dict(id=record['id'], source_id=record['source_id'], token_count=count,
        full_token_count=record['token_count'], donor_audit=donor_audit, query_errors=errors,
        native_guard=dict(response=guard.tolist(), archive=archived_first.tolist(),
                          reconstruction_max_error=guard_audit['reconstruction_max_error']),
        forwards=forwards, seconds=time.time() - started,
        baseline='native archive at all targets; freshly verified at first target',
        arrays=str((directory / 'arrays.npz').resolve()), label_files_opened=False)
    write_json(directory / 'COMPLETE.json', audit)
    del native_cache, donor_cache
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=INPUTS)
    parser.add_argument('--capture', '--native', dest='capture', type=Path, default=NATIVE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-tokens', type=int, default=0)
    args = parser.parse_args()
    document, records = freeze_protocol(args.inputs, args.capture, args.output, args.profile_tokens)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(document['model'])
    started = time.time()
    results = [capture_answer(model, record, args.capture, args.output, args.profile_tokens) for record in records]
    write_json(args.output / 'MANIFEST.json', dict(status='DONE', records=results,
        full_answer_capture=not bool(args.profile_tokens), conditions=CONDITIONS,
        layer=LAYER, amplitude=AMPLITUDE, seconds=time.time() - started,
        new_native_forwards=sum(r['forwards'] for r in results), new_fits=0,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), entropy_in_score=False))


if __name__ == '__main__':
    main()
