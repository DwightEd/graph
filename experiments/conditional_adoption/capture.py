"""Capture whole original answers, keeping native fixed-past query responses."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from experiments.decision_risk_flow.run import load_model
from experiments.token_backtrace.grounded_projection_data import DATASET, MODEL
from .mechanism import (GROUPS, OBJECTIVES, measure_query, norm_preserving_control,
                        output_grams, prefill_past, signed_responses)
from .mechanism_run import finite_check, write_json


RESPONSE_IDS = ('12219', '12216', '12297', '12294', '15604', '15600', '11907', '11904')
SOURCE_FIRST = Path('outputs/native_support_ragtruth_all/source_first_v1')
MEMORY_CAP_BYTES = 22 * 1024 ** 3


def prepare(output):
    """Reuse exact source-first prompt, response IDs and offsets, without labels."""
    source_manifest = json.loads((SOURCE_FIRST / 'manifest.json').read_text())
    indexed = {record['id']: record for record in source_manifest['records']}
    records = []
    for identity in RESPONSE_IDS:
        original = indexed[identity]
        cached = SOURCE_FIRST / original['directory']
        source = json.loads((SOURCE_FIRST / original['source_file']).read_text())
        response = json.loads((cached / 'response.json').read_text())
        records.append(dict(id=identity, source_id=original['source_id'], task=original['task'],
            generator=original['generator'], split=original['split'],
            prompt=source['prompt_with_source'], source_mask=source['source_mask'], response=response,
            prompt_length=len(source['prompt_with_source']), token_count=len(response['answer_ids']),
            old_source_cache=str(cached.resolve())))
    output.mkdir(parents=True, exist_ok=False)
    inputs = dict(model=MODEL, dataset=str(DATASET), records=records,
        source_manifest=str((SOURCE_FIRST / 'manifest.json').resolve()),
        roster_label_selection='previous error cases + smallest official normal response ID per source',
        official_roster='3 error answers, 5 normal answers; 11907 is official normal',
        labels_used_for_score=False, observer='Llama3.1-8B canonical observer replay',
        entropy_role='boundary/uncertainty diagnostic only', groups=GROUPS, objectives=OBJECTIVES,
        local_window=16, random_norm_seed=37)
    write_json(output / 'INPUTS.json', inputs)
    return inputs


def truncate_native_past(cache, position):
    """Future KV is excluded physically, independently for every original query."""
    pairs = [(layer.keys[:, :, :position].detach(), layer.values[:, :, :position].detach())
             for layer in cache.layers]
    return DynamicCache(pairs)


def check_canary(model, past, query_token, measured):
    check = finite_check(model, past, query_token, measured)
    assert check['identity_max_error'] <= 1e-3
    for point in check['central_differences']:
        passed = [absolute <= .02 or relative <= .02
                  for absolute, relative in zip(point['absolute_error'], point['relative_error'])]
        assert all(passed), f'Finite native suffix check failed: {point}'
    return check


def save_node(path, measured, randomized):
    values = {name: value.numpy() for name, value in measured.items() if isinstance(value, torch.Tensor)}
    values.pop('output_gram')
    values['response_random_norm'] = randomized.numpy()
    np.savez_compressed(path, **values)


def capture_answer(model, gram_cache, record, output, profile_tokens):
    directory = output / record['id']
    directory.mkdir()
    (directory / 'nodes').mkdir()
    tokens = record['prompt'] + record['response']['answer_ids']
    cache = prefill_past(model, tokens[:-1])
    count = min(profile_tokens, record['token_count']) if profile_tokens else record['token_count']
    rows, scalars, canary = [], [], None
    started = time.time()
    for target in range(count):
        position = record['prompt_length'] + target - 1
        past = truncate_native_past(cache, position)
        measured = measure_query(model, past, tokens[position], tokens[position + 1],
            record['source_mask'], record['prompt_length'], local_window=16, gram_cache=gram_cache)
        randomized_message = norm_preserving_control(measured['messages'], gram_cache, seed=37)
        randomized = signed_responses(measured['gradient'], randomized_message)
        maximum = float(measured['reconstruction_error'].max())
        reference = float(measured['messages'].abs().max())
        assert maximum <= 1e-3 + 5e-3 * reference, f'Native AV reconstruction failed: {maximum}'
        assert torch.cuda.max_memory_allocated() <= MEMORY_CAP_BYTES
        if target == 0:
            canary = check_canary(model, past, tokens[position], measured)
        save_node(directory / 'nodes' / f'{target:06d}.npz', measured, randomized)
        rows.append({name: measured[name].numpy() for name in
                     ('response', 'response_residual', 'response_ffn', 'projected_norm', 'attention_mass')})
        rows[-1]['response_random_norm'] = randomized.numpy()
        scalars.append({name: measured[name] for name in
                        ('actual_id', 'rival_id', 'query_position', 'actual_logp', 'margin', 'entropy')})
        if (target + 1) % 16 == 0 or target + 1 == count:
            print(json.dumps(dict(id=record['id'], captured=target + 1, total=count,
                                  seconds=time.time() - started)), flush=True)
    arrays = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
    arrays.update({name: np.asarray([row[name] for row in scalars]) for name in scalars[0]})
    np.savez_compressed(directory / 'arrays.npz', **arrays)
    result = dict(id=record['id'], source_id=record['source_id'], prompt_length=record['prompt_length'],
        token_count=count, full_token_count=record['token_count'], arrays=str((directory / 'arrays.npz').resolve()),
        nodes=str((directory / 'nodes').resolve()), canary=canary, seconds=time.time() - started)
    write_json(directory / 'COMPLETE.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--profile-tokens', type=int, default=0)
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output)
        return
    document = json.loads(args.inputs.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'PROTOCOL.json', dict(input_path=str(args.inputs.resolve()),
        input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(), groups=GROUPS, objectives=OBJECTIVES,
        source_fit=0, natural_label_fit=0, past_scope='detached original native KV; current query suffix autograd',
        simultaneous_site_derivative='layer/head sum is a joint local perturbation, not a conserved attribution',
        ffn_scope='within-layer FFN-mediated branch only; later native FFNs remain in all suffix gradients',
        entropy_role='saved diagnostic, excluded from risk', profile_tokens=args.profile_tokens))
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(document['model'])
    grams = output_grams(model)
    np.save(args.output / 'OUTPUT_GRAMS.npy', grams.numpy())
    records = document['records'][:1] if args.profile_tokens else document['records']
    started = time.time()
    results = [capture_answer(model, grams, record, args.output, args.profile_tokens) for record in records]
    write_json(args.output / 'MANIFEST.json', dict(status='DONE', records=results,
        input_path=str(args.inputs.resolve()), seconds=time.time() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), backward_calls=2 * sum(r['token_count'] for r in results),
        new_label_fits=0, full_answer_capture=not bool(args.profile_tokens)))


if __name__ == '__main__':
    main()
