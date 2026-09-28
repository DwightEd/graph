"""CPU readout of all saved natural generations; no prompt roles or truth labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

from .features import (candidate_ids, lens_readouts, routing_layer,
                       source_balanced_percentile, state_turns)

ROUTING_FIELDS = ('prompt_mass', 'prompt_address_js', 'weighted_address_js',
                  'history_mass', 'local_history_mass', 'history_age')
EVENT_CHANNELS = ('weighted_address_js', 'late_candidate_js',
                  'entropy_nats', 'surprisal_minus_entropy')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_readout_weights(model):
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(model / index['lm_head.weight'], framework='pt', device='cpu') as saved:
        unembedding = saved.get_tensor('lm_head.weight')
    with safe_open(model / index['model.norm.weight'], framework='pt', device='cpu') as saved:
        norm = saved.get_tensor('model.norm.weight').float().numpy()
    epsilon = json.loads((model / 'config.json').read_text())['rms_norm_eps']
    return unembedding, norm, epsilon


def candidate_lens(hidden, prompt, ids, weights, norm, epsilon):
    """Hidden[32] is already final-normalized in HuggingFace's saved state tuple."""
    selected = weights[torch.as_tensor(ids)].float().numpy()
    steps = len(ids)
    logits = np.empty((32, steps, ids.shape[1]), dtype=np.float32)
    for layer in range(1, 33):
        state = hidden[layer, prompt - 1:prompt + steps - 1].astype(np.float32)
        if layer != 32:
            state = state / np.sqrt(np.mean(state ** 2, axis=-1, keepdims=True) + epsilon)
            state *= norm
        logits[layer - 1] = np.einsum('td,tcd->tc', state, selected)
    return logits


def read_sample(samples, states, record, weights, norm, epsilon):
    with np.load(samples / record['trace'], allow_pickle=False) as saved:
        trace = {key: saved[key] for key in saved.files}
    with np.load(states / record['trace'], allow_pickle=False) as saved:
        hidden = saved['hidden']
        entropy = saved['logit_entropy'] * np.log(2.)
        np.testing.assert_array_equal(saved['token_ids'], trace['token_ids'])
        errors = {key: float(saved[key]) for key in ('max_logit_error', 'max_attention_error')}
    prompt = int(trace['prompt_length'])
    chosen = trace['token_ids'][prompt:]
    ids, valid = candidate_ids(trace['top_ids'], chosen)
    logits = candidate_lens(hidden, prompt, ids, weights, norm, epsilon)
    readouts = lens_readouts(logits, valid, ids, chosen)
    geometry = state_turns(hidden, prompt, len(chosen))
    routing = np.stack([routing_layer(layer, prompt, trace['special_mask'])
                        for layer in trace['attention']])
    # Averages summarize the pilot; the physical layer/head axes remain in NPZ.
    features = {name: routing[16:, :, :, column].mean((0, 1))
                for column, name in enumerate(ROUTING_FIELDS)}
    features.update({name: value for name, value in readouts.items() if value.ndim == 1})
    features.update(geometry)
    features['entropy_nats'] = entropy
    features['surprisal_nats'] = trace['log_normalizer'] - trace['chosen_logit']
    features['surprisal_minus_entropy'] = features['surprisal_nats'] - entropy
    features['native_top1_margin'] = trace['top_logits'][:, 0] - trace['top_logits'][:, 1]
    errors['fp32_candidate_logit_max_error'] = float(np.max(np.abs(logits[-1, :, :5] - trace['top_logits'])))
    return trace, features, dict(routing=routing, candidate_ids=ids, candidate_valid=valid,
        candidate_logits=logits, candidate_probability=readouts['candidate_probability'],
        chosen_margin=readouts['chosen_margin']), errors


def extract(args):
    settings = json.loads((args.samples / 'settings.json').read_text())
    records = [json.loads(line) for line in (args.samples / 'samples.jsonl').read_text().splitlines()]
    weights, norm, epsilon = load_readout_weights(Path(settings['model']))
    tables, inventory = [], []
    for record in records:
        trace, features, arrays, errors = read_sample(args.samples, args.states, record, weights, norm, epsilon)
        prompt = int(trace['prompt_length'])
        table = pd.DataFrame(features)
        table['position'] = np.arange(len(table))
        table['trace'] = record['trace']
        table['source_id'] = str(record['source_id'])
        table['seed'] = record['seed']
        table['token'] = trace['token_text'][prompt:]
        table['valid'] = ~trace['special_mask'][prompt:]
        tables.append(table)
        np.savez_compressed(args.output / record['trace'], **arrays)
        inventory.append(dict(trace=record['trace'], source_id=record['source_id'],
                              seed=record['seed'], tokens=len(table), **errors))
        print('extracted', record['trace'], len(table), errors, flush=True)
    pd.concat(tables, ignore_index=True).to_csv(args.output / 'tokens.csv', index=False)
    pd.DataFrame(inventory).to_csv(args.output / 'inventory.csv', index=False)


def calibrate(output):
    table = pd.read_csv(output / 'tokens.csv', dtype={'source_id': str})
    for name in EVENT_CHANNELS:
        table[name + '_percentile'] = np.nan
    for source in table.source_id.unique():
        selected = table.source_id == source
        reference = table[(~selected) & table.valid & (table.position > 0)]
        for name in EVENT_CHANNELS:
            table.loc[selected, name + '_percentile'] = source_balanced_percentile(
                reference[name].to_numpy(), reference.source_id.to_numpy(),
                table.loc[selected, name].to_numpy())
    columns = [name + '_percentile' for name in EVENT_CHANNELS]
    table['event_union'] = (table[columns].max(axis=1) >= .9) & table.valid & (table.position > 0)
    for name in EVENT_CHANNELS:
        table[name + '_event'] = (table[name + '_percentile'] >= .9) & table.valid & (table.position > 0)
    table['episode_start'] = False
    for _, rows in table.groupby('trace', sort=False):
        events = rows.event_union.to_numpy()
        table.loc[rows.index, 'episode_start'] = events & ~np.r_[False, events[:-1]]
    table.to_csv(output / 'events.csv', index=False)
    valid = table[table.valid]
    coverage = valid.groupby('trace')[['event_union', 'episode_start']].agg(['sum', 'count'])
    coverage.to_csv(output / 'coverage.csv')
    write_json(output / 'frozen.json', dict(stage='all_annotation_free_features_and_events_saved',
        sources=int(table.source_id.nunique()), answers=int(table.trace.nunique()),
        tokens=len(table), valid_tokens=len(valid), event_tokens=int(valid.event_union.sum()),
        event_fraction=float(valid.event_union.mean()), events_are_not_error_predictions=True,
        labels_used=False, prompt_roles_used=False, manual_candidates_used=False,
        reference='other three sources; unlabeled, mixed correctness; source-balanced CDF',
        event_channels=EVENT_CHANNELS, percentile_threshold=.9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'protocol.json', dict(schema='role-free-flow-v1',
        samples=str(args.samples.resolve()), states=str(args.states.resolve()),
        final_norm='saved hidden[32] already normalized; never normalize twice',
        entropy='native pre-temperature full-vocabulary entropy in nats',
        candidate_lens='raw final-norm lens, final top5 plus chosen; not calibrated',
        routing_fields=ROUTING_FIELDS, event_channels=EVENT_CHANNELS,
        scope='CPU measurement and event proposals; no detector fit or new LLM forwards',
        manual_prompt_annotation=False, labels_used=False))
    extract(args)
    calibrate(args.output)


if __name__ == '__main__':
    main()
