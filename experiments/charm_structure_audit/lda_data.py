"""Read original node inputs. Optional prompt audit reads retained edges one answer at a time."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import read_json, original_parts
from .matching import surface_class
from .positions import merge_spans


def token_metadata(record, saved):
    labels = saved['gold'].astype(int)
    offsets = saved['offsets']
    text = str(saved['response'])
    count = len(labels)
    previous = np.r_[-1, labels[:-1]]
    run = np.zeros(count, int)
    for token in range(1, count):
        run[token] = run[token - 1] + 1 if labels[token - 1] else 0
    span_offset = np.full(count, -1)
    span_length = np.zeros(count, int)
    role = np.full(count, 'normal', dtype='<U16')
    for index, (start, end) in enumerate(merge_spans(saved['spans'])):
        span_offset[start:end] = np.arange(end - start)
        span_length[start:end] = end - start
        role[start:end] = 'continuation'
        role[start] = 'answer_first' if index == 0 else 'later_onset'
    tokens = [text[start:end] for start, end in offsets]
    return pd.DataFrame(dict(id=str(record['id']), source_id=str(record['source_id']),
        token=np.arange(count), gold=labels, text=tokens, surface=[surface_class(t) for t in tokens],
        position_bin=np.minimum(4, np.arange(count) * 5 // count), previous_gold=previous,
        past_run=run, past_run_bin=np.searchsorted([1, 2, 4, 8, 16], run, side='right'),
        role=role, span_offset=span_offset, span_length=span_length,
        text_valid=offsets[:, 1] > offsets[:, 0]))


def prompt_features(saved):
    """Per LLM channel: sum retained prompt weights; no semantic evidence assignment."""
    prompt = int(saved['prompt_length'])
    source, target = saved['edge_index']
    attributes = saved['edge_attr']
    size, channels = len(saved['gold']), saved['x'].shape[1]
    prompt_mass = np.zeros((size, channels))
    retained = np.zeros_like(prompt_mass)
    for start in range(0, len(source), 4096):
        stop = start + 4096
        destination = target[start:stop] - prompt
        weights = attributes[start:stop].astype(float)
        selected = source[start:stop] < prompt
        np.add.at(prompt_mass, destination[selected], weights[selected])
        np.add.at(retained, destination, weights)
    retained += saved['x'][prompt:]
    return prompt_mass, retained.mean(axis=1)


def read_split(records, prepared, include_prompt=False):
    blocks, tables, prompts = [], [], []
    for record in tqdm(records, desc='read LDA inputs', unit='answer'):
        path = Path(prepared) / 'graphs' / record['split'] / (str(record['id']) + '.npz')
        with np.load(path, allow_pickle=False) as saved:
            geometry = int(saved['layers']), int(saved['heads'])
            blocks.append(saved['x'][int(saved['prompt_length']):].astype(float))
            table = token_metadata(record, saved)
            if include_prompt:
                mass, retained = prompt_features(saved)
                prompts.append(mass)
                table['retained_mass_mean'] = retained
                table['prompt_mass_mean'] = mass.mean(axis=1)
            tables.append(table)
    prompt = np.concatenate(prompts) if include_prompt else None
    return np.concatenate(blocks), pd.concat(tables, ignore_index=True), prompt, geometry


def read_audit_inputs(args):
    recipe = read_json(Path(args.root) / 'node_only/training.json')
    parts = original_parts(args.prepared, recipe)
    data = {name: read_split(parts[name], args.prepared, args.lda_prompt)
            for name in ('fit', 'calibration', 'test')}
    geometry = data['fit'][3]
    assert all(value[3] == geometry for value in data.values()), 'Mixed LLM channel layouts'
    return data, parts


def read_saved_scores(root, window):
    """Reuse completed mixture/LDA scores; no arrays, new fitting or checkpoints."""
    from .lda_math import past_mean

    folder = Path(root) / 'audit_mixture'
    seed = read_json(folder / 'frozen.json')['seed_by_unlabelled_select_density']
    table = pd.read_csv(folder / 'test_tokens.csv.gz', dtype={'id': str, 'source_id': str}, keep_default_na=False)
    table = table[table.seed == seed].reset_index(drop=True)
    table['full'] = table.supervised_lda
    table['history_mean'] = past_mean(table.full, table.id.to_numpy(), window)
    table['current_increment'] = table.full - table.history_mean
    table['position_bin'] = np.minimum(4, (table.position * 5).astype(int))
    table['previous_gold'] = -1
    table['past_run'] = 0
    for _, group in table.groupby('id', sort=False):
        labels = group.gold.to_numpy()
        table.loc[group.index, 'previous_gold'] = np.r_[-1, labels[:-1]]
        run = np.zeros(len(group), int)
        for token in range(1, len(group)):
            run[token] = run[token - 1] + 1 if labels[token - 1] else 0
        table.loc[group.index, 'past_run'] = run
    table['past_run_bin'] = np.searchsorted([1, 2, 4, 8, 16], table.past_run, side='right')
    table['row'] = np.arange(len(table))
    return table
