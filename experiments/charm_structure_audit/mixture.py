"""FIT raw x without labels -> freeze -> held-out diagnosis against gold and P.

Existing CHARM/GNN computations and previous experiments remain unchanged.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .data import read_json, write_json, save_scores, original_parts
from .mixture_math import prepare_coordinates, transform_values, fit_two, log_densities, raw_parameters


def read_nodes(record, prepared):
    path = Path(prepared) / 'graphs' / record['split'] / (str(record['id']) + '.npz')
    with np.load(path, allow_pickle=False) as saved:
        prompt = int(saved['prompt_length'])
        values = saved['x'][prompt:].astype(np.float64)
        geometry = int(saved['layers']), int(saved['heads'])
    return values, geometry


def fit_records(args):
    recipe = read_json(Path(args.root) / 'node_only/training.json')
    original = original_parts(args.prepared, recipe)
    # Only membership/identity survives here; positives and gold never enter fitting.
    parts = {}
    for name, records in original.items():
        parts[name] = [{key: record[key] for key in ('id', 'source_id', 'split')} for record in records]
    return parts


def fit_models(args, output, parts):
    blocks = [read_nodes(row, args.prepared)[0] for row in tqdm(parts['fit'], desc='read FIT x')]
    values = np.concatenate(blocks)
    del blocks
    write_json(output / 'fit_population.json', dict(answers=len(parts['fit']),
        sources=len({row['source_id'] for row in parts['fit']}), tokens=len(values), channels=values.shape[1]))
    coordinates, transform = prepare_coordinates(values, args.mixture_ridge)
    save_scores(output / 'transform.npz', **transform)
    seeds, states, history_rows = [], [], []
    standardized = (values - transform['center']) / transform['scale']
    for seed in tqdm(args.mixture_seeds, desc='unlabelled mixture starts'):
        initial = KMeans(n_clusters=2, n_init=1, max_iter=30, random_state=seed).fit_predict(standardized)
        state, history = fit_two(coordinates, transform, initial, args.mixture_iterations, progress=True)
        save_scores(output / f'model_{seed}.npz', **state)
        raw = raw_parameters(state, transform)
        save_scores(output / f'raw_parameters_{seed}.npz', **raw)
        history_rows.extend(dict(row, seed=seed) for row in history)
        seeds.append(seed)
        states.append(state)
    pd.DataFrame(history_rows).to_csv(output / 'fit_history.csv', index=False)
    return transform, seeds, states


def freeze_scores(args, output, parts, transform, seeds, states):
    """Score every held-out token before opening any gold array or checkpoint."""
    rows = []
    for split in ('select', 'calibration', 'test'):
        for record in tqdm(parts[split], desc='freeze ' + split):
            values, geometry = read_nodes(record, args.prepared)
            coordinates = transform_values(values, transform)
            arrays = dict(id=np.asarray(str(record['id'])), source_id=np.asarray(str(record['source_id'])))
            for seed, state in zip(seeds, states):
                one, two = log_densities(coordinates, state, transform)
                arrays['gaussian1_log_density'] = one
                arrays[f'mixture_log_density_{seed}'] = two
                arrays[f'component_log_odds_{seed}'] = coordinates @ state['coefficient'] + state['intercept']
                rows.append(dict(split=split, id=str(record['id']), source_id=str(record['source_id']),
                    seed=seed, tokens=len(values), gaussian1=float(one.mean()), mixture=float(two.mean()),
                    density_gain=float((two - one).mean())))
            save_scores(output / 'frozen_scores' / split / (str(record['id']) + '.npz'), **arrays)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'density_by_answer.csv', index=False)
    selected = frame[frame.split == 'select'].groupby(['seed', 'source_id']).mixture.mean()
    best_seed = int(selected.groupby('seed').mean().idxmax())
    write_json(output / 'frozen.json', dict(seed_by_unlabelled_select_density=best_seed,
        seeds=seeds, layers=geometry[0], heads=geometry[1], scope='all saved response x; no gold span selection',
        fit_reads='x and prompt_length only; no gold, predictions, embeddings, edge tensors or checkpoint',
        orientation='largest absolute raw coefficient positive; component identity is NOT truth identity',
        selection='mean SELECT log density, equal source weighting; no label-based choice'))


def load_parameters(output):
    with np.load(output / 'transform.npz') as saved:
        transform = {name: saved[name] for name in saved.files}
    frozen = read_json(output / 'frozen.json')
    states = {}
    for seed in frozen['seeds']:
        with np.load(output / f'model_{seed}.npz') as saved:
            states[seed] = {name: saved[name] for name in saved.files}
    return transform, states, frozen


def run_mixture(args):
    from .mixture_report import diagnose

    output = Path(args.output) if args.output else Path(args.root) / 'audit_mixture'
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() == Path(args.prepared).resolve():
        raise ValueError('Use an audit output directory, not the prepared input directory')
    parts = fit_records(args)
    protocol = dict(root=str(Path(args.root).resolve()), prepared=str(Path(args.prepared).resolve()),
                    seeds=args.mixture_seeds, ridge=args.mixture_ridge, iterations=args.mixture_iterations, parts=parts)
    existing = output / 'mixture_config.json'
    if existing.exists() and read_json(existing) != protocol:
        raise ValueError('Use a new --output for a different mixture protocol')
    write_json(existing, protocol)
    with threadpool_limits(limits=4):
        if args.mixture_stage != 'report' and not (output / 'frozen.json').exists():
            transform, seeds, states = fit_models(args, output, parts)
            freeze_scores(args, output, parts, transform, seeds, states)
        if args.mixture_stage == 'fit':
            print('Unlabelled fit and held-out scores frozen:', output, flush=True)
            return
        transform, states, frozen = load_parameters(output)
        diagnose(args, output, parts, transform, states, frozen)
    print('Mixture audit:', output, flush=True)
