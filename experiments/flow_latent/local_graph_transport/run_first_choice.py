"""Validate first-only BCE versus BCE+pair on source-heldout controls.

Prepare compact CPU inputs once; fit never reopens large observer captures.
Both objectives use the unchanged SourceCompatibilityReader and identical
inputs/initialization. Natural hallucination labels are never read here.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import read_fields
from .first_choice import compact_graph, first_batch, first_loss, first_metrics, first_records, source_groups
from .run_capture import OUTPUT, write_json
from .run_source_selfsup import normalize_fields
from .source_readout import SourceCompatibilityReader


DEFAULT_OUTPUT = Path('outputs/first_choice_validation_20261008')
FIELD_NAMES = ('node_fields', 'boundary_fields', 'value_fields', 'local_attention',
               'indices', 'valid', 'scalars', 'candidate_vectors')


def prepare(args):
    """Copy normalized first receiver/sender coordinates into a compact cache."""
    started = time.time()
    choices = json.loads((args.choices / 'choices.json').read_text())
    records = first_records(choices)
    normalization = dict(np.load(args.parent / 'fits/normalization.npz'))
    embedding = np.load(args.parent / 'native_unembedding.npy', mmap_mode='r')
    directory = args.output / 'cache'
    directory.mkdir(parents=True, exist_ok=False)
    parts, coverage = defaultdict(list), defaultdict(Counter)
    offset = 0
    for index, record in enumerate(records):
        fields = normalize_fields(read_fields(args.parent / 'capture' / record['id']), normalization)
        compact, receiver = compact_graph(fields, record['first_divergence'])
        for name, value in compact.items():
            parts[name].append(value)
        parts['candidate_vectors'].append(np.array(embedding[record['candidate_ids']], dtype=np.float32))
        record.update(compact_start=offset, compact_stop=offset + len(compact['node_fields']),
                      receiver_row=receiver, record_index=index)
        coverage[record['partition']][int(compact['valid'][receiver].sum())] += 1
        offset = record['compact_stop']
        if index % 200 == 0:
            print(f'PREPARE graphs={index}/{len(records)} rows={offset} wall={time.time()-started:.1f}', flush=True)
    save_cache(directory, parts, records, coverage, args, started)


def save_cache(directory, parts, records, coverage, args, started):
    sizes = {}
    for name, values in parts.items():
        axis = 1 if name == 'local_attention' else 0
        path = directory / f'{name}.npy'
        np.save(path, np.concatenate(values, axis=axis))
        sizes[name] = path.stat().st_size
    write_json(directory / 'records.json', records)
    counts = {split: dict(sources=len({r['source_id'] for r in records if r['partition'] == split}),
                         graphs=sum(r['partition'] == split for r in records),
                         valid_local_neighbor_counts=dict(coverage[split])) for split in ('fit', 'dev')}
    write_json(directory / 'COMPLETE.json', dict(counts=counts, bytes=sizes,
        parent=str(args.parent), choices=str(args.choices), natural_labels_used=False,
        new_observer_forwards=0, scalar_inputs='eight zeros, same as v2',
        compact_graph='receiver plus direct true senders; no recursive graph layer',
        normalization='unchanged parent source-only fit normalization', wall_seconds=time.time()-started))
    print(json.dumps(counts), flush=True)


def load_cache(args):
    directory = args.output / 'cache'
    records = json.loads((directory / 'records.json').read_text())
    arrays = {name: np.load(directory / f'{name}.npy', mmap_mode='r') for name in FIELD_NAMES}
    return records, arrays


@torch.no_grad()
def predictions(models, records, arrays, args):
    results = {objective: {} for objective in models}
    for model in models.values():
        model.eval()
    for start in range(0, len(records), args.batch_sources * 2):
        selected = records[start:start + args.batch_sources * 2]
        fields = first_batch(selected, arrays, args.device)
        for objective, model in models.items():
            risk = model(**fields, variant='real').cpu().numpy().reshape(-1, 2)
            results[objective].update({record['id']: risk[index] for index, record in enumerate(selected)})
    return results


def initialize_models(args, seed):
    models, optimizers = {}, {}
    for objective in args.objectives:
        torch.manual_seed(seed)
        model = SourceCompatibilityReader().to(args.device)
        models[objective] = model
        optimizers[objective] = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    return models, optimizers


def train_epoch(models, optimizers, groups, arrays, args, seed, epoch):
    """All sources have four first candidates and equal total loss weight."""
    order = np.random.default_rng(seed + epoch).permutation(len(groups))
    losses = defaultdict(list)
    for start in range(0, len(order), args.batch_sources):
        selected = [record for index in order[start:start + args.batch_sources] for record in groups[index]]
        fields = first_batch(selected, arrays, args.device)
        for objective, model in models.items():
            model.train()
            torch.manual_seed(seed + epoch * 100000 + start)
            optimizers[objective].zero_grad(set_to_none=True)
            risk = model(**fields, variant='real')
            loss = first_loss(risk, objective)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizers[objective].step()
            losses[objective].append(float(loss.detach()))
    return {objective: float(np.mean(values)) for objective, values in losses.items()}


def save_checkpoint(path, model, epoch, seed, objective, metrics, args):
    normalization = dict(np.load(args.parent / 'fits/normalization.npz'))
    torch.save(dict(state_dict=model.state_dict(), normalization=normalization,
        epoch=epoch, seed=seed, objective=objective, variant='real', dev_metrics=metrics,
        label_origin='source_only_first_divergence_same_correct_prefix_choices'), path)


def save_epoch(directory, models, best, history, fit_scores, dev_scores, selected, development,
               epoch, seed, losses, args):
    metrics = {}
    for objective, model in models.items():
        fit_metric = first_metrics(selected, fit_scores[objective])
        dev_metric = first_metrics(development, dev_scores[objective])
        metrics[objective] = dict(fit=fit_metric, dev=dev_metric, train_objective=losses[objective])
        if dev_metric['auroc'] > best[objective]:
            best[objective] = dev_metric['auroc']
            save_checkpoint(directory / f'{objective}_best.pt', model, epoch, seed, objective, dev_metric, args)
            np.savez(directory / f'{objective}_best_dev_scores.npz', **dev_scores[objective])
        if epoch == args.epochs:
            save_checkpoint(directory / f'{objective}_last.pt', model, epoch, seed, objective, dev_metric, args)
            np.savez(directory / f'{objective}_last_dev_scores.npz', **dev_scores[objective])
            np.savez(directory / f'{objective}_last_fit_scores.npz', **fit_scores[objective])
    history.append(dict(epoch=epoch, metrics=metrics))
    write_json(directory / 'HISTORY.json', history)
    print(f'FIRST epoch={epoch} seed={seed} {json.dumps(metrics)}', flush=True)


def fit_seed(args, seed, records, arrays):
    started = time.time()
    directory = args.output / f'seed_{seed}'
    directory.mkdir(exist_ok=False)
    selected = [record for record in records if record['partition'] == 'fit']
    development = [record for record in records if record['partition'] == 'dev']
    groups = source_groups(selected)
    models, optimizers = initialize_models(args, seed)
    parameters = sum(parameter.numel() for parameter in next(iter(models.values())).parameters())
    write_json(directory / 'PROTOCOL.json', dict(seed=seed, objectives=args.objectives,
        epochs=args.epochs, learning_rate=args.learning_rate, batch_sources=args.batch_sources,
        parameter_count=parameters, selection='program-dev first AUROC; earliest tie',
        natural_labels_used=False, new_observer_forwards=0, graph_variant='real',
        pair_loss='mean softplus(risk_correct-risk_wrong), coefficient 1',
        source_minibatch='original and swapped always together'))
    best = {objective: -float('inf') for objective in models}
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(models, optimizers, groups, arrays, args, seed, epoch)
        fit_scores = predictions(models, selected, arrays, args)
        dev_scores = predictions(models, development, arrays, args)
        save_epoch(directory, models, best, history, fit_scores, dev_scores, selected,
                   development, epoch, seed, losses, args)
    write_json(directory / 'COMPLETE.json', dict(best_first_auroc=best, fits=len(models),
        natural_label_fits=0, new_observer_forwards=0, wall_seconds=time.time()-started))


def fit(args):
    records, arrays = load_cache(args)
    for seed in args.seeds:
        fit_seed(args, seed, records, arrays)


def evaluate(args):
    """Recompute source-dev scores for best and fixed-last frozen checkpoints."""
    records, arrays = load_cache(args)
    development = [record for record in records if record['partition'] == 'dev']
    report = {}
    for seed in args.seeds:
        directory = args.output / f'seed_{seed}'
        for objective in args.objectives:
            for selection in ('best', 'last'):
                checkpoint = torch.load(directory / f'{objective}_{selection}.pt', map_location=args.device,
                                        weights_only=False)
                model = SourceCompatibilityReader().to(args.device)
                model.load_state_dict(checkpoint['state_dict'])
                scores = predictions({objective: model}, development, arrays, args)[objective]
                saved = dict(np.load(directory / f'{objective}_{selection}_dev_scores.npz'))
                maximum_error = max(float(np.max(np.abs(scores[key] - saved[key]))) for key in scores)
                report[f'{seed}_{objective}_{selection}'] = dict(epoch=checkpoint['epoch'],
                    metrics=first_metrics(development, scores), frozen_score_max_error=maximum_error)
    write_json(args.output / 'SOURCE_DEV_REEVALUATION.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'fit', 'evaluate', 'all'))
    parser.add_argument('--parent', type=Path, default=OUTPUT / 'source_selfsup')
    parser.add_argument('--choices', type=Path, default=OUTPUT / 'source_choices_v2')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2026])
    parser.add_argument('--objectives', nargs='+', choices=('bce', 'bce_pair'), default=['bce', 'bce_pair'])
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--batch-sources', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    phases = dict(prepare=prepare, fit=fit, evaluate=evaluate)
    for phase in phases if args.phase == 'all' else [args.phase]:
        phases[phase](args)


if __name__ == '__main__':
    main()
