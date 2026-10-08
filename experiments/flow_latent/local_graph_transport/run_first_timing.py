"""First-candidate source controls at prechoice versus postcandidate nodes.

Postcandidate includes the observed candidate in the input and is an offline
detector measurement. These controls do not establish advance error prevention.
No later token or natural hallucination training label is read.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import read_fields
from .first_choice import compact_graph, first_loss
from .first_timing import timing_batch, timing_metrics, timing_records, timing_source_groups
from .run_capture import OUTPUT, write_json
from .run_first_choice import FIELD_NAMES
from .run_source_selfsup import normalize_fields
from .source_readout import SourceCompatibilityReader


TIMINGS = ('prechoice', 'postcandidate')
DEFAULT_OUTPUT = Path('outputs/first_timing_validation_20261008')


def append_compact(parts, records, base, fields, embedding, timing, offset, index):
    receiver = base['first_divergence'] + (timing == 'postcandidate')
    compact, target = compact_graph(fields, receiver)
    for name, value in compact.items():
        parts[name].append(value)
    parts['candidate_vectors'].append(np.array(embedding[[base['candidate_id']]], dtype=np.float32))
    record = dict(base, compact_start=offset, compact_stop=offset + len(compact['node_fields']),
                  receiver_row=target, record_index=index, observer_row=receiver)
    records.append(record)
    return record['compact_stop'], int(compact['valid'][target].sum())


def prepare(args):
    started = time.time()
    programs = json.loads((args.parent / 'programs.json').read_text())
    base_records = timing_records(programs)
    normalization = dict(np.load(args.parent / 'fits/normalization.npz'))
    embedding = np.load(args.parent / 'native_unembedding.npy', mmap_mode='r')
    args.output.mkdir(exist_ok=False)
    parts = {timing: defaultdict(list) for timing in TIMINGS}
    records = {timing: [] for timing in TIMINGS}
    coverage = {timing: defaultdict(Counter) for timing in TIMINGS}
    offsets = dict.fromkeys(TIMINGS, 0)
    for index, base in enumerate(base_records):
        directory = args.parent / 'capture' / base['id']
        with np.load(directory / 'arrays.npz') as arrays:
            candidate = arrays['answer_ids'][base['first_divergence']]
            receiver_position = arrays['query_positions'][base['first_divergence'] + 1]
            if int(candidate) != base['candidate_id'] or int(arrays['token_ids'][receiver_position]) != int(candidate):
                raise ValueError(f"{base['id']}: postcandidate token/row mismatch")
        fields = normalize_fields(read_fields(directory), normalization)
        for timing in TIMINGS:
            offsets[timing], neighbors = append_compact(parts[timing], records[timing], base,
                fields, embedding, timing, offsets[timing], index)
            coverage[timing][base['partition']][neighbors] += 1
        if index % 400 == 0:
            print(f'TIMING PREPARE captures={index}/{len(base_records)} wall={time.time()-started:.1f}', flush=True)
    save_cache(args, parts, records, coverage, started)


def save_cache(args, parts, records, coverage, started):
    for timing in TIMINGS:
        directory = args.output / 'cache' / timing
        directory.mkdir(parents=True)
        for name, values in parts[timing].items():
            axis = 1 if name == 'local_attention' else 0
            np.save(directory / f'{name}.npy', np.concatenate(values, axis=axis))
        write_json(directory / 'records.json', records[timing])
    counts = {split: dict(sources=len({row['source_id'] for row in records['prechoice'] if row['partition']==split}),
        candidates=sum(row['partition']==split for row in records['prechoice'])) for split in ('fit','dev')}
    write_json(args.output / 'CACHE_COMPLETE.json', dict(counts=counts,
        valid_neighbor_counts={timing:{split:dict(values) for split,values in groups.items()}
                               for timing,groups in coverage.items()},
        parent=str(args.parent), timings=dict(prechoice='row=t, past excludes candidate',
            postcandidate='row=t+1, observed candidate included; offline'),
        natural_labels_used=False, new_observer_forwards=0, scalar_inputs='eight zeros',
        normalization='unchanged parent source-only fit normalization', wall_seconds=time.time()-started))
    print(json.dumps(counts), flush=True)


def load_cache(args):
    records, arrays = {}, {}
    for timing in TIMINGS:
        directory = args.output / 'cache' / timing
        records[timing] = json.loads((directory / 'records.json').read_text())
        arrays[timing] = {name:np.load(directory / f'{name}.npy', mmap_mode='r') for name in FIELD_NAMES}
    return records, arrays


@torch.no_grad()
def predictions(model, records, arrays, args):
    model.eval()
    scores = {}
    for start in range(0, len(records), args.batch_sources * 4):
        selected = records[start:start + args.batch_sources * 4]
        fields = timing_batch(selected, arrays, args.device)
        risk = model(**fields, variant='real').cpu().numpy()
        scores.update({record['id']:risk[index] for index,record in enumerate(selected)})
    return scores


def train_epoch(models, optimizers, groups, arrays, args, epoch):
    order = np.random.default_rng(args.seed + epoch).permutation(len(groups['prechoice']))
    losses = defaultdict(list)
    for start in range(0, len(order), args.batch_sources):
        selected_sources = order[start:start + args.batch_sources]
        for timing, model in models.items():
            selected = [record for index in selected_sources for record in groups[timing][index]]
            fields = timing_batch(selected, arrays[timing], args.device)
            model.train()
            torch.manual_seed(args.seed + epoch * 100000 + start)
            optimizers[timing].zero_grad(set_to_none=True)
            loss = first_loss(model(**fields, variant='real'), 'bce_pair')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizers[timing].step()
            losses[timing].append(float(loss.detach()))
    return {timing:float(np.mean(values)) for timing,values in losses.items()}


def save_checkpoint(path, model, metrics, args, epoch, timing):
    torch.save(dict(state_dict=model.state_dict(), seed=args.seed, epoch=epoch, timing=timing,
        dev_metrics=metrics, normalization=dict(np.load(args.parent / 'fits/normalization.npz')),
        label_origin='source_only_first_actual_candidate_controls'), path)


def save_epoch(directory, models, partitions, arrays, best, history, losses, args, epoch):
    metrics = {}
    for timing, model in models.items():
        fit_scores = predictions(model, partitions[timing]['fit'], arrays[timing], args)
        dev_scores = predictions(model, partitions[timing]['dev'], arrays[timing], args)
        fit_metric = timing_metrics(partitions[timing]['fit'], fit_scores)
        dev_metric = timing_metrics(partitions[timing]['dev'], dev_scores)
        metrics[timing] = dict(fit=fit_metric, dev=dev_metric, train_objective=losses[timing])
        if dev_metric['auroc'] > best[timing]:
            best[timing] = dev_metric['auroc']
            save_checkpoint(directory / f'{timing}_best.pt', model, dev_metric, args, epoch, timing)
            np.savez(directory / f'{timing}_best_dev_scores.npz', **dev_scores)
        if epoch == args.epochs:
            save_checkpoint(directory / f'{timing}_last.pt', model, dev_metric, args, epoch, timing)
            np.savez(directory / f'{timing}_last_dev_scores.npz', **dev_scores)
            np.savez(directory / f'{timing}_last_fit_scores.npz', **fit_scores)
    history.append(dict(epoch=epoch, metrics=metrics))
    write_json(directory / 'HISTORY.json', history)
    print(f'TIMING epoch={epoch} {json.dumps(metrics)}', flush=True)


def fit(args):
    started = time.time()
    records, arrays = load_cache(args)
    directory = args.output / f'seed_{args.seed}'
    directory.mkdir(exist_ok=False)
    partitions = {timing:{split:[row for row in records[timing] if row['partition']==split]
                         for split in ('fit','dev')} for timing in TIMINGS}
    groups = {timing:timing_source_groups(partitions[timing]['fit']) for timing in TIMINGS}
    models, optimizers = {}, {}
    for timing in TIMINGS:
        torch.manual_seed(args.seed)
        models[timing] = SourceCompatibilityReader().to(args.device)
        optimizers[timing] = torch.optim.AdamW(models[timing].parameters(), lr=args.learning_rate, weight_decay=.01)
    write_json(directory / 'PROTOCOL.json', dict(seed=args.seed, epochs=args.epochs,
        learning_rate=args.learning_rate, objective='first-only BCE+pair, coefficient 1',
        batch_sources=args.batch_sources, parameter_count=sum(p.numel() for p in models['prechoice'].parameters()),
        source_minibatch='four source assignment/candidate controls together',
        selection='program-dev first AUROC; earliest tie', natural_labels_used=False,
        new_observer_forwards=0, postcandidate_timing='offline candidate already present'))
    best = dict.fromkeys(TIMINGS, -float('inf'))
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(models, optimizers, groups, arrays, args, epoch)
        save_epoch(directory, models, partitions, arrays, best, history, losses, args, epoch)
    write_json(directory / 'COMPLETE.json', dict(fits=2, best_dev_auroc=best,
        natural_label_fits=0, new_observer_forwards=0, wall_seconds=time.time()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'fit', 'all'))
    parser.add_argument('--parent', type=Path, default=OUTPUT / 'source_selfsup')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=.0003)
    parser.add_argument('--batch-sources', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    phases = dict(prepare=prepare, fit=fit)
    for phase in phases if args.phase == 'all' else [args.phase]:
        phases[phase](args)


if __name__ == '__main__':
    main()
