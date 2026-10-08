"""Freeze and validate an attention-conditioned routing readout on full QA.

Fixed 16 bins, unchanged .75/.25 blend and existing native Huber parameters.
Natural labels enter evaluation only. The historical test is exposed, so this
is an exploratory comparison rather than a new blind confirmation.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import evaluate_all, source_bootstrap
from .conditional_source import ATTENTION_BINS, conditional_rank, fit_conditional_reference
from .data import load_partition
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_case_analysis import CASE_IDS, summarize_methods
from .run_unlabeled import weighted_reference_threshold
from .unlabeled import equal_source_weights, graph_fields, local_weights, reference_rank


DEFAULT_OUTPUT = Path('outputs/conditional_route_validation_20261008')
METHODS = ('conditional_route_unary', 'conditional_source_route_unary',
           'conditional_source_route_native_huber')


def score_partition(reference, source_reference, pack, metadata, args):
    scores = {name: np.full(len(pack['token_id']), np.nan) for name in METHODS}
    for index, record in enumerate(metadata['records']):
        region = slice(record['packed_start'], record['packed_stop'])
        target = pack['target'][region]
        with np.load(args.cache / record['directory'] / 'observations.npz') as raw:
            values = {name: raw[name].copy() for name in
                      ('token_id', 'source_local', 'source_full', 'raw_route', 'raw_attention')}
        with np.load(args.parent / 'capture' / record['id'] / 'arrays.npz') as archived:
            attention = archived['local_attention'][0, 1:]
            if not np.array_equal(archived['answer_ids'], values['token_id']):
                raise ValueError(f"{record['id']}: capture identity mismatch")
        if not np.array_equal(pack['token_id'][region], values['token_id'][target]):
            raise ValueError(f"{record['id']}: packed identity mismatch")
        route = conditional_rank(reference, values['raw_route'].astype(np.float32),
                                 values['raw_attention'].astype(np.float32))
        source = .5 * sum(reference_rank(source_reference, name, values[name])
                          for name in ('source_local', 'source_full'))
        unary = .75 * source + .25 * route
        fields, _ = graph_fields(unary, dict(native=local_weights(attention)))
        scores['conditional_route_unary'][region] = route[target]
        scores['conditional_source_route_unary'][region] = unary[target]
        scores['conditional_source_route_native_huber'][region] = fields['native_huber'][target]
        if (index + 1) % 500 == 0:
            print(f"CONDITIONAL {metadata['split']} {index+1}/{len(metadata['records'])}", flush=True)
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError('Incomplete conditional scores')
    return scores


def score(args):
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    train, train_metadata = load_partition('train')
    test, test_metadata = load_partition('test')
    fit = ~train['development']
    reference = fit_conditional_reference(train['scalars'][fit, 3], train['scalars'][fit, 4],
                                           train['source_index'][fit])
    np.savez(args.output / 'reference.npz', **reference)
    source_reference = dict(np.load(args.parent / 'unlabeled/reference.npz'))
    predictions = {}
    for split, pack, metadata in (('train', train, train_metadata), ('test', test, test_metadata)):
        predictions[split] = score_partition(reference, source_reference, pack, metadata, args)
        np.savez(args.output / f'{split}_scores.npz', **predictions[split])
    weights = equal_source_weights(train['source_index'][fit])
    thresholds = {name: weighted_reference_threshold(values[fit], weights)
                  for name, values in predictions['train'].items()}
    write_json(args.output / 'THRESHOLDS.json', thresholds)
    write_json(args.output / 'FREEZE.json', dict(natural_labels_used_for_fit=False,
        natural_labels_used_for_selection=False, attention_bins=ATTENTION_BINS,
        source_blend=.75, conditional_route_blend=.25, penalty=.5, huber_delta=1.,
        timing='offline observed-token; post-token native local attention',
        historical_test_exposure=True, new_observer_forwards=0,
        source_reference=str(args.parent / 'unlabeled/reference.npz'),
        source_reference_sha256=file_hash(args.parent / 'unlabeled/reference.npz'),
        hashes={name: file_hash(args.output / name) for name in
                ('reference.npz', 'train_scores.npz', 'test_scores.npz', 'THRESHOLDS.json')},
        code_hashes={name: file_hash(Path(__file__).with_name(name)) for name in
                     ('conditional_source.py', 'run_conditional_source.py')},
        wall_seconds=time.time()-started))


def evaluate(args):
    frozen = json.loads((args.output / 'FREEZE.json').read_text())
    for name, expected in frozen['hashes'].items():
        if file_hash(args.output / name) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')
    train, _ = load_partition('train')
    test, metadata = load_partition('test')
    test.update(evaluation_labels(args.cache, test, metadata))
    thresholds = json.loads((args.output / 'THRESHOLDS.json').read_text())
    scores = dict(np.load(args.output / 'test_scores.npz'))
    old = dict(np.load(args.parent / 'unlabeled/test_scores.npz'))
    old_thresholds = json.loads((args.parent / 'unlabeled/UNLABELED_THRESHOLDS.json').read_text())['reference95']
    methods = {name: dict(scores=values, natural_label_fit=False,
               thresholds={'reference95': thresholds[name]},
               threshold_rules={'reference95': 'fit_mixture95_strict_gt'})
               for name, values in scores.items()}
    baseline = 'source_route_native_huber'
    methods[baseline] = dict(scores=old[baseline], natural_label_fit=False,
        thresholds={'reference95': old_thresholds[baseline]},
        threshold_rules={'reference95': 'fit_mixture95_strict_gt'})
    summaries, counts = summarize_methods(test, metadata['records'], methods)
    write_json(args.output / 'TEST_RESULTS.json', dict(counts=counts, methods=summaries,
        bootstrap=source_bootstrap(test, scores[METHODS[-1]], old[baseline], 300)))
    dev = train['development']
    dev_pack = {name: value[dev] for name, value in train.items()}
    dev_scores = {name: value[dev] for name, value in dict(np.load(args.output / 'train_scores.npz')).items()}
    write_json(args.output / 'DEV_RESULTS.json', evaluate_all(dev_pack, dev_scores, thresholds))
    export_cases(args, test, metadata, methods)
    print(json.dumps({name: result['all_tokens'] for name, result in summaries.items()}), flush=True)


def export_cases(args, pack, metadata, methods):
    results = {}
    for record in metadata['records']:
        if record['id'] not in CASE_IDS:
            continue
        region = slice(record['packed_start'], record['packed_stop'])
        response = json.loads((args.cache / record['directory'] / 'response.json').read_text())
        rows, counts = [], {}
        labels = pack['labels'][region]
        for name, method in methods.items():
            flag = method['scores'][region] > method['thresholds']['reference95']
            counts[name] = dict(errors=int(labels.sum()), detected=int(flag[labels == 1].sum()),
                                false_alarms=int(flag[labels == 0].sum()))
        for position in range(record['packed_start'], record['packed_stop']):
            token = pack['target'][position]
            row = dict(token_index=int(token), token_text=response['token_text'][token],
                       gold=int(pack['labels'][position]))
            for name, method in methods.items():
                row[name] = float(method['scores'][position])
                row[name + '_flag'] = int(row[name] > method['thresholds']['reference95'])
            rows.append(row)
        with (args.output / f"case_{record['id']}.csv").open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        results[record['id']] = counts
    write_json(args.output / 'CASE_RESULTS.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('score', 'evaluate', 'all'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--parent', type=Path, default=OUTPUT)
    parser.add_argument('--cache', type=Path, default=CACHE)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase in ('score', 'all'):
        score(args)
    if args.phase in ('evaluate', 'all'):
        evaluate(args)


if __name__ == '__main__':
    main()
