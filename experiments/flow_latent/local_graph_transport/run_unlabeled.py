"""Freeze full-QA label-free source scores, then evaluate threshold policies.

Training a natural-label classifier is separate. These empirical reference
ranks and fixed graph optimizers never access the loaded training labels.
The official test was historically exposed; results are exploratory replication.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import (evaluate_all, source_bootstrap,
                                                           threshold_at_fpr)
from .data import PACKS, load_partition
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .unlabeled import (CHANNELS, HUBER_DELTA, PENALTY, equal_source_weights,
                        fit_reference, fit_weighted_cdf, score_answer)


METHODS = tuple(f'{family}_{variant}' for family in ('source', 'source_route')
                for variant in ('unary', 'native_huber', 'chain_huber', 'rewired_huber'))


def fit_scalar_reference(train, metadata, cache):
    """Read original fit observations, retaining exact ties; no label array access."""
    channels = {name: [] for name in CHANNELS}
    sources = []
    for record in metadata['records']:
        if record['partition'] != 'fit':
            continue
        region = slice(record['packed_start'], record['packed_stop'])
        targets = train['target'][region]
        with np.load(cache / record['directory'] / 'observations.npz') as observed:
            if not np.array_equal(observed['token_id'][targets], train['token_id'][region]):
                raise ValueError(f"{record['id']}: fit-reference token IDs differ")
            for name in CHANNELS:
                channels[name].append(observed[name][targets])
        sources.append(train['source_index'][region])
    values = [np.concatenate(channels[name]) for name in CHANNELS]
    return fit_reference(*values, np.concatenate(sources))


def require_complete_capture(output, train_metadata, test_metadata):
    """Require every official QA record, not a capture pilot or a partial cohort."""
    ledger = json.loads((output / 'CAPTURE_COMPLETE.json').read_text())
    protocol = json.loads((output / 'CAPTURE_PROTOCOL.json').read_text())
    expected = train_metadata['records'] + test_metadata['records']
    wanted = {row['id'] for row in expected}
    captured = {row['id'] for row in protocol['records']}
    if not ledger['full_qa_complete'] or ledger['completed_answers'] != len(expected):
        raise ValueError('Unlabeled scoring requires the completed full-QA capture')
    if captured != wanted or len(protocol['records']) != len(expected):
        raise ValueError('Capture and packed official-QA answer coverage differ')
    fit = {row['source_id'] for row in train_metadata['records'] if row['partition'] == 'fit'}
    dev = {row['source_id'] for row in train_metadata['records'] if row['partition'] == 'dev'}
    test = set(test_metadata['sources'])
    if fit & dev or fit & test or dev & test:
        raise ValueError('Fit/dev/test source partitions overlap')
    return dict(fit_sources=sorted(fit), dev_sources=sorted(dev), test_sources=sorted(test))


def answer_inputs(cache, capture, record, pack):
    """Read all original tokens before graph smoothing, then validate packed rows."""
    with np.load(cache / record['directory'] / 'observations.npz') as observed:
        values = {name: observed[name] for name in ('token_id', *CHANNELS)}
    with np.load(capture / record['id'] / 'arrays.npz') as archived:
        answer_ids = archived['answer_ids']
        attention = archived['local_attention'][0, 1:]
    region = slice(record['packed_start'], record['packed_stop'])
    targets = pack['target'][region]
    if not np.array_equal(answer_ids, values['token_id']):
        raise ValueError(f"{record['id']}: capture and original scalar token IDs differ")
    if len(answer_ids) != record['tokens'] or len(attention) != len(answer_ids):
        raise ValueError(f"{record['id']}: post-token graph rows do not cover all answer tokens")
    if not np.array_equal(values['token_id'][targets], pack['token_id'][region]):
        raise ValueError(f"{record['id']}: packed valid-token identities differ")
    return values, attention, targets, region


def score_partition(reference, pack, metadata, cache, capture):
    """Score every complete answer; valid-token filtering happens after optimization."""
    scores = {name: np.full(len(pack['token_id']), np.nan) for name in METHODS}
    ranks = {name: np.full(len(pack['token_id']), np.nan) for name in CHANNELS}
    answer_diagnostics = []
    invalid = 0
    for index, record in enumerate(metadata['records']):
        values, attention, targets, region = answer_inputs(cache, capture, record, pack)
        result = score_answer(reference, values['source_local'], values['source_full'],
                              values['raw_route'], attention)
        for name in METHODS:
            scores[name][region] = result['scores'][name][targets]
        for name in CHANNELS:
            ranks[name][region] = result['ranks'][name][targets]
        removed = record['tokens'] - len(targets)
        invalid += removed
        answer_diagnostics.append(dict(id=record['id'], source_id=record['source_id'],
            answer_tokens=record['tokens'], evaluated_tokens=len(targets),
            invalid_tokens=removed, **result['diagnostics']))
        if (index + 1) % 300 == 0:
            print(f"UNLABELED {metadata['split']} {index + 1}/{len(metadata['records'])}", flush=True)
    if not all(np.isfinite(value).all() for value in (*scores.values(), *ranks.values())):
        raise ValueError('Full-QA unlabeled predictions contain missing or nonfinite values')
    return scores, ranks, dict(invalid_tokens=invalid, answers=answer_diagnostics)


def weighted_reference_threshold(values, weights, quantile=.95):
    """Fit-mixture weighted quantile; it does not promise a normal-token FPR."""
    distinct, cumulative = fit_weighted_cdf(values, weights)
    return float(distinct[np.searchsorted(cumulative[1:], quantile, side='left')])


def freeze_predictions(args, directory, train, train_metadata, test_metadata,
                       train_scores, partitions, started):
    fit = ~train['development']
    weights = equal_source_weights(train['source_index'][fit])
    thresholds = dict(fixed95={name: .95 for name in METHODS},
        reference95={name: weighted_reference_threshold(values[fit], weights)
                     for name, values in train_scores.items()})
    write_json(directory / 'UNLABELED_THRESHOLDS.json', thresholds)
    hashes = {name: file_hash(directory / name) for name in
              ('reference.npz', 'train_scores.npz', 'test_scores.npz', 'train_ranks.npz',
               'test_ranks.npz', 'UNLABELED_THRESHOLDS.json')}
    write_json(directory / 'FREEZE.json', dict(status='all_predictions_frozen', hashes=hashes,
        partitions=partitions, fit_tokens=int(fit.sum()), dev_tokens=int((~fit).sum()),
        test_tokens=sum(row['packed_stop'] - row['packed_start'] for row in test_metadata['records']),
        train_answers=len(train_metadata['records']), test_answers=len(test_metadata['records']),
        natural_labels_used_for_reference_or_scores=False, test_labels_used=False,
        supervised_prediction_input=False, primary='source_native_huber', primary_unary='source_unary',
        source_route_status='fixed_secondary_not_label_selected', penalty=PENALTY,
        huber_delta=HUBER_DELTA, historical_test_exposure=True,
        timing='offline; post-token native local graph; original-token scalar contrast',
        fit_reference_scope='fit sources only; equal-source valid-token weights; mixed truth population',
        fixed95_rule='strict score>.95; ranks are not factual probabilities',
        reference95_rule='strict score>equal-source fit-mixture 95th percentile; not normal FPR',
        pack_hashes={f'QA_{split}.{suffix}': file_hash(PACKS / f'QA_{split}.{suffix}')
                     for split in ('train', 'test') for suffix in ('npz', 'json')},
        code_hashes={path.name: file_hash(path) for path in (directory / 'code_snapshot').iterdir()},
        capture_protocol_sha256=file_hash(args.output / 'CAPTURE_PROTOCOL.json'),
        wall_seconds=time.time() - started))


def score(args):
    started = time.time()
    train, train_metadata = load_partition('train')
    test, test_metadata = load_partition('test')
    partitions = require_complete_capture(args.output, train_metadata, test_metadata)
    directory = args.output / 'unlabeled'
    directory.mkdir(exist_ok=False)
    snapshot = directory / 'code_snapshot'
    snapshot.mkdir()
    for name in ('run_unlabeled.py', 'unlabeled.py', 'smooth.py', 'data.py'):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    reference = fit_scalar_reference(train, train_metadata, args.cache)
    np.savez(directory / 'reference.npz', **reference)
    predictions = {}
    for split, pack, metadata in (('train', train, train_metadata), ('test', test, test_metadata)):
        values, ranks, diagnostics = score_partition(reference, pack, metadata,
                                                     args.cache, args.output / 'capture')
        np.savez(directory / f'{split}_scores.npz', **values)
        np.savez(directory / f'{split}_ranks.npz', **ranks)
        write_json(directory / f'{split}_diagnostics.json', diagnostics)
        predictions[split] = values
    freeze_predictions(args, directory, train, train_metadata, test_metadata,
                       predictions['train'], partitions, started)
    print(f'UNLABELED frozen train={len(train["token_id"])} test={len(test["token_id"])}', flush=True)


def verify_freeze(directory):
    frozen = json.loads((directory / 'FREEZE.json').read_text())
    for name, expected in frozen['hashes'].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Frozen unlabeled artifact changed before evaluation: {name}')
    for name, expected in frozen['pack_hashes'].items():
        if file_hash(PACKS / name) != expected:
            raise ValueError(f'Frozen official pack changed before evaluation: {name}')
    for name, expected in frozen['code_hashes'].items():
        if file_hash(directory / 'code_snapshot' / name) != expected:
            raise ValueError(f'Frozen unlabeled execution snapshot changed: {name}')
    return frozen


def policy_metrics(pack, scores, thresholds, rule, label_assisted):
    """Reuse official localization metrics while correcting inherited threshold metadata."""
    metrics = evaluate_all(pack, scores, thresholds)
    for values in metrics.values():
        values.update(threshold_rule=rule, natural_labels_used_for_score_fit=False,
                      labels_used_for_threshold=label_assisted)
    return metrics


def evaluate_policies(directory, train, test, train_scores, test_scores):
    dev_mask = train['development']
    dev = {name: value[dev_mask] for name, value in train.items()}
    dev_scores = {name: value[dev_mask] for name, value in train_scores.items()}
    policies = json.loads((directory / 'UNLABELED_THRESHOLDS.json').read_text())
    policies['dev_normal95'] = {name: threshold_at_fpr(dev['labels'], values)
                                for name, values in dev_scores.items()}
    rules = dict(fixed95='fixed_score_.95_strict_gt',
        reference95='equal_source_fit_mixture_95th_percentile_strict_gt',
        dev_normal95='label_assisted_dev_normal_token_95th_percentile_strict_gt')
    results = {}
    for policy, thresholds in policies.items():
        write_json(directory / f'dev_metrics_{policy}.json', policy_metrics(
            dev, dev_scores, thresholds, rules[policy], policy == 'dev_normal95'))
        results[policy] = policy_metrics(test, test_scores, thresholds,
                                        rules[policy], policy == 'dev_normal95')
        write_json(directory / f'test_metrics_{policy}.json', results[policy])
    write_json(directory / 'ALL_THRESHOLDS.json', dict(policies=policies,
        score_fit_labels_used=False, dev_normal95_is_label_assisted=True,
        test_labels_used_for_threshold=False, no_method_selection=True))
    return results, policies, rules


def bootstrap_controls(test, scores, repeats):
    """Compare actual local edges with unary, mass-matched chain and rewiring."""
    comparisons = {}
    for family in ('source', 'source_route'):
        candidate = scores[f'{family}_native_huber']
        for control in ('unary', 'chain_huber', 'rewired_huber'):
            baseline = scores[f'{family}_{control}']
            result = source_bootstrap(test, candidate, baseline, repeats)
            comparisons[f'{family}_native_minus_{control}'] = result
    return comparisons


def evaluate_generators(test, metadata, scores, policies, rules):
    results = {}
    for generator in sorted({row['generator'] for row in metadata['records']}):
        indices = np.concatenate([np.arange(row['packed_start'], row['packed_stop'])
                                  for row in metadata['records'] if row['generator'] == generator])
        selected = {name: value[indices] for name, value in test.items()}
        predictions = {name: value[indices] for name, value in scores.items()}
        results[generator] = {policy: policy_metrics(selected, predictions, thresholds,
            rules[policy], policy == 'dev_normal95') for policy, thresholds in policies.items()}
    return results


def evaluate(args):
    started = time.time()
    directory = args.output / 'unlabeled'
    frozen = verify_freeze(directory)
    train, _ = load_partition('train')
    test, metadata = load_partition('test')
    with np.load(directory / 'train_scores.npz') as saved:
        train_scores = dict(saved)
    with np.load(directory / 'test_scores.npz') as saved:
        scores = dict(saved)
    # This is the first function that requests test annotations after FREEZE.
    test.update(evaluation_labels(args.cache, test, metadata))
    metrics, policies, rules = evaluate_policies(directory, train, test, train_scores, scores)
    comparisons = bootstrap_controls(test, scores, args.bootstrap)
    write_json(directory / 'bootstrap.json', comparisons)
    write_json(directory / 'test_by_generator.json', evaluate_generators(
        test, metadata, scores, policies, rules))
    write_json(directory / 'COMPLETE.json', dict(status='evaluated',
        fit_tokens=frozen['fit_tokens'], dev_tokens=frozen['dev_tokens'],
        test_tokens=frozen['test_tokens'], test_answers=frozen['test_answers'],
        natural_label_fits=0, reference_fits=3, methods=len(METHODS),
        test_labels_used_for_score_or_threshold_selection=False,
        dev_normal95_is_label_assisted=True, historical_test_exposure=True,
        evaluation_wall_seconds=time.time() - started))
    print(json.dumps({name: dict(auroc=row['auroc'], ap=row['ap'],
        within_answer=row['within_answer_auroc']) for name, row in metrics['fixed95'].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('score', 'evaluate', 'all'), nargs='?', default='all')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--bootstrap', type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase in ('score', 'all'):
        score(args)
    if args.phase in ('evaluate', 'all'):
        evaluate(args)


if __name__ == '__main__':
    main()
