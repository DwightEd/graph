"""Freeze first-only source-program readers on all natural QA test tokens.

Three source-selected checkpoints are averaged at each token. This is source
self-supervision, with no natural hallucination labels for fitting/selection.
Postcandidate observes the actual candidate; it is an offline measurement.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import source_bootstrap
from .data import PACKS, load_partition
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_case_analysis import CASE_IDS, case_rows, summarize_methods
from .run_source_selfsup import natural_batch, reader_inputs
from .source_readout import SourceCompatibilityReader


DEFAULT_OUTPUT = Path('outputs/first_timing_validation_20261008/natural_test')
TIMINGS = ('prechoice', 'postcandidate')


def load_models(args):
    models, checkpoints, program_scores = {}, {}, {}
    for timing in TIMINGS:
        program_scores[timing] = []
        for seed in args.seeds:
            name = f'{timing}_seed{seed}'
            path = args.fits / f'seed_{seed}' / f'{timing}_best.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model = SourceCompatibilityReader().to('cuda')
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            models[name], checkpoints[name] = model, checkpoint
            program_scores[timing].append(dict(np.load(path.with_name(f'{timing}_best_dev_scores.npz'))))
    return models, checkpoints, program_scores


def program_thresholds(args, program_scores):
    records = json.loads((args.fits / 'cache/postcandidate/records.json').read_text())
    compatible = [row['id'] for row in records if row['partition'] == 'dev' and row['label'] == 0]
    thresholds = {}
    for timing, scores in program_scores.items():
        values = np.asarray([[float(saved[key]) for key in compatible] for saved in scores])
        thresholds[timing + '_mean'] = float(np.quantile(values.mean(0), .95, method='higher'))
        for index, seed in enumerate(args.seeds):
            thresholds[f'{timing}_seed{seed}'] = float(np.quantile(values[index], .95, method='higher'))
    # This masking control is not calibrated by this threshold; report zero only.
    return thresholds


def validated_normalization(args, checkpoints):
    gates = [checkpoints[f'postcandidate_seed{seed}']['dev_metrics'] for seed in args.seeds]
    passed = sum(row['auroc'] >= .70 and row['source_swap']['direction_rate'] > .70 for row in gates)
    if passed < 2 or min(row['auroc'] for row in gates) < .60:
        raise ValueError('Postcandidate source-heldout three-seed gate failed')
    normalization = checkpoints[next(iter(checkpoints))]['normalization']
    for checkpoint in checkpoints.values():
        if any(not np.array_equal(checkpoint['normalization'][key], value)
               for key, value in normalization.items()):
            raise ValueError('Checkpoint normalizations differ')
    return normalization, gates, passed


@torch.no_grad()
def natural_predictions(args, models, normalization, pack, metadata):
    embedding = np.load(args.parent / 'source_selfsup/native_unembedding.npy', mmap_mode='r')
    predictions = {name: np.full(len(pack['token_id']), np.nan) for name in models}
    predictions['postcandidate_native_only_seed42'] = np.full(len(pack['token_id']), np.nan)
    for start in range(0, len(metadata['records']), args.batch_answers):
        records = metadata['records'][start:start + args.batch_answers]
        fields, selected = natural_batch(records, args.parent / 'capture', pack, normalization, embedding)
        post_fields = dict(fields, rows=fields['rows'] + 1)
        for name, model in models.items():
            inputs = fields if name.startswith('prechoice') else post_fields
            predictions[name][selected] = model(**inputs, variant='real').cpu().numpy()
        native, variant = reader_inputs(post_fields, 'native_only_real')
        predictions['postcandidate_native_only_seed42'][selected] = models['postcandidate_seed42'](
            **native, variant=variant).cpu().numpy()
        if start % 100 == 0:
            print(f'FIRST TRANSFER answers={start}/{len(metadata["records"])}', flush=True)
    for timing in TIMINGS:
        predictions[timing + '_mean'] = np.mean([predictions[f'{timing}_seed{seed}'] for seed in args.seeds], axis=0)
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError('Incomplete natural predictions')
    return predictions


def freeze_predictions(args, checkpoints, gates, passed, program_scores, pack, metadata,
                       predictions, started):
    np.savez(args.output / 'scores.npz', **predictions)
    write_json(args.output / 'THRESHOLDS.json', program_thresholds(args, program_scores))
    write_json(args.output / 'FREEZE.json', dict(primary='postcandidate_mean',
        timings=dict(prechoice='candidate supplied to reader only; row t',
                     postcandidate='observed candidate supplied to observer; row t+1; offline'),
        seeds=args.seeds, gate_passed_seeds=passed, source_gate=gates,
        natural_labels_used_for_fit=False, natural_labels_used_for_selection=False,
        natural_labels_used_for_threshold=False, historical_test_exposure=True,
        test_answers=len(metadata['records']), test_tokens=len(pack['token_id']),
        source_selfsupervised=True, new_observer_forwards=0, new_fits=0,
        native_only_control='paired-difference fields masked after normalization; not a native intervention',
        score_sha256=file_hash(args.output / 'scores.npz'), threshold_sha256=file_hash(args.output / 'THRESHOLDS.json'),
        checkpoint_sha256={name: file_hash(args.fits / f'seed_{checkpoint["seed"]}' /
                                         f'{checkpoint["timing"]}_best.pt')
                           for name, checkpoint in checkpoints.items()},
        test_pack_sha256=file_hash(PACKS / 'QA_test.npz'),
        code_sha256=file_hash(Path(__file__)), wall_seconds=time.time()-started))


def score(args):
    started = time.time()
    models, checkpoints, program_scores = load_models(args)
    normalization, gates, passed = validated_normalization(args, checkpoints)
    args.output.mkdir(parents=True, exist_ok=False)
    pack, metadata = load_partition('test')
    predictions = natural_predictions(args, models, normalization, pack, metadata)
    freeze_predictions(args, checkpoints, gates, passed, program_scores, pack, metadata,
                       predictions, started)


def evaluate(args):
    frozen = json.loads((args.output / 'FREEZE.json').read_text())
    if file_hash(args.output / 'scores.npz') != frozen['score_sha256']:
        raise ValueError('Frozen natural scores changed')
    if file_hash(args.output / 'THRESHOLDS.json') != frozen['threshold_sha256']:
        raise ValueError('Frozen natural thresholds changed')
    pack, metadata = load_partition('test')
    pack.update(evaluation_labels(args.cache, pack, metadata))
    predictions = dict(np.load(args.output / 'scores.npz'))
    thresholds = json.loads((args.output / 'THRESHOLDS.json').read_text())
    methods = {}
    for name, values in predictions.items():
        policies = dict(zero=0.)
        if name in thresholds:
            policies['program_correct95'] = thresholds[name]
        methods[name] = dict(scores=values, natural_label_fit=False, thresholds=policies,
            threshold_rules={key: 'fixed_zero' if key == 'zero' else 'source_program_correct95_strict_gt'
                             for key in policies})
    summaries, counts = summarize_methods(pack, metadata['records'], methods)
    old = dict(np.load(args.parent / 'unlabeled/test_scores.npz'))
    baseline = old['source_route_native_huber']
    write_json(args.output / 'RESULTS.json', dict(counts=counts, methods=summaries,
        bootstrap=source_bootstrap(pack, predictions['postcandidate_mean'], baseline, 300),
        limitations='observed-token cross-generator replay; exposed historical test; no online prevention claim'))
    export_cases(args, pack, metadata, methods)
    print(json.dumps({name: row['all_tokens'] for name, row in summaries.items()}), flush=True)


def export_cases(args, pack, metadata, methods):
    results = {}
    for record in metadata['records']:
        if record['id'] not in CASE_IDS:
            continue
        response = json.loads((args.cache / record['directory'] / 'response.json').read_text())
        rows = case_rows(record, response, pack, methods, {})
        with (args.output / f"case_{record['id']}.csv").open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        counts = {}
        for name, method in methods.items():
            region = slice(record['packed_start'], record['packed_stop'])
            labels, values = pack['labels'][region], method['scores'][region]
            counts[name] = {policy: dict(errors=int(labels.sum()),
                detected=int((values[labels == 1] > threshold).sum()),
                false_alarms=int((values[labels == 0] > threshold).sum()))
                for policy, threshold in method['thresholds'].items()}
        results[record['id']] = counts
    write_json(args.output / 'CASE_RESULTS.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('score', 'evaluate', 'all'))
    parser.add_argument('--fits', type=Path, default=Path('outputs/first_timing_validation_20261008'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--parent', type=Path, default=OUTPUT)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2026])
    parser.add_argument('--batch-answers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase in ('score', 'all'):
        score(args)
    if args.phase in ('evaluate', 'all'):
        evaluate(args)


if __name__ == '__main__':
    main()
