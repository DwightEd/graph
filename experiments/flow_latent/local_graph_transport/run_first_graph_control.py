"""Frozen postcandidate reader without explicit local graph messages.

This is an input diagnostic, not a selected detector or a new fit. Native
residuals, source boundaries and values retain their original history content.
Natural labels are joined only after scores and source thresholds are frozen.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import source_bootstrap
from .data import PACKS, load_partition
from .first_timing import timing_batch, timing_metrics
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_case_analysis import summarize_methods
from .run_first_choice import FIELD_NAMES
from .run_first_transfer import export_cases, load_models, validated_normalization
from .run_source_selfsup import natural_batch, reader_inputs
from .source_readout import SourceCompatibilityReader


DEFAULT_OUTPUT = Path('outputs/first_timing_validation_20261008/no_local_control')
MEAN = 'postcandidate_mean_no_local'
NATIVE = 'postcandidate_native_only_seed42_no_local'


def remove_local_edges(fields):
    """Replace only edge weights, preserving tensors and the original mapping."""
    return dict(fields, local_attention=torch.zeros_like(fields['local_attention']))


def post_checkpoint(args, seed, device):
    path = args.fits / f'seed_{seed}' / 'postcandidate_best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = SourceCompatibilityReader().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def program(args):
    """Check source-heldout edge ablation and fix its own source threshold."""
    if (args.output / 'PROGRAM.json').exists():
        raise FileExistsError('Program diagnostic already exists')
    args.output.mkdir(parents=True, exist_ok=True)
    directory = args.fits / 'cache/postcandidate'
    records = json.loads((directory / 'records.json').read_text())
    records = [row for row in records if row['partition'] == 'dev']
    arrays = {name: np.load(directory / f'{name}.npy', mmap_mode='r') for name in FIELD_NAMES}
    scores, checks, metrics = {}, {}, {}
    for seed in args.seeds:
        model, _ = post_checkpoint(args, seed, 'cpu')
        saved = dict(np.load(args.fits / f'seed_{seed}/postcandidate_best_dev_scores.npz'))
        values, reference_errors, empty_errors, changed = {}, [], [], []
        for start in range(0, len(records), args.batch_sources * 4):
            selected = records[start:start + args.batch_sources * 4]
            fields = timing_batch(selected, arrays, 'cpu')
            control = remove_local_edges(fields)
            if any(control[key] is not value for key, value in fields.items() if key != 'local_attention'):
                raise AssertionError('Local edge control changed another input')
            reference = model(**fields, variant='real').numpy()
            result = model(**control, variant='real').numpy()
            valid = fields['valid'][fields['rows']].any(dim=-1).numpy()
            for index, record in enumerate(selected):
                values[record['id']] = float(result[index])
                reference_errors.append(abs(float(reference[index]) - float(saved[record['id']])))
                if not valid[index]:
                    empty_errors.append(abs(float(reference[index]) - float(result[index])))
                changed.append(abs(float(reference[index]) - float(result[index])))
        if max(empty_errors, default=0.) > 1e-6:
            raise AssertionError('Removing absent graph edges changed source risk')
        scores[f'seed{seed}'] = values
        checks[f'seed{seed}'] = dict(cpu_vs_saved_max_abs=max(reference_errors),
            empty_edge_max_abs=max(empty_errors), max_ablation_change=max(changed),
            zero_edge_candidates=len(empty_errors), candidates=len(records))
        metrics[f'seed{seed}'] = timing_metrics(records, values)
    mean = {row['id']: float(np.mean([scores[f'seed{seed}'][row['id']] for seed in args.seeds]))
            for row in records}
    metrics['mean'] = timing_metrics(records, mean)
    threshold = float(np.quantile([mean[row['id']] for row in records if row['label'] == 0],
                                  .95, method='higher'))
    np.savez(args.output / 'program_no_local_scores.npz', **{key: np.array([values[row['id']] for row in records])
                                                           for key, values in scores.items()})
    write_json(args.output / 'THRESHOLDS.json', {MEAN: threshold})
    write_json(args.output / 'PROGRAM.json', dict(checks=checks, metrics=metrics,
        source_dev_candidates=len(records), seeds=args.seeds, new_fits=0,
        changed_input='local_attention only; both native and paired-blocked weights set to zero',
        graph_control_is_not_history_deletion=True, natural_labels_used=False,
        checkpoint_sha256={f'seed{seed}': file_hash(args.fits / f'seed_{seed}/postcandidate_best.pt')
                           for seed in args.seeds},
        threshold_sha256=file_hash(args.output / 'THRESHOLDS.json'),
        code_sha256=file_hash(Path(__file__))))
    print(json.dumps(dict(checks=checks, mean_source_metrics=metrics['mean'])), flush=True)


@torch.no_grad()
def score(args):
    started = time.time()
    source_contract = json.loads((args.output / 'PROGRAM.json').read_text())
    if (args.output / 'scores.npz').exists():
        raise FileExistsError('Natural control scores already exist')
    for seed in args.seeds:
        if file_hash(args.fits / f'seed_{seed}/postcandidate_best.pt') != source_contract['checkpoint_sha256'][f'seed{seed}']:
            raise ValueError('Parent source checkpoint changed')
    if file_hash(args.output / 'THRESHOLDS.json') != source_contract['threshold_sha256']:
        raise ValueError('Source thresholds changed')
    models, checkpoints, _ = load_models(args)
    normalization, gates, passed = validated_normalization(args, checkpoints)
    pack, metadata = load_partition('test')
    embedding = np.load(args.parent / 'source_selfsup/native_unembedding.npy', mmap_mode='r')
    predictions = {name: np.full(len(pack['token_id']), np.nan) for name in (MEAN, NATIVE)}
    for start in range(0, len(metadata['records']), args.batch_answers):
        records = metadata['records'][start:start + args.batch_answers]
        fields, selected = natural_batch(records, args.parent / 'capture', pack, normalization, embedding)
        post = dict(fields, rows=fields['rows'] + 1)
        control = remove_local_edges(post)
        risks = [models[f'postcandidate_seed{seed}'](**control, variant='real').cpu().numpy()
                 for seed in args.seeds]
        predictions[MEAN][selected] = np.mean(risks, axis=0)
        native, variant = reader_inputs(control, 'native_only_real')
        predictions[NATIVE][selected] = models['postcandidate_seed42'](**native, variant=variant).cpu().numpy()
        if start % 100 == 0:
            print(f'NO LOCAL CONTROL answers={start}/{len(metadata["records"])}', flush=True)
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError('Incomplete natural control scores')
    np.savez(args.output / 'scores.npz', **predictions)
    write_json(args.output / 'FREEZE.json', dict(diagnostic_only=True,
        primary_method_unchanged='postcandidate_mean in natural_test',
        source_gate=gates, gate_passed_seeds=passed, seeds=args.seeds,
        natural_labels_used_for_fit=False, natural_labels_used_for_selection=False,
        natural_labels_used_for_threshold=False, historical_test_exposure=True,
        timing='row t+1, actual observed candidate; offline',
        changed_input='local_attention zeros only; residuals/boundaries/values unchanged',
        graph_control_is_not_history_deletion=True, new_fits=0, new_observer_forwards=0,
        native_control='paired-difference inputs masked after normalization; zero threshold only',
        score_sha256=file_hash(args.output / 'scores.npz'),
        threshold_sha256=file_hash(args.output / 'THRESHOLDS.json'),
        program_sha256=file_hash(args.output / 'PROGRAM.json'),
        checkpoint_sha256=source_contract['checkpoint_sha256'],
        test_pack_sha256=file_hash(PACKS / 'QA_test.npz'),
        code_sha256=file_hash(Path(__file__)), test_answers=len(metadata['records']),
        test_tokens=len(pack['token_id']), wall_seconds=time.time()-started))


def evaluate(args):
    frozen = json.loads((args.output / 'FREEZE.json').read_text())
    for path, key in [('scores.npz', 'score_sha256'), ('THRESHOLDS.json', 'threshold_sha256'),
                      ('PROGRAM.json', 'program_sha256')]:
        if file_hash(args.output / path) != frozen[key]:
            raise ValueError(f'Frozen control artifact changed: {path}')
    pack, metadata = load_partition('test')
    pack.update(evaluation_labels(args.cache, pack, metadata))
    predictions = dict(np.load(args.output / 'scores.npz'))
    threshold = json.loads((args.output / 'THRESHOLDS.json').read_text())[MEAN]
    methods = {name: dict(scores=values, natural_label_fit=False,
        thresholds=dict(zero=0., program_correct95=threshold) if name == MEAN else dict(zero=0.),
        threshold_rules=dict(zero='fixed_zero', program_correct95='matched_source_no_edges_correct95_strict_gt')
                        if name == MEAN else dict(zero='fixed_zero'))
        for name, values in predictions.items()}
    summaries, counts = summarize_methods(pack, metadata['records'], methods)
    original = np.load(args.fits / 'natural_test/scores.npz')['postcandidate_mean']
    write_json(args.output / 'RESULTS.json', dict(diagnostic_only=True, counts=counts, methods=summaries,
        bootstrap_vs_parent=source_bootstrap(pack, predictions[MEAN], original, 300),
        limitations='No local graph messages; hidden states still encode history. Offline observed-word diagnostic.'))
    export_cases(args, pack, metadata, methods)
    print(json.dumps({name: row['all_tokens'] for name, row in summaries.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('program', 'score', 'evaluate', 'all'))
    parser.add_argument('--fits', type=Path, default=Path('outputs/first_timing_validation_20261008'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--parent', type=Path, default=OUTPUT)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2026])
    parser.add_argument('--batch-sources', type=int, default=16)
    parser.add_argument('--batch-answers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    phases = dict(program=program, score=score, evaluate=evaluate)
    for name in phases if args.phase == 'all' else [args.phase]:
        phases[name](args)


if __name__ == '__main__':
    main()
