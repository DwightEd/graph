"""Freeze zero-effect-anchored source ranks on complete FIT, pilot and QA DEV.

Only source coordinates change. References, route, fusion and native graph are
reused. A separate evaluation stage opens official labels after all scores.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from experiments.native_support.evaluate import ranking
from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import source_bootstrap
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_likelihood_calibration import (array_hash, native_likelihood,
    phase_inputs, verify as verify_previous, view_reference)
from .run_state_graph import snapshot_code
from .run_unlabeled import answer_inputs, weighted_reference_threshold
from .source_null_calibration import balanced_center, bounded_center, conditional_coordinates
from .source_route_refine_eval import answer_masks, alarm_counts, method_report, transitions
from .unlabeled import CHANNELS, equal_source_weights, graph_fields, local_weights, reference_rank


PREVIOUS = Path('outputs/source_likelihood_calibration_20261009')
GLOBAL = Path('outputs/source_route_refine_20261009/scalar_reference.npz')
DEFAULT_OUTPUT = Path('outputs/source_null_calibration_20261009')
PLAN = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/source_null_calibration_20261009/EXPERIMENT_PLAN.md')
METHODS = ('old_unary', 'old_native', 'likelihood_native', 'null_unary', 'null_native',
           'balanced_native', 'global_null_native', 'shuffled_null_native')
PRIMARY = 'null_native'


def source_coordinates(global_reference, references, values, likelihood):
    """Keep actual effects and their same-bin null separately for audit."""
    ranks = {name: reference_rank(global_reference, name, values[name]) for name in CHANNELS}
    diagnostics = dict(ranks)
    conditional, anchored, balanced, global_null, shuffled = {}, {}, {}, {}, {}
    for view in ('local', 'full'):
        effect = values['source_' + view]
        rank, null = conditional_coordinates(view_reference(references, view), effect, likelihood[view])
        conditional[view] = rank
        anchored[view] = bounded_center(rank, null)
        balanced[view] = balanced_center(rank, null)
        global_rank = ranks['source_' + view]
        global_zero = reference_rank(global_reference, 'source_' + view, np.zeros_like(effect))
        global_null[view] = bounded_center(global_rank, global_zero)
        shuffled_rank, shuffled_zero = conditional_coordinates(
            view_reference(references, 'shuffle_' + view), effect, likelihood[view])
        shuffled[view] = bounded_center(shuffled_rank, shuffled_zero)
        diagnostics.update({view + '_conditional': rank, view + '_zero_rank': null,
            view + '_anchored': anchored[view], view + '_effect': effect,
            view + '_native_logp': likelihood[view], view + '_balanced': balanced[view],
            view + '_global_zero_rank': global_zero, view + '_shuffled_zero_rank': shuffled_zero})
    coordinates = dict(old={v: ranks['source_' + v] for v in ('local', 'full')},
        likelihood=conditional, null=anchored, balanced=balanced, global_null=global_null, shuffled_null=shuffled)
    return coordinates, diagnostics


def fields_for_answer(global_reference, references, values, likelihood, attention):
    coordinates, diagnostics = source_coordinates(global_reference, references, values, likelihood)
    route = diagnostics['raw_route']
    graph = dict(native=local_weights(attention))
    scores = {}
    for name, source in coordinates.items():
        unary = .375 * (source['local'] + source['full']) + .25 * route
        if name in ('old', 'null'):
            scores[name + '_unary'] = unary
        fields, _ = graph_fields(unary, graph)
        scores[name + '_native'] = fields['native_huber']
    return scores, diagnostics


def initialize(directory):
    """Reuse the full frozen reference and bind the actual executable evaluator."""
    verify_previous(PREVIOUS, 'fit')
    directory.mkdir(parents=True, exist_ok=False)
    dependencies = snapshot_code(directory)
    test = Path(__file__).with_name('test_source_null_calibration.py').resolve()
    destination = directory / 'code_snapshot' / test.relative_to(Path.cwd().resolve())
    shutil.copy2(test, destination)
    dependencies[str(test)] = file_hash(test)
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    inputs = [PLAN, GLOBAL, PREVIOUS / 'reference.npz', PREVIOUS / 'fit_FREEZE.json',
        CACHE / 'manifest.json', Path(manifest['dataset']) / 'response.jsonl',
        Path('.aris/compute/env-spec.json')]
    dependencies.update({str(path.resolve()): file_hash(path) for path in inputs})
    write_json(directory / 'DEPENDENCIES.json', dependencies)
    shutil.copy2(PLAN, directory / 'EXPERIMENT_PLAN.md')
    shutil.copy2(PREVIOUS / 'reference.npz', directory / 'reference.npz')
    shutil.copy2(GLOBAL, directory / 'global_reference.npz')


def previous_predictions(phase):
    frozen = json.loads((PREVIOUS / f'{phase}_FREEZE.json').read_text())
    path = PREVIOUS / f'{phase}_scores.npz'
    if file_hash(path) != frozen['artifacts'][str(path.resolve())]:
        raise ValueError('Archived calibration scores changed')
    with np.load(path) as saved:
        return {name: saved[name].copy() for name in ('old_unary', 'old_native', 'likelihood_native')}


def score_records(global_reference, references, pack, metadata, expected):
    scores = {name: np.full(len(pack['target']), np.nan) for name in METHODS}
    diagnostics, files, capture_fields = {}, {}, {}
    differences = dict.fromkeys(expected, 0.)
    zero_checks = {view: dict(count=0, max_error=0.) for view in ('local', 'full')}
    for index, record in enumerate(metadata['records']):
        values, attention, targets, region = answer_inputs(CACHE, OUTPUT / 'capture', record, pack)
        likelihood = native_likelihood(record, values['token_id'])
        result, readouts = fields_for_answer(global_reference, references, values, likelihood, attention)
        for name in METHODS:
            scores[name][region] = result[name][targets]
        for name, value in readouts.items():
            diagnostics.setdefault(name, np.empty(len(pack['target'])))[region] = value[targets]
        for name in expected:
            differences[name] = max(differences[name], float(np.abs(scores[name][region] - expected[name][region]).max()))
        for view in zero_checks:
            selected = values['source_' + view] == 0
            zero_checks[view]['count'] += int(selected.sum())
            if selected.any():
                zero_checks[view]['max_error'] = max(zero_checks[view]['max_error'],
                    float(np.abs(readouts[view + '_anchored'][selected] - .5).max()))
        for name in ('observations.npz', 'with_source.npz', 'response.json'):
            path = CACHE / record['directory'] / name
            files[str(path.resolve())] = file_hash(path)
        path = OUTPUT / 'capture' / record['id'] / 'arrays.npz'
        capture_fields[str(path.resolve())] = dict(attention=array_hash(attention), answer_ids=array_hash(values['token_id']))
        if (index + 1) % 300 == 0:
            print(f'NULL SCORE {index + 1}/{len(metadata["records"])}', flush=True)
    if max(differences.values()) > 1e-8:
        raise ValueError(f'Frozen baseline identity failed: {differences}')
    if not all(np.isfinite(v).all() for v in (*scores.values(), *diagnostics.values())):
        raise ValueError('Incomplete source-null population')
    return scores, diagnostics, dict(files=files, capture_fields=capture_fields,
        baseline_max_differences=differences, exact_zero_checks=zero_checks)


def verify(directory, phase):
    frozen = json.loads((directory / f'{phase}_FREEZE.json').read_text())
    for category in ('artifacts', 'dependencies'):
        for path, expected in frozen[category].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f'Changed frozen {category}: {path}')
    inputs = json.loads((directory / f'{phase}_inputs.json').read_text())
    for path, expected in inputs['files'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Changed consumed scalar input: {path}')
    for path, expected in inputs['capture_fields'].items():
        with np.load(path) as arrays:
            actual = dict(attention=array_hash(arrays['local_attention'][0, 1:]), answer_ids=array_hash(arrays['answer_ids']))
        if actual != expected:
            raise ValueError(f'Changed consumed graph field: {path}')
    return frozen


def run_score(directory, phase):
    started = time.time()
    pack, metadata, _ = phase_inputs(phase)
    expected = previous_predictions(phase)
    if phase == 'fit':
        initialize(directory)
    else:
        verify(directory, 'fit')
    if (directory / f'{phase}_FREEZE.json').exists():
        raise FileExistsError('Completed scores must not be overwritten')
    dependencies = json.loads((directory / 'DEPENDENCIES.json').read_text())
    with np.load(directory / 'reference.npz') as saved:
        references = dict(saved)
    with np.load(directory / 'global_reference.npz') as saved:
        global_reference = dict(saved)
    scores, diagnostics, inputs = score_records(global_reference, references, pack, metadata, expected)
    np.savez(directory / f'{phase}_scores.npz', **scores)
    np.savez(directory / f'{phase}_ranks.npz', **diagnostics)
    np.savez(directory / f'{phase}_pack.npz', **pack)
    write_json(directory / f'{phase}_metadata.json', metadata)
    write_json(directory / f'{phase}_inputs.json', inputs)
    if phase == 'fit':
        weights = equal_source_weights(pack['source_index'])
        thresholds = {name: weighted_reference_threshold(values, weights) for name, values in scores.items()}
        write_json(directory / 'CONFIG.json', dict(primary=PRIMARY, methods=METHODS, thresholds=thresholds,
            natural_label_fits=0, new_llm_forwards=0, reference_reused=True, graph_changed=False,
            threshold_rule='equal-source mixed fit95; not normal FPR', conditional_bins=16,
            anchor='bounded slope; same-bin zero actual-token logp effect maps .5; not truth probability'))
    artifacts = [directory / f'{phase}_{suffix}' for suffix in ('scores.npz', 'ranks.npz', 'pack.npz', 'metadata.json', 'inputs.json')]
    artifacts += [directory / name for name in ('CONFIG.json', 'reference.npz', 'global_reference.npz', 'DEPENDENCIES.json', 'EXPERIMENT_PLAN.md')]
    artifacts += [PREVIOUS / f'{phase}_FREEZE.json', PREVIOUS / f'{phase}_scores.npz']
    write_json(directory / f'{phase}_FREEZE.json', dict(status='all_predictions_frozen',
        artifacts={str(path.resolve()): file_hash(path) for path in artifacts}, dependencies=dependencies,
        baseline_max_differences=inputs['baseline_max_differences'], natural_labels_opened=False,
        seconds=time.time() - started, command=sys.argv, python=sys.version, numpy=np.__version__, torch=torch.__version__))
    print(f'NULL {phase} FROZEN seconds={time.time() - started:.2f}', flush=True)


def write_tokens(directory, phase, pack, metadata, scores, thresholds, diagnostics):
    columns = ['id', 'source_id', 'generator', 'target', 'word', 'label', 'onset', 'first']
    columns += [field for name in METHODS for field in (name, name + '__alarm')]
    columns += list(diagnostics)
    with (directory / f'{phase}_tokens.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in metadata['records']:
            response = json.loads((CACHE / record['directory'] / 'response.json').read_text())
            for index in range(record['packed_start'], record['packed_stop']):
                target = int(pack['target'][index])
                row = dict(id=record['id'], source_id=record['source_id'], generator=record['generator'],
                    target=target, word=response['token_text'][target], label=int(pack['labels'][index]),
                    onset=int(pack['onsets'][index]), first=int(pack['firsts'][index]))
                for name in METHODS:
                    row[name] = float(scores[name][index])
                    row[name + '__alarm'] = int(scores[name][index] > thresholds[name])
                row.update({name: float(value[index]) for name, value in diagnostics.items()})
                writer.writerow(row)


def run_evaluate(directory, phase):
    if phase == 'fit':
        raise ValueError('Fit labels are not needed by this experiment')
    if (directory / f'{phase}_evaluation.json').exists():
        raise FileExistsError('Completed evaluation must not be overwritten')
    # The entire predeclared population is frozen before any new evaluation.
    for required in ('fit', 'pilot', 'dev'):
        verify(directory, required)
    with np.load(directory / f'{phase}_pack.npz') as saved:
        pack = dict(saved)
    with np.load(directory / f'{phase}_scores.npz') as saved:
        scores = dict(saved)
    with np.load(directory / f'{phase}_ranks.npz') as saved:
        diagnostics = dict(saved)
    metadata = json.loads((directory / f'{phase}_metadata.json').read_text())
    thresholds = json.loads((directory / 'CONFIG.json').read_text())['thresholds']
    pack.update(evaluation_labels(CACHE, pack, metadata))
    clean, first, answers = answer_masks(pack, metadata['records'])
    baseline = alarm_counts(pack, scores['old_native'], thresholds['old_native'], clean, first, answers)
    budgets = dict(normal_token_budget=baseline['fp'], normal_answer_budget=baseline['normal_answer_alarms'])
    methods = {name: method_report(pack, values, thresholds[name], clean, first, answers, budgets)
               for name, values in scores.items()}
    comparisons = {name: source_bootstrap(pack, scores[PRIMARY], scores[name], repeats=300)
                   for name in ('old_native', 'likelihood_native', 'balanced_native', 'global_null_native', 'shuffled_null_native')}
    changed = {}
    for policy in ('fit_mixture95', 'normal_token_budget'):
        old, new = methods['old_native'][policy], methods[PRIMARY][policy]
        changed[policy] = transitions(pack['labels'], scores['old_native'], scores[PRIMARY], old['threshold'], new['threshold'])
    write_tokens(directory, phase, pack, metadata, scores, thresholds, diagnostics)
    generators = {}
    for generator in sorted({row['generator'] for row in metadata['records']}):
        selected = np.zeros(len(pack['labels']), dtype=bool)
        for record in metadata['records']:
            if record['generator'] == generator:
                selected[record['packed_start']:record['packed_stop']] = True
        generators[generator] = {name: ranking(pack['labels'][selected], scores[name][selected]) for name in METHODS}
    write_json(directory / f'{phase}_evaluation.json', dict(primary=PRIMARY, methods=methods,
        source_bootstrap=comparisons, baseline_to_primary=changed, generators=generators,
        counts=dict(answers=len(metadata['records']), tokens=len(pack['labels']), positives=int(pack['labels'].sum())),
        freeze_sha256=file_hash(directory / f'{phase}_FREEZE.json'), historical_exposure=True,
        natural_label_fits=0, oracle_budgets_are_not_deployment_calibration=True))
    print(json.dumps({name: row['all_tokens'] for name, row in methods.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('score', 'evaluate'))
    parser.add_argument('--phase', required=True, choices=('fit', 'pilot', 'dev'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    (run_score if args.stage == 'score' else run_evaluate)(args.output, args.phase)


if __name__ == '__main__':
    main()
