"""Score fixed KL graph fidelity on complete FIT/pilot/QA DEV before labels.

Source/route unaries and their references are unchanged. Working Bernoulli
coordinates do not make these scores calibrated factual probabilities.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import scipy
import torch

from experiments.native_support.evaluate import ranking
from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import source_bootstrap
from .kl_graph import prepare_graph, kl_smooth_normalized, quadratic_smooth_normalized
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_likelihood_calibration import array_hash, native_likelihood, view_reference
from .run_state_graph import snapshot_code
from .run_unlabeled import answer_inputs, weighted_reference_threshold
from .source_null_calibration import bounded_center, conditional_coordinates
from .source_route_refine_eval import answer_masks, alarm_counts, method_report, transitions
from .unlabeled import equal_source_weights, local_graph_edges, local_weights, reference_rank, rewire_weights


PREVIOUS = Path('outputs/source_null_calibration_20261009')
DEFAULT_OUTPUT = Path('outputs/qa_kl_graph_20261009')
PLAN = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_directed_propagation_20261009/EXPERIMENT_PLAN.md')
PRIMARY = 'kl_native'
METHODS = ('old_native', 'null_unary', 'null_native', 'kl_native',
           'kl_weak', 'logit_native', 'kl_rewired')


def previous_inputs(phase):
    """Open frozen pure identities and historical scores, never token labels."""
    freeze = json.loads((PREVIOUS / f'{phase}_FREEZE.json').read_text())
    names = [f'{phase}_{suffix}' for suffix in ('pack.npz', 'metadata.json', 'scores.npz', 'inputs.json')]
    for name in names:
        path = PREVIOUS / name
        if file_hash(path) != freeze['artifacts'][str(path.resolve())]:
            raise ValueError(f'Previous artifact changed: {path}')
    with np.load(PREVIOUS / names[0]) as saved:
        pack = dict(saved)
    if set(pack) != {'target', 'token_id', 'source_index', 'answer_index', 'unit_index'}:
        raise ValueError('Only identity fields may enter scoring')
    with np.load(PREVIOUS / names[2]) as saved:
        expected = {name: saved[name].copy() for name in METHODS[:3]}
    return pack, json.loads((PREVIOUS / names[1]).read_text()), expected, json.loads((PREVIOUS / names[3]).read_text())


def initialize(directory):
    """Bind exact executable dependencies and copies before the first score."""
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'full_nodes').mkdir()
    dependencies = snapshot_code(directory)
    for path in (Path(__file__).resolve(), Path(__file__).with_name('test_kl_graph.py').resolve()):
        copy = directory / 'code_snapshot' / path.relative_to(Path.cwd())
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        dependencies[str(path)] = file_hash(path)
    for name in ('reference.npz', 'global_reference.npz'):
        path = PREVIOUS / name
        frozen = json.loads((PREVIOUS / 'fit_FREEZE.json').read_text())
        if file_hash(path) != frozen['artifacts'][str(path.resolve())]:
            raise ValueError('Previous source reference changed')
        dependencies[str(path.resolve())] = file_hash(path)
        shutil.copy2(path, directory / name)
    dependencies[str(PLAN)] = file_hash(PLAN)
    dependencies[str(Path('.aris/compute/env-spec.json').resolve())] = file_hash(Path('.aris/compute/env-spec.json'))
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    for path in (CACHE / 'manifest.json', Path(manifest['dataset']) / 'response.jsonl'):
        dependencies[str(path.resolve())] = file_hash(path)
    shutil.copy2(PLAN, directory / 'EXPERIMENT_PLAN.md')
    copies = {str(path.resolve()): file_hash(path) for path in (directory / 'code_snapshot').rglob('*.py')}
    write_json(directory / 'DEPENDENCIES.json', dependencies)
    write_json(directory / 'SNAPSHOT_BINDING.json', copies)


def normalized_graph(weights):
    sources, targets, raw = local_graph_edges(weights)
    return prepare_graph(sources.numpy(), targets.numpy(), raw.numpy(), len(weights))


def answer_fields(values, attention, likelihood, references, global_reference):
    """Keep source coordinates; change only the fixed full-answer solver."""
    anchored = {}
    for view in ('local', 'full'):
        rank, zero = conditional_coordinates(view_reference(references, view), values['source_' + view], likelihood[view])
        anchored[view] = bounded_center(rank, zero)
    route = reference_rank(global_reference, 'raw_route', values['raw_route'])
    unary = .375 * (anchored['local'] + anchored['full']) + .25 * route
    old_source = sum(reference_rank(global_reference, 'source_' + view, values['source_' + view]) for view in ('local', 'full'))
    old_unary = .375 * old_source + .25 * route
    weights = local_weights(attention)
    native = normalized_graph(weights)
    rewired = normalized_graph(rewire_weights(weights, seed=42))
    jobs = dict(old_native=quadratic_smooth_normalized(old_unary, native),
                null_native=quadratic_smooth_normalized(unary, native),
                kl_native=kl_smooth_normalized(unary, native, gamma=2.),
                kl_weak=kl_smooth_normalized(unary, native, gamma=.5),
                logit_native=quadratic_smooth_normalized(unary, native, coordinate='logit'),
                kl_rewired=kl_smooth_normalized(unary, rewired, gamma=2.))
    scores = {name: result['solution'] for name, result in jobs.items()}
    scores['null_unary'] = unary
    diagnostics = {name: {key: value for key, value in result.items() if np.ndim(value) == 0}
                   for name, result in jobs.items()}
    return scores, diagnostics, weights


def score_population(directory, phase, pack, metadata, expected, previous_raw):
    with np.load(directory / 'reference.npz') as saved:
        references = dict(saved)
    with np.load(directory / 'global_reference.npz') as saved:
        global_reference = dict(saved)
    scores = {name: np.full(len(pack['target']), np.nan) for name in METHODS}
    differences = {name: 0. for name in expected}
    files, captures, full_files, diagnostics = {}, {}, {}, {}
    for index, record in enumerate(metadata['records']):
        values, attention, targets, region = answer_inputs(CACHE, OUTPUT / 'capture', record, pack)
        likelihood = native_likelihood(record, values['token_id'])
        fields, stats, weights = answer_fields(values, attention, likelihood, references, global_reference)
        for name in METHODS:
            scores[name][region] = fields[name][targets]
        for name in expected:
            differences[name] = max(differences[name], float(np.max(np.abs(scores[name][region] - expected[name][region]))))
        for name in ('observations.npz', 'with_source.npz', 'response.json'):
            path = CACHE / record['directory'] / name
            actual = file_hash(path)
            if actual != previous_raw['files'][str(path.resolve())]:
                raise ValueError(f'Previous scalar input changed: {path}')
            files[str(path.resolve())] = actual
        capture = OUTPUT / 'capture' / record['id'] / 'arrays.npz'
        actual = dict(attention=array_hash(attention), answer_ids=array_hash(values['token_id']))
        if actual != previous_raw['capture_fields'][str(capture.resolve())]:
            raise ValueError(f'Previous graph input changed: {capture}')
        captures[str(capture.resolve())] = actual
        path = directory / 'full_nodes' / f'{phase}_{record["id"]}.npz'
        np.savez(path, **fields, raw_weights=weights, token_id=values['token_id'])
        full_files[str(path.resolve())] = file_hash(path)
        diagnostics[record['id']] = stats
        if (index + 1) % 300 == 0:
            print(f'KL SCORE {phase} {index + 1}/{len(metadata["records"])}', flush=True)
    if max(differences.values()) > 1.1e-8:
        raise ValueError(f'Historical field identity failed: {differences}')
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError('Incomplete scoring population')
    return scores, dict(files=files, capture_fields=captures, full_nodes=full_files,
                        baseline_max_differences=differences, solver=diagnostics)


def verify(directory, phase):
    freeze = json.loads((directory / f'{phase}_FREEZE.json').read_text())
    for category in ('artifacts', 'dependencies', 'snapshots'):
        for path, expected in freeze[category].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f'Changed frozen {category}: {path}')
    inputs = json.loads((directory / f'{phase}_inputs.json').read_text())
    for category in ('files', 'full_nodes'):
        for path, expected in inputs[category].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f'Changed consumed {category}: {path}')
    return freeze


def run_score(directory, phase):
    started = time.time()
    pack, metadata, expected, previous_raw = previous_inputs(phase)
    if phase == 'fit':
        initialize(directory)
    else:
        verify(directory, 'fit')
    if (directory / f'{phase}_FREEZE.json').exists():
        raise FileExistsError('Completed scores must not be overwritten')
    scores, inputs = score_population(directory, phase, pack, metadata, expected, previous_raw)
    np.savez(directory / f'{phase}_scores.npz', **scores)
    np.savez(directory / f'{phase}_pack.npz', **pack)
    write_json(directory / f'{phase}_metadata.json', metadata)
    write_json(directory / f'{phase}_inputs.json', inputs)
    if phase == 'fit':
        thresholds = {name: weighted_reference_threshold(values, equal_source_weights(pack['source_index'])) for name, values in scores.items()}
        write_json(directory / 'CONFIG.json', dict(primary=PRIMARY, methods=METHODS, thresholds=thresholds,
            gamma=2., gamma_weak=.5, quadratic_penalty=.5, rewired_seed=42,
            natural_label_fits=0, new_llm_forwards=0, source_unchanged=True,
            working_bernoulli_not_truth_probability=True, offline=True,
            threshold_rule='equal-source mixed FIT95, not normal FPR'))
    names = [f'{phase}_{suffix}' for suffix in ('scores.npz', 'pack.npz', 'metadata.json', 'inputs.json')]
    names += ['CONFIG.json', 'reference.npz', 'global_reference.npz', 'DEPENDENCIES.json', 'SNAPSHOT_BINDING.json', 'EXPERIMENT_PLAN.md']
    dependencies = json.loads((directory / 'DEPENDENCIES.json').read_text())
    snapshots = json.loads((directory / 'SNAPSHOT_BINDING.json').read_text())
    for path, expected_hash in dependencies.items():
        if file_hash(Path(path)) != expected_hash:
            raise ValueError(f'Executable dependency changed while scoring: {path}')
    write_json(directory / f'{phase}_FREEZE.json', dict(status='all_predictions_frozen',
        artifacts={str((directory / name).resolve()): file_hash(directory / name) for name in names},
        dependencies=dependencies, snapshots=snapshots, baseline_max_differences=inputs['baseline_max_differences'],
        natural_labels_opened=False, historical_exposure=True, seconds=time.time() - started,
        command=sys.argv, python=sys.version, numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__))
    print(f'KL {phase} FROZEN seconds={time.time() - started:.2f}', flush=True)


def write_tokens(directory, phase, pack, metadata, scores, thresholds):
    columns = ['id', 'source_id', 'generator', 'target', 'word', 'label', 'onset', 'first']
    columns += [column for name in METHODS for column in (name, name + '__alarm')]
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
                writer.writerow(row)


def run_evaluate(directory, phase):
    if phase == 'fit':
        raise ValueError('FIT labels are not used')
    if (directory / f'{phase}_evaluation.json').exists():
        raise FileExistsError('Completed evaluation must not be overwritten')
    for required in ('fit', 'pilot', 'dev'):
        verify(directory, required)
    with np.load(directory / f'{phase}_pack.npz') as saved:
        pack = dict(saved)
    with np.load(directory / f'{phase}_scores.npz') as saved:
        scores = dict(saved)
    metadata = json.loads((directory / f'{phase}_metadata.json').read_text())
    thresholds = json.loads((directory / 'CONFIG.json').read_text())['thresholds']
    pack.update(evaluation_labels(CACHE, pack, metadata))
    clean, first, answers = answer_masks(pack, metadata['records'])
    baseline = alarm_counts(pack, scores['old_native'], thresholds['old_native'], clean, first, answers)
    budgets = dict(normal_token_budget=baseline['fp'], normal_answer_budget=baseline['normal_answer_alarms'])
    methods = {name: method_report(pack, values, thresholds[name], clean, first, answers, budgets) for name, values in scores.items()}
    comparisons = {name: source_bootstrap(pack, scores[PRIMARY], scores[name], repeats=300) for name in ('old_native', 'null_native', 'logit_native')}
    changed = {policy: transitions(pack['labels'], scores['old_native'], scores[PRIMARY],
                                  methods['old_native'][policy]['threshold'], methods[PRIMARY][policy]['threshold'])
               for policy in ('fit_mixture95', 'normal_token_budget')}
    generators = {}
    for generator in sorted({row['generator'] for row in metadata['records']}):
        selected = np.zeros(len(pack['labels']), dtype=bool)
        for record in metadata['records']:
            if record['generator'] == generator:
                selected[record['packed_start']:record['packed_stop']] = True
        generators[generator] = {name: ranking(pack['labels'][selected], scores[name][selected]) for name in METHODS}
    write_tokens(directory, phase, pack, metadata, scores, thresholds)
    write_json(directory / f'{phase}_evaluation.json', dict(primary=PRIMARY, methods=methods,
        source_bootstrap=comparisons, baseline_to_primary=changed, generators=generators,
        counts=dict(answers=len(metadata['records']), tokens=len(pack['labels']), positives=int(pack['labels'].sum())),
        historical_exposure=True, natural_label_fits=0, oracle_budgets_are_not_deployment_calibration=True,
        freeze_sha256=file_hash(directory / f'{phase}_FREEZE.json')))
    print(json.dumps({name: row['all_tokens'] for name, row in methods.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('score', 'evaluate'), required=True)
    parser.add_argument('--phase', choices=('fit', 'pilot', 'dev'), required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    (run_score if args.stage == 'score' else run_evaluate)(args.output, args.phase)


if __name__ == '__main__':
    main()
