"""Fixed source-event readout with the unchanged cached arithmetic graph.

All natural populations freeze before evaluation. The probability mixture is
an artificial observer-view mixture, not a native factual posterior.
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
from .kl_graph import quadratic_smooth_normalized
from .run_capture import CACHE, file_hash, write_json
from .run_kl_graph import normalized_graph, previous_inputs
from .run_state_graph import snapshot_code
from .run_unlabeled import weighted_reference_threshold
from .source_event import source_event_effects
from .source_null_calibration import bounded_center
from .source_route_refine_eval import answer_masks, alarm_counts, method_report, transitions
from .unlabeled import equal_source_weights, fit_weighted_cdf, reference_rank


PREVIOUS = Path('outputs/source_null_calibration_20261009')
GRAPH_CACHE = Path('outputs/qa_kl_graph_20261009')
DEFAULT_OUTPUT = Path('outputs/qa_source_event_20261009')
PLAN = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/qa_source_event_20261009/EXPERIMENT_PLAN.md')
QUALIFICATION = PLAN.with_name('QUALIFICATION.json')
PRIMARY = 'event_fisher_native'
METHODS = ('old_native', 'null_unary', 'null_native', 'event_fisher_unary',
           'event_fisher_native', 'event_logp_unary', 'event_logp_native',
           'event_probability_unary', 'event_probability_native')
GEOMETRIES = ('fisher', 'logp', 'probability')


def initialize(directory):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'full_nodes').mkdir()
    dependencies = snapshot_code(directory)
    for path in (Path(__file__).resolve(), Path(__file__).with_name('test_source_event.py').resolve()):
        copy = directory / 'code_snapshot' / path.relative_to(Path.cwd())
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        dependencies[str(path)] = file_hash(path)
    reference = PREVIOUS / 'global_reference.npz'
    prior = json.loads((PREVIOUS / 'fit_FREEZE.json').read_text())
    if file_hash(reference) != prior['artifacts'][str(reference.resolve())]:
        raise ValueError('Prior route reference changed')
    shutil.copy2(reference, directory / 'global_reference.npz')
    dependencies[str(reference.resolve())] = file_hash(reference)
    dependencies[str(PLAN)] = file_hash(PLAN)
    dependencies[str(QUALIFICATION)] = file_hash(QUALIFICATION)
    for phase in ('fit', 'pilot', 'dev'):
        for suffix in ('pack.npz', 'metadata.json', 'scores.npz', 'inputs.json', 'FREEZE.json'):
            path = PREVIOUS / f'{phase}_{suffix}'
            dependencies[str(path.resolve())] = file_hash(path)
    dependencies[str(Path('.aris/compute/env-spec.json').resolve())] = file_hash(Path('.aris/compute/env-spec.json'))
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    for path in (CACHE / 'manifest.json', Path(manifest['dataset']) / 'response.jsonl'):
        dependencies[str(path.resolve())] = file_hash(path)
    shutil.copy2(PLAN, directory / 'EXPERIMENT_PLAN.md')
    write_json(directory / 'DEPENDENCIES.json', dependencies)
    write_json(directory / 'SNAPSHOT_BINDING.json', {
        str(path.resolve()): file_hash(path) for path in (directory / 'code_snapshot').rglob('*.py')})


def native_events(record, qualified_files=None):
    base = CACHE / record['directory']
    if qualified_files is not None:
        for name in ('with_source.npz', 'without_source.npz', 'observations.npz'):
            path = base / name
            if file_hash(path) != qualified_files[str(path.resolve())]:
                raise ValueError('FIT qualification bytes changed before reading')
    with np.load(base / 'with_source.npz') as saved:
        present = {name: saved[name].copy() for name in ('local', 'full', 'token_id')}
    with np.load(base / 'without_source.npz') as saved:
        absent = {name: saved[name].copy() for name in ('local', 'full', 'token_id')}
    with np.load(base / 'observations.npz') as saved:
        observations = {name: saved[name].copy() for name in ('source_local', 'source_full', 'raw_route', 'token_id')}
    if not np.array_equal(present['token_id'], absent['token_id']) or not np.array_equal(present['token_id'], observations['token_id']):
        raise ValueError('Paired native token IDs changed')
    for view in ('local', 'full'):
        delta = absent[view].astype(np.float64) - present[view].astype(np.float64)
        if not np.array_equal(delta, observations['source_' + view]):
            raise ValueError('Stored source effects differ from paired logp')
    events = source_event_effects(present['local'], present['full'], absent['local'], absent['full'])
    events.update({condition + '_' + view + '_logp': values[view].astype(np.float64)
                   for condition, values in (('present', present), ('absent', absent))
                   for view in ('local', 'full')})
    events.update(observations)
    return events


def fit_event_reference(directory, pack, metadata):
    qualified_files = json.loads(QUALIFICATION.read_text())['files']
    effects = {name: np.full(len(pack['target']), np.nan) for name in GEOMETRIES}
    for record in metadata['records']:
        events = native_events(record, qualified_files)
        region = slice(record['packed_start'], record['packed_stop'])
        targets = pack['target'][region]
        if not np.array_equal(events['token_id'][targets], pack['token_id'][region]):
            raise ValueError('FIT native/pack IDs changed')
        for name in GEOMETRIES:
            effects[name][region] = events[name + '_effect'][targets]
    weights = equal_source_weights(pack['source_index'])
    reference = {}
    for name, values in effects.items():
        if not np.isfinite(values).all():
            raise ValueError('Incomplete FIT event reference')
        distinct, cumulative = fit_weighted_cdf(values, weights)
        reference[name + '_values'] = distinct
        reference[name + '_cumulative'] = cumulative
    np.savez(directory / 'event_reference.npz', **reference)
    return reference, {name: float(reference_rank(reference, name, np.array([0.]))[0]) for name in GEOMETRIES}


def score_population(directory, phase, pack, metadata, expected, previous_raw, reference):
    with np.load(directory / 'global_reference.npz') as saved:
        global_reference = dict(saved)
    graph_freeze = json.loads((GRAPH_CACHE / f'{phase}_FREEZE.json').read_text())
    graph_inputs = json.loads((GRAPH_CACHE / f'{phase}_inputs.json').read_text())
    files = {str((GRAPH_CACHE / (phase + '_' + suffix)).resolve()): file_hash(GRAPH_CACHE / (phase + '_' + suffix))
             for suffix in ('FREEZE.json', 'inputs.json')}
    if files[str((GRAPH_CACHE / f'{phase}_inputs.json').resolve())] != graph_freeze['artifacts'][str((GRAPH_CACHE / f'{phase}_inputs.json').resolve())]:
        raise ValueError('Frozen graph inputs changed')
    scores = {name: np.full(len(pack['target']), np.nan) for name in METHODS}
    differences, full_files, diagnostics = {name: 0. for name in expected}, {}, {}
    for index, record in enumerate(metadata['records']):
        region = slice(record['packed_start'], record['packed_stop'])
        targets = pack['target'][region]
        events = native_events(record)
        if not np.array_equal(events['token_id'][targets], pack['token_id'][region]):
            raise ValueError('Native/pack IDs changed')
        path = GRAPH_CACHE / 'full_nodes' / f'{phase}_{record["id"]}.npz'
        if file_hash(path) != graph_inputs['full_nodes'][str(path.resolve())]:
            raise ValueError('Frozen graph full nodes changed')
        files[str(path.resolve())] = file_hash(path)
        with np.load(path) as saved:
            if not np.array_equal(saved['token_id'], events['token_id']):
                raise ValueError('Graph/native IDs changed')
            weights = saved['raw_weights'].copy()
            fields = {name: saved[name].copy() for name in METHODS[:3]}
        graph = normalized_graph(weights)
        route = reference_rank(global_reference, 'raw_route', events['raw_route'])
        stats = {}
        for name in GEOMETRIES:
            rank = reference_rank(reference, name, events[name + '_effect'])
            zero = reference_rank(reference, name, np.zeros(len(rank)))
            anchor = bounded_center(rank, zero)
            unary = .75 * anchor + .25 * route
            result = quadratic_smooth_normalized(unary, graph)
            fields['event_' + name + '_unary'] = unary
            fields['event_' + name + '_native'] = result['solution']
            events[name + '_rank'], events[name + '_anchor'] = rank, anchor
            stats[name] = {key: value for key, value in result.items() if np.ndim(value) == 0}
        for name in METHODS:
            scores[name][region] = fields[name][targets]
        for name in expected:
            differences[name] = max(differences[name], float(np.max(np.abs(scores[name][region] - expected[name][region]))))
        for name in ('observations.npz', 'with_source.npz', 'without_source.npz', 'response.json'):
            path = CACHE / record['directory'] / name
            actual = file_hash(path)
            if name != 'without_source.npz' and actual != previous_raw['files'][str(path.resolve())]:
                raise ValueError('Prior native source bytes changed')
            files[str(path.resolve())] = actual
        path = directory / 'full_nodes' / f'{phase}_{record["id"]}.npz'
        np.savez(path, **fields, **events, raw_weights=weights, route_rank=route, node_degree=graph['degree'])
        full_files[str(path.resolve())] = file_hash(path)
        diagnostics[record['id']] = stats
        if (index + 1) % 300 == 0:
            print(f'EVENT SCORE {phase} {index + 1}/{len(metadata["records"])}', flush=True)
    if max(differences.values()) > 1.1e-8:
        raise ValueError(f'Historical baseline drift: {differences}')
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError('Incomplete population')
    return scores, dict(files=files, full_nodes=full_files, baseline_max_differences=differences, solver=diagnostics)


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
        reference, zero_ranks = fit_event_reference(directory, pack, metadata)
    else:
        verify(directory, 'fit')
        with np.load(directory / 'event_reference.npz') as saved:
            reference = dict(saved)
    if (directory / f'{phase}_FREEZE.json').exists():
        raise FileExistsError('Completed scores cannot be overwritten')
    scores, inputs = score_population(directory, phase, pack, metadata, expected, previous_raw, reference)
    np.savez(directory / f'{phase}_scores.npz', **scores)
    np.savez(directory / f'{phase}_pack.npz', **pack)
    write_json(directory / f'{phase}_metadata.json', metadata)
    write_json(directory / f'{phase}_inputs.json', inputs)
    if phase == 'fit':
        thresholds = {name: weighted_reference_threshold(value, equal_source_weights(pack['source_index'])) for name, value in scores.items()}
        write_json(directory / 'CONFIG.json', dict(primary=PRIMARY, methods=METHODS, thresholds=thresholds,
            zero_ranks=zero_ranks, view_weights=[.5, .5], source_route_weights=[.75, .25],
            graph_penalty=.5, natural_label_fits=0, new_llm_forwards=0, offline=True,
            artificial_event_mixture_not_native_truth_probability=True,
            threshold_rule='equal-source mixed FIT95, not normal FPR'))
    names = [f'{phase}_{suffix}' for suffix in ('scores.npz', 'pack.npz', 'metadata.json', 'inputs.json')]
    names += ['CONFIG.json', 'event_reference.npz', 'global_reference.npz', 'DEPENDENCIES.json', 'SNAPSHOT_BINDING.json', 'EXPERIMENT_PLAN.md']
    dependencies = json.loads((directory / 'DEPENDENCIES.json').read_text())
    for path, expected_hash in dependencies.items():
        if file_hash(Path(path)) != expected_hash:
            raise ValueError('Dependency changed while scoring')
    write_json(directory / f'{phase}_FREEZE.json', dict(status='all_predictions_frozen',
        artifacts={str((directory / name).resolve()): file_hash(directory / name) for name in names},
        dependencies=dependencies, snapshots=json.loads((directory / 'SNAPSHOT_BINDING.json').read_text()),
        baseline_max_differences=inputs['baseline_max_differences'], natural_labels_opened=False,
        historical_exposure=True, seconds=time.time()-started, command=sys.argv, python=sys.version,
        numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__))
    print(f'EVENT {phase} FROZEN seconds={time.time()-started:.2f}', flush=True)


def run_evaluate(directory, phase):
    if phase == 'fit':
        raise ValueError('FIT labels are forbidden')
    if (directory / f'{phase}_evaluation.json').exists():
        raise FileExistsError('Evaluation cannot be overwritten')
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
    comparisons = {name: source_bootstrap(pack, scores[PRIMARY], scores[name], repeats=300)
                   for name in ('old_native', 'null_native', 'event_logp_native', 'event_probability_native')}
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
    columns = ['id', 'source_id', 'generator', 'target', 'word', 'label', 'onset', 'first']
    columns += [column for name in METHODS for column in (name, name+'__alarm')]
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
                    row[name+'__alarm'] = int(scores[name][index] > thresholds[name])
                writer.writerow(row)
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
