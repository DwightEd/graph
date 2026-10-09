"""Replicate the fixed nine-field source-event detector on complete QA test.

Frozen FIT references and thresholds are reused verbatim. The legacy pack is
opened lazily for five identity arrays only; official labels follow score freeze.
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

from . import run_source_event as event
from .data import PACKS
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_likelihood_calibration import array_hash, view_reference
from .run_state_graph import snapshot_code
from .run_unlabeled import answer_inputs
from .source_null_calibration import bounded_center, conditional_coordinates
from .unlabeled import local_weights, reference_rank


SOURCE = event.DEFAULT_OUTPUT
NULL = event.PREVIOUS
OLD = OUTPUT / 'unlabeled'
DEFAULT_OUTPUT = Path('outputs/qa_source_event_test_20261009')
PLAN = event.PLAN.parent.parent / 'qa_source_event_test_20261009/EXPERIMENT_PLAN.md'
IDENTITIES = ('target', 'token_id', 'source_index', 'answer_index', 'unit_index')


def identities():
    with np.load(PACKS / 'QA_test.npz') as saved:
        pack = {name: saved[name].copy() for name in IDENTITIES}
    original = json.loads((PACKS / 'QA_test.json').read_text())
    metadata = {name: original[name] for name in ('records', 'sources', 'task', 'split')}
    if len(metadata['records']) != 900 or len(metadata['sources']) != 150 or len(pack['target']) != 124817:
        raise ValueError('Fixed official QA test population changed')
    test_sources = {row['source_id'] for row in metadata['records']}
    test_answers = {row['id'] for row in metadata['records']}
    for phase in ('fit', 'dev'):
        prior = json.loads((SOURCE / f'{phase}_metadata.json').read_text())['records']
        if test_sources & {row['source_id'] for row in prior} or test_answers & {row['id'] for row in prior}:
            raise ValueError('FIT/DEV/test identities overlap')
    return pack, metadata


def initialize(directory):
    event.verify(SOURCE, 'fit')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'full_nodes').mkdir()
    dependencies = snapshot_code(directory)
    for base, names in ((SOURCE, ('event_reference.npz', 'global_reference.npz', 'CONFIG.json', 'fit_FREEZE.json', 'fit_inputs.json')),
                        (NULL, ('reference.npz', 'fit_FREEZE.json')),
                        (OLD, ('FREEZE.json', 'test_scores.npz'))):
        frozen = json.loads((base / ('FREEZE.json' if base == OLD else 'fit_FREEZE.json')).read_text())
        for name in names:
            path = base / name
            actual = file_hash(path)
            expected = frozen.get('hashes', {}).get(name) if base == OLD else frozen['artifacts'].get(str(path.resolve()))
            if name not in ('fit_FREEZE.json', 'FREEZE.json') and expected != actual:
                raise ValueError('Frozen FIT or archival baseline input changed')
            dependencies[str(path.resolve())] = actual
            if base == SOURCE and name in ('event_reference.npz', 'global_reference.npz', 'CONFIG.json'):
                shutil.copy2(path, directory / name)
            if base == NULL and name == 'reference.npz':
                shutil.copy2(path, directory / 'null_reference.npz')
    for path in (PLAN, PLAN.with_name('RUN.md'), event.PLAN.with_name('TEST_SCOPE_REVIEW.md'),
                 PACKS / 'QA_test.npz', PACKS / 'QA_test.json', CACHE / 'manifest.json',
                 Path('.aris/compute/env-spec.json')):
        dependencies[str(path.resolve())] = file_hash(path)
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    dataset = Path(manifest['dataset']) / 'response.jsonl'
    dependencies[str(dataset.resolve())] = file_hash(dataset)
    shutil.copy2(PLAN, directory / 'EXPERIMENT_PLAN.md')
    write_json(directory / 'DEPENDENCIES.json', dependencies)
    write_json(directory / 'SNAPSHOT_BINDING.json', {
        str(path.resolve()): file_hash(path) for path in (directory / 'code_snapshot').rglob('*.py')})


def score(directory):
    started = time.time()
    pack, metadata = identities()
    initialize(directory)
    with np.load(directory / 'event_reference.npz') as saved:
        reference = dict(saved)
    with np.load(directory / 'global_reference.npz') as saved:
        global_reference = dict(saved)
    with np.load(directory / 'null_reference.npz') as saved:
        null_reference = dict(saved)
    with np.load(OLD / 'test_scores.npz') as saved:
        expected_old = saved['source_route_native_huber'].copy()
    scores = {name: np.full(len(pack['target']), np.nan) for name in event.METHODS}
    files, captures, full_files, diagnostics = {}, {}, {}, {}
    max_old_difference = 0.
    for index, record in enumerate(metadata['records']):
        values, attention, targets, region = answer_inputs(CACHE, OUTPUT / 'capture', record, pack)
        events = event.native_events(record)
        if not np.array_equal(events['token_id'], values['token_id']):
            raise ValueError('Test source/capture IDs changed')
        graph_weights = local_weights(attention)
        graph = event.normalized_graph(graph_weights)
        route = reference_rank(global_reference, 'raw_route', events['raw_route'])
        old_source = sum(reference_rank(global_reference, 'source_' + view, events['source_' + view]) for view in ('local', 'full'))
        old_unary = .375 * old_source + .25 * route
        anchored = {}
        for view in ('local', 'full'):
            rank, zero = conditional_coordinates(view_reference(null_reference, view), events['source_' + view], events['present_' + view + '_logp'])
            anchored[view] = bounded_center(rank, zero)
        null_unary = .375 * (anchored['local'] + anchored['full']) + .25 * route
        fields = dict(old_native=event.quadratic_smooth_normalized(old_unary, graph)['solution'],
                      null_unary=null_unary, null_native=event.quadratic_smooth_normalized(null_unary, graph)['solution'])
        stats = {}
        for name in event.GEOMETRIES:
            rank = reference_rank(reference, name, events[name + '_effect'])
            zero = reference_rank(reference, name, np.zeros(len(rank)))
            anchor = bounded_center(rank, zero)
            unary = .75 * anchor + .25 * route
            result = event.quadratic_smooth_normalized(unary, graph)
            fields['event_' + name + '_unary'] = unary
            fields['event_' + name + '_native'] = result['solution']
            events[name + '_rank'], events[name + '_anchor'] = rank, anchor
            stats[name] = {key: value for key, value in result.items() if np.ndim(value) == 0}
        for name in event.METHODS:
            scores[name][region] = fields[name][targets]
        max_old_difference = max(max_old_difference, float(np.max(np.abs(scores['old_native'][region] - expected_old[region]))))
        for name in ('observations.npz', 'with_source.npz', 'without_source.npz', 'response.json'):
            path = CACHE / record['directory'] / name
            files[str(path.resolve())] = file_hash(path)
        capture = OUTPUT / 'capture' / record['id'] / 'arrays.npz'
        captures[str(capture.resolve())] = dict(attention=array_hash(attention), answer_ids=array_hash(values['token_id']))
        path = directory / 'full_nodes' / f'test_{record["id"]}.npz'
        np.savez(path, **fields, **events, raw_weights=graph_weights, route_rank=route, node_degree=graph['degree'])
        full_files[str(path.resolve())] = file_hash(path)
        diagnostics[record['id']] = stats
        if (index + 1) % 150 == 0:
            print(f'EVENT TEST SCORE {index+1}/900', flush=True)
    if max_old_difference > 1.1e-8 or not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError(f'Test population or archival baseline failed: {max_old_difference}')
    np.savez(directory / 'test_scores.npz', **scores)
    np.savez(directory / 'test_pack.npz', **pack)
    write_json(directory / 'test_metadata.json', metadata)
    write_json(directory / 'test_inputs.json', dict(files=files, capture_fields=captures, full_nodes=full_files,
        baseline_max_differences=dict(old_native=max_old_difference), solver=diagnostics))
    names = ('test_scores.npz', 'test_pack.npz', 'test_metadata.json', 'test_inputs.json',
             'CONFIG.json', 'event_reference.npz', 'global_reference.npz', 'null_reference.npz',
             'EXPERIMENT_PLAN.md', 'DEPENDENCIES.json', 'SNAPSHOT_BINDING.json')
    dependencies = json.loads((directory / 'DEPENDENCIES.json').read_text())
    for path, expected in dependencies.items():
        if file_hash(Path(path)) != expected:
            raise ValueError('Dependency changed while scoring')
    write_json(directory / 'test_FREEZE.json', dict(status='all_predictions_frozen', primary=event.PRIMARY,
        methods=event.METHODS, artifacts={str((directory/name).resolve()):file_hash(directory/name) for name in names},
        dependencies=dependencies, snapshots=json.loads((directory/'SNAPSHOT_BINDING.json').read_text()),
        natural_labels_opened=False, fit_references_rebuilt=False, fit_thresholds_changed=False,
        historical_test_exposure=True, baseline_max_difference=max_old_difference,
        seconds=time.time()-started, command=sys.argv))
    print(f'EVENT TEST FROZEN seconds={time.time()-started:.2f}', flush=True)


def evaluate(directory):
    event.verify(directory, 'test')
    if (directory / 'test_evaluation.json').exists():
        raise FileExistsError('Completed test evaluation cannot be overwritten')
    with np.load(directory / 'test_pack.npz') as saved:
        pack = dict(saved)
    with np.load(directory / 'test_scores.npz') as saved:
        scores = dict(saved)
    metadata = json.loads((directory / 'test_metadata.json').read_text())
    thresholds = json.loads((directory / 'CONFIG.json').read_text())['thresholds']
    pack.update(event.evaluation_labels(CACHE, pack, metadata))
    clean, first, answers = event.answer_masks(pack, metadata['records'])
    baseline = event.alarm_counts(pack, scores['old_native'], thresholds['old_native'], clean, first, answers)
    budgets = dict(normal_token_budget=baseline['fp'], normal_answer_budget=baseline['normal_answer_alarms'])
    methods = {name:event.method_report(pack, values, thresholds[name], clean, first, answers, budgets) for name,values in scores.items()}
    comparisons = {name:event.source_bootstrap(pack, scores[event.PRIMARY], scores[name], repeats=300)
                   for name in ('old_native', 'null_native', 'event_logp_native', 'event_probability_native')}
    generators = {}
    for generator in sorted({row['generator'] for row in metadata['records']}):
        selected = np.zeros(len(pack['labels']), dtype=bool)
        for record in metadata['records']:
            if record['generator'] == generator:
                selected[record['packed_start']:record['packed_stop']] = True
        generators[generator] = {name:event.ranking(pack['labels'][selected], scores[name][selected]) for name in event.METHODS}
    columns = ['id', 'source_id', 'generator', 'target', 'word', 'label', 'onset', 'first']
    columns += [column for name in event.METHODS for column in (name,name+'__alarm')]
    with (directory/'test_tokens.csv').open('x',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for record in metadata['records']:
            response = json.loads((CACHE/record['directory']/'response.json').read_text())
            for index in range(record['packed_start'],record['packed_stop']):
                target = int(pack['target'][index])
                row = dict(id=record['id'],source_id=record['source_id'],generator=record['generator'],target=target,
                    word=response['token_text'][target],label=int(pack['labels'][index]),
                    onset=int(pack['onsets'][index]),first=int(pack['firsts'][index]))
                for name in event.METHODS:
                    row[name],row[name+'__alarm'] = float(scores[name][index]),int(scores[name][index]>thresholds[name])
                writer.writerow(row)
    primary, old = methods[event.PRIMARY],methods['old_native']
    budget = primary['normal_token_budget']
    gates = dict(qa_auc_at_least_point8=primary['all_tokens']['auroc']>=.8,
        qa_auc_at_least_old=primary['all_tokens']['auroc']>=old['all_tokens']['auroc'],
        ap_at_least_old=primary['all_tokens']['ap']>=old['all_tokens']['ap'],
        tp_at_old_fp_budget=budget['tp']>=old['normal_token_budget']['tp'] and budget['fp']<=baseline['fp'],
        normal_answer_alarms_not_above_old=budget['normal_answer_alarms']<=old['normal_token_budget']['normal_answer_alarms'])
    write_json(directory/'test_evaluation.json',dict(primary=event.PRIMARY,methods=methods,source_bootstrap=comparisons,
        generators=generators,counts=dict(answers=900,sources=150,tokens=len(pack['labels']),positives=int(pack['labels'].sum())),
        gates=gates,joint_gate_pass=all(gates.values()),natural_label_fits=0,historical_test_exposure=True,
        fit_thresholds_changed=False,oracle_budgets_are_not_deployment_calibration=True,
        freeze_sha256=file_hash(directory/'test_FREEZE.json')))
    print(json.dumps(dict(primary=primary['all_tokens'],matched=budget,gates=gates)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('score','evaluate'),required=True)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    (score if args.stage=='score' else evaluate)(args.output)


if __name__=='__main__':
    main()
