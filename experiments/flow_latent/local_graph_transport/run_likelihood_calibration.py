"""Freeze seven likelihood-calibration fields on complete fit, pilot and QA DEV.

Only source-effect references change. The raw routing, fusion and old graph
remain fixed; natural labels are loaded by the separate evaluate stage.
"""
import argparse
import csv
import hashlib
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
from .likelihood_calibration import conditional_cdf, fit_conditional_cdf, shuffle_condition
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_state_graph import snapshot_code
from .run_unlabeled import answer_inputs, weighted_reference_threshold
from .source_route_refine_eval import answer_masks, alarm_counts, method_report, transitions
from .unlabeled import CHANNELS, equal_source_weights, graph_fields, local_weights, reference_rank


PREVIOUS = Path('outputs/source_route_refine_20261009')
DEFAULT_OUTPUT = Path('outputs/source_likelihood_calibration_20261009')
PLAN = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/source_likelihood_calibration_20261009/EXPERIMENT_PLAN.md')
METHODS = ('old_unary', 'old_native', 'likelihood_unary', 'likelihood_native',
           'local_likelihood_native', 'full_likelihood_native', 'reference_shuffle_native')
PRIMARY = 'likelihood_native'


def phase_inputs(phase):
    """Open only pure token identity packs and frozen preexisting predictions."""
    frozen = json.loads((PREVIOUS / f'{phase}_FREEZE.json').read_text())
    names = [f'{phase}_{suffix}' for suffix in ('pack.npz', 'metadata.json', 'scores.npz')]
    for name in names:
        if file_hash(PREVIOUS / name) != frozen['hashes'][name]:
            raise ValueError(f'Previous input changed: {name}')
    with np.load(PREVIOUS / names[0]) as saved:
        pack = dict(saved)
    if set(pack) != {'target', 'token_id', 'source_index', 'answer_index', 'unit_index'}:
        raise ValueError('Only pure identity fields may enter scoring')
    metadata = json.loads((PREVIOUS / names[1]).read_text())
    with np.load(PREVIOUS / names[2]) as saved:
        baseline = {name: saved[name].copy() for name in METHODS[:2]}
    return pack, metadata, baseline


def native_likelihood(record, expected_ids):
    """Read the two matching actual-token likelihoods, not a sparse-layer lens."""
    path = CACHE / record['directory'] / 'with_source.npz'
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['token_id'], expected_ids)
        return {name: saved[name].copy() for name in ('local', 'full')}


def fit_references(pack, metadata):
    batches = {name: [] for name in ('source_local', 'source_full', 'local', 'full')}
    for record in metadata['records']:
        region = slice(record['packed_start'], record['packed_stop'])
        targets = pack['target'][region]
        with np.load(CACHE / record['directory'] / 'observations.npz') as saved:
            ids = saved['token_id'].copy()
            np.testing.assert_array_equal(ids[targets], pack['token_id'][region])
            for name in ('source_local', 'source_full'):
                batches[name].append(saved[name][targets])
        likelihoods = native_likelihood(record, ids)
        for name in ('local', 'full'):
            batches[name].append(likelihoods[name][targets])
    values = {name: np.concatenate(rows) for name, rows in batches.items()}
    references = {}
    for name in ('local', 'full'):
        condition = values[name]
        for suffix, likelihood in (('', condition), ('shuffle_', shuffle_condition(condition, pack['source_index']))):
            reference = fit_conditional_cdf(values['source_' + name], likelihood, pack['source_index'])
            references.update({suffix + name + '__' + key: value for key, value in reference.items()})
    return references


def view_reference(references, name):
    prefix = name + '__'
    return {key[len(prefix):]: value for key, value in references.items() if key.startswith(prefix)}


def fields_for_answer(global_reference, references, values, likelihood, attention):
    ranks = {name: reference_rank(global_reference, name, values[name]) for name in CHANNELS}
    conditional, shuffled = {}, {}
    for name in ('local', 'full'):
        conditional[name] = conditional_cdf(view_reference(references, name), values['source_' + name], likelihood[name])
        shuffled[name] = conditional_cdf(view_reference(references, 'shuffle_' + name), values['source_' + name], likelihood[name])
    blend = lambda local, full: .375 * (local + full) + .25 * ranks['raw_route']
    unaries = dict(old=blend(ranks['source_local'], ranks['source_full']),
        likelihood=blend(conditional['local'], conditional['full']),
        local_likelihood=blend(conditional['local'], ranks['source_full']),
        full_likelihood=blend(ranks['source_local'], conditional['full']),
        reference_shuffle=blend(shuffled['local'], shuffled['full']))
    scores = dict(old_unary=unaries['old'], likelihood_unary=unaries['likelihood'])
    graph = dict(native=local_weights(attention))
    for name, unary in unaries.items():
        fields, _ = graph_fields(unary, graph)
        scores[name + '_native'] = fields['native_huber']
    return scores, dict(ranks, local_conditional=conditional['local'], full_conditional=conditional['full'])


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def score_records(global_reference, references, pack, metadata, baseline):
    scores = {name: np.full(len(pack['target']), np.nan) for name in METHODS}
    ranks, inputs, attention_hashes = {}, {}, {}
    max_difference = 0.
    for index, record in enumerate(metadata['records']):
        values, attention, targets, region = answer_inputs(CACHE, OUTPUT / 'capture', record, pack)
        likelihood = native_likelihood(record, values['token_id'])
        result, diagnostics = fields_for_answer(global_reference, references, values, likelihood, attention)
        for name in METHODS:
            scores[name][region] = result[name][targets]
        for name, value in diagnostics.items():
            ranks.setdefault(name, np.empty(len(pack['target'])))[region] = value[targets]
        for name in METHODS[:2]:
            max_difference = max(max_difference, float(np.abs(scores[name][region] - baseline[name][region]).max()))
        if max_difference > 1e-8:
            raise ValueError(f"Old baseline changed: {record['id']} {max_difference}")
        for name in ('observations.npz', 'with_source.npz', 'response.json'):
            path = CACHE / record['directory'] / name
            inputs[str(path.resolve())] = file_hash(path)
        path = OUTPUT / 'capture' / record['id'] / 'arrays.npz'
        attention_hashes[str(path.resolve())] = dict(native_posttoken_local_attention=array_hash(attention),
            answer_ids=array_hash(values['token_id']), recipe='local_attention[0,1:]; answer_ids[:]')
        if (index + 1) % 300 == 0:
            print(f'CALIBRATE SCORE {index + 1}/{len(metadata["records"])}', flush=True)
    if not all(np.isfinite(values).all() for values in (*scores.values(), *ranks.values())):
        raise ValueError('Incomplete source-calibration population')
    return scores, ranks, dict(files=inputs, capture_fields=attention_hashes, old_max_difference=max_difference)


def verify(directory, phase):
    frozen = json.loads((directory / f'{phase}_FREEZE.json').read_text())
    for category in ('artifacts', 'dependencies'):
        for path, expected in frozen[category].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f'Changed frozen {category}: {path}')
    inputs = json.loads((directory / f'{phase}_inputs.json').read_text())
    for path, expected in inputs['files'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Changed consumed source input: {path}')
    for path, expected in inputs['capture_fields'].items():
        with np.load(path) as arrays:
            actual = dict(native_posttoken_local_attention=array_hash(arrays['local_attention'][0, 1:]),
                          answer_ids=array_hash(arrays['answer_ids']))
        if any(actual[name] != expected[name] for name in actual):
            raise ValueError(f'Changed consumed capture field: {path}')
    return frozen


def run_score(directory, phase):
    started = time.time()
    pack, metadata, baseline = phase_inputs(phase)
    if phase == 'fit':
        directory.mkdir(parents=True, exist_ok=False)
        code = snapshot_code(directory)
        # Bind the executed scientific tests too, although they are not imported.
        for name in ('test_likelihood_calibration.py', 'test_source_route_refine.py', 'test_source_route_refine_eval.py'):
            path = Path(__file__).with_name(name).resolve()
            target = directory / 'code_snapshot' / path.relative_to(Path.cwd().resolve())
            shutil.copy2(path, target)
            code[str(path)] = file_hash(path)
        manifest = json.loads((CACHE / 'manifest.json').read_text())
        truth = Path(manifest['dataset']) / 'response.jsonl'
        files = [PLAN, truth, CACHE / 'manifest.json', OUTPUT / 'CAPTURE_PROTOCOL.json', PREVIOUS / 'scalar_reference.npz']
        dependencies = dict(code, **{str(path.resolve()): file_hash(path) for path in files})
        write_json(directory / 'DEPENDENCIES.json', dependencies)
        shutil.copy2(PLAN, directory / 'EXPERIMENT_PLAN.md')
        references = fit_references(pack, metadata)
        np.savez(directory / 'reference.npz', **references)
    else:
        verify(directory, 'fit')
        dependencies = json.loads((directory / 'DEPENDENCIES.json').read_text())
        with np.load(directory / 'reference.npz') as saved:
            references = dict(saved)
    with np.load(PREVIOUS / 'scalar_reference.npz') as saved:
        global_reference = dict(saved)
    scores, ranks, inputs = score_records(global_reference, references, pack, metadata, baseline)
    np.savez(directory / f'{phase}_scores.npz', **scores)
    np.savez(directory / f'{phase}_ranks.npz', **ranks)
    np.savez(directory / f'{phase}_pack.npz', **pack)
    write_json(directory / f'{phase}_metadata.json', metadata)
    write_json(directory / f'{phase}_inputs.json', inputs)
    if phase == 'fit':
        weights = equal_source_weights(pack['source_index'])
        thresholds = {name: weighted_reference_threshold(values, weights) for name, values in scores.items()}
        write_json(directory / 'CONFIG.json', dict(primary=PRIMARY, methods=METHODS, bins=16,
            thresholds=thresholds, complete_fit=True, natural_label_fit=False, new_llm_forwards=0,
            threshold_rule='equal-source mixed fit95; not normal FPR', observer='Llama3.1 offline mixed-generator QA',
            calibration='source effect conditional on matching with-source actual-token logp'))
    files = [directory / f'{phase}_{suffix}' for suffix in ('scores.npz', 'ranks.npz', 'pack.npz', 'metadata.json', 'inputs.json')]
    files += [directory / name for name in ('CONFIG.json', 'reference.npz', 'DEPENDENCIES.json', 'EXPERIMENT_PLAN.md')]
    # These previous pure identities and baseline scores were checked before use.
    files += [PREVIOUS / f'{phase}_{suffix}' for suffix in ('pack.npz', 'metadata.json', 'scores.npz')]
    write_json(directory / f'{phase}_FREEZE.json', dict(status='all_predictions_frozen',
        artifacts={str(path.resolve()): file_hash(path) for path in files}, dependencies=dependencies,
        old_max_difference=inputs['old_max_difference'], natural_labels_opened=False,
        seconds=time.time() - started, command=sys.argv, python=sys.version,
        numpy=np.__version__, torch=torch.__version__))
    print(f'CALIBRATION {phase} FROZEN seconds={time.time() - started:.2f}', flush=True)


def run_evaluate(directory, phase):
    verify(directory, phase)
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
    methods = {name: method_report(pack, values, thresholds[name], clean, first, answers, budgets)
               for name, values in scores.items()}
    comparisons = {name: source_bootstrap(pack, scores[PRIMARY], scores[name], repeats=300)
                   for name in ('old_native', 'local_likelihood_native', 'full_likelihood_native', 'reference_shuffle_native')}
    changed = {}
    for policy in ('fit_mixture95', 'normal_token_budget'):
        old, new = methods['old_native'][policy], methods[PRIMARY][policy]
        changed[policy] = transitions(pack['labels'], scores['old_native'], scores[PRIMARY], old['threshold'], new['threshold'])
    write_token_csv(directory, phase, pack, metadata, scores, thresholds)
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
    print(json.dumps({name: row['all_tokens'] for name, row in methods.items()}, indent=2), flush=True)


def write_token_csv(directory, phase, pack, metadata, scores, thresholds):
    columns = ['id', 'target', 'word', 'label', 'onset', 'first']
    columns += [field for name in METHODS for field in (name, name + '__alarm')]
    with (directory / f'{phase}_tokens.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in metadata['records']:
            response = json.loads((CACHE / record['directory'] / 'response.json').read_text())
            for index in range(record['packed_start'], record['packed_stop']):
                target = int(pack['target'][index])
                row = dict(id=record['id'], target=target, word=response['token_text'][target],
                    label=int(pack['labels'][index]), onset=int(pack['onsets'][index]), first=int(pack['firsts'][index]))
                for name in METHODS:
                    row[name] = float(scores[name][index])
                    row[name + '__alarm'] = int(scores[name][index] > thresholds[name])
                writer.writerow(row)


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
