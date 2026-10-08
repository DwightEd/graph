"""Fixed nine-field historical pilot; score/freeze before official label access.

The new posterior is a conditional risk compatibility, not calibrated truth.
No fit threshold is claimed; matched false-alarm budgets are oracle diagnostics.
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
from experiments.probabilistic_detection.evaluation import pairwise_within
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_unlabeled import answer_inputs
from .source_route_refine_eval import answer_masks, alarm_counts, threshold_for_budget, transitions
from .state_graph import gated_weights, infer_states
from .unlabeled import CHANNELS, chain_weights, local_weights, reference_rank, rewire_weights


PREVIOUS = Path('outputs/source_route_refine_20261009')
DEFAULT_OUTPUT = Path('outputs/source_route_state_20261009_v2')
PLAN = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/source_route_state_20261009/EXPERIMENT_PLAN.md')
METHODS = ('old_unary', 'old_native', 'state_native', 'state_gated', 'state_massmatched',
           'state_chain', 'state_rewired', 'state_shuffle', 'paper_chain')
PRIMARY = 'state_gated'
COUPLING = float(np.log(.993 / .007))


def mass_match(weights, target):
    """Keep each receiver's total raw edge mass while controlling its allocation."""
    mass = weights.sum(axis=1)
    scale = np.divide(target.sum(axis=1), mass, out=np.zeros_like(mass), where=mass > 0)
    return weights * scale[:, None]


def graph_controls(attention, heads):
    """Use full physical-head messages only to gate continuity, never the unary."""
    native = local_weights(attention)
    gated, gates = gated_weights(attention, heads)
    permutation = np.random.default_rng(42).permutation(len(attention)) + 1
    shuffled_heads = heads.copy()
    shuffled_heads[:, 1:] = heads[:, permutation]
    shuffled, _ = gated_weights(attention, shuffled_heads)
    graphs = dict(state_native=native, state_gated=gated,
        state_massmatched=mass_match(native, gated), state_chain=chain_weights(gated),
        state_rewired=rewire_weights(gated), state_shuffle=mass_match(shuffled, gated))
    paper = np.zeros((len(attention), 1))
    paper[1:, 0] = 1.
    graphs['paper_chain'] = paper
    return graphs, gates


def snapshot_code(directory):
    """Bind and copy every imported project Python dependency, including evaluation."""
    root = Path.cwd().resolve()
    files = {Path(module.__file__).resolve() for module in list(sys.modules.values())
             if getattr(module, '__file__', None) and Path(module.__file__).is_absolute()
             and str(module.__file__).endswith('.py')}
    files.add(Path(__file__).resolve())
    hashes = {}
    for path in sorted(files):
        if not path.is_relative_to(root):
            continue
        relative = path.relative_to(root)
        target = directory / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path)] = file_hash(path)
    return hashes


def pilot_inputs():
    """Reuse the complete prior pilot identities without reading its evaluation."""
    frozen = json.loads((PREVIOUS / 'pilot_FREEZE.json').read_text())
    names = ('pilot_pack.npz', 'pilot_metadata.json', 'pilot_scores.npz',
             'scalar_reference.npz', 'REFINE_REFERENCE.json')
    for name in names:
        if file_hash(PREVIOUS / name) != frozen['hashes'][name]:
            raise ValueError(f'Previous frozen input changed: {name}')
    with np.load(PREVIOUS / names[0]) as saved:
        pack = dict(saved)
    metadata = json.loads((PREVIOUS / names[1]).read_text())
    with np.load(PREVIOUS / names[2]) as saved:
        baseline = {name: saved[name].copy() for name in METHODS[:2]}
    with np.load(PREVIOUS / names[3]) as saved:
        reference = dict(saved)
    return pack, metadata, baseline, reference


def bind_inputs(directory, metadata, code_hashes):
    """Hash GT bytes before inference; their contents are not parsed until evaluate."""
    files = [PLAN, CACHE / 'manifest.json', OUTPUT / 'CAPTURE_PROTOCOL.json']
    files += [PREVIOUS / name for name in ('pilot_pack.npz', 'pilot_metadata.json',
              'pilot_scores.npz', 'scalar_reference.npz', 'REFINE_REFERENCE.json')]
    for record in metadata['records']:
        files += [CACHE / record['directory'] / name for name in ('response.json', 'observations.npz')]
        files.append(OUTPUT / 'capture' / record['id'] / 'arrays.npz')
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    if 'cache_input' in manifest:
        raise ValueError('This pilot requires the recorded official annotation file')
    files.append(Path(manifest['dataset']) / 'response.jsonl')
    inputs = {str(path.resolve()): file_hash(path) for path in files}
    write_json(directory / 'INPUT_BINDING.json', dict(inputs=inputs, code=code_hashes,
        ground_truth_bytes_bound_but_not_parsed=True))
    shutil.copy2(PLAN, directory / 'EXPERIMENT_PLAN.md')


def score(directory):
    started = time.time()
    pack, metadata, scores, reference = pilot_inputs()
    directory.mkdir(parents=True, exist_ok=False)
    bind_inputs(directory, metadata, snapshot_code(directory))
    fields = {name: np.empty(len(pack['target'])) for name in METHODS[2:]}
    states, diagnostics = {}, []
    for record in metadata['records']:
        values, attention, targets, region = answer_inputs(CACHE, OUTPUT / 'capture', record, pack)
        ranks = {name: reference_rank(reference, name, values[name]) for name in CHANNELS}
        unary = .375 * (ranks['source_local'] + ranks['source_full']) + .25 * ranks['raw_route']
        np.testing.assert_allclose(unary[targets], scores['old_unary'][region], atol=2e-16, rtol=0)
        with np.load(OUTPUT / 'capture' / record['id'] / 'arrays.npz') as arrays:
            heads = arrays['group_heads'].astype(np.float64)
        graphs, gates = graph_controls(attention, heads)
        key = record['id']
        states[key + '__head_gates'] = gates
        states[key + '__unary'] = unary
        for name, weights in graphs.items():
            result = infer_states(unary, weights, COUPLING, normalize=name != 'paper_chain')
            fields[name][region] = result['state1'][targets]
            if name == PRIMARY:
                for field in ('state1', 'adjacent_joint', 'enter', 'continue_state', 'exit'):
                    states[key + '__' + field] = result[field]
                states[key + '__raw_weights'] = weights
        diagnostics.append(dict(id=key, native_raw_mass=float(graphs['state_native'].sum()),
            gated_raw_mass=float(graphs[PRIMARY].sum()), full_tokens=len(unary),
            head_gate_mean=float(gates.mean())))
        print(f'STATE SCORE {key} complete', flush=True)
    scores.update(fields)
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError('Every method must cover the entire pilot with finite scores')
    np.savez(directory / 'scores.npz', **scores)
    np.savez(directory / 'states.npz', **states)
    np.savez(directory / 'pack.npz', **pack)
    write_json(directory / 'metadata.json', metadata)
    write_json(directory / 'config.json', dict(primary=PRIMARY, methods=METHODS, coupling=COUPLING,
        coupling_basis='CORTEX v1 textual token p_stay=.993; inherited, not label selected',
        no_new_fit_threshold=True, calibrated_truth_probability=False,
        offline=True, historical_exposure=True, natural_label_fits=0, new_llm_forwards=0))
    write_json(directory / 'diagnostics.json', dict(answers=diagnostics, seconds=time.time() - started))
    names = ('scores.npz', 'states.npz', 'pack.npz', 'metadata.json', 'config.json',
             'diagnostics.json', 'INPUT_BINDING.json', 'EXPERIMENT_PLAN.md')
    write_json(directory / 'FREEZE.json', dict(status='all_predictions_frozen',
        hashes={name: file_hash(directory / name) for name in names}, labels_opened_for_score=False))
    print(f'STATE FROZEN seconds={time.time() - started:.2f}', flush=True)


def verify_freeze(directory):
    frozen = json.loads((directory / 'FREEZE.json').read_text())
    for name, expected in frozen['hashes'].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Frozen prediction artifact changed: {name}')
    binding = json.loads((directory / 'INPUT_BINDING.json').read_text())
    for category in ('inputs', 'code'):
        for name, expected in binding[category].items():
            if file_hash(Path(name)) != expected:
                raise ValueError(f'Bound {category} changed: {name}')
            if category == 'code':
                relative = Path(name).relative_to(Path.cwd().resolve())
                if file_hash(directory / 'code_snapshot' / relative) != expected:
                    raise ValueError(f'Code snapshot changed: {relative}')
    return file_hash(directory / 'FREEZE.json')


def evaluate(directory):
    """Rank frozen fields and diagnose budgets; evaluation never modifies any field."""
    freeze_hash = verify_freeze(directory)
    with np.load(directory / 'pack.npz') as saved:
        pack = dict(saved)
    with np.load(directory / 'scores.npz') as saved:
        scores = dict(saved)
    metadata = json.loads((directory / 'metadata.json').read_text())
    pack.update(evaluation_labels(CACHE, pack, metadata))
    clean, first, answers = answer_masks(pack, metadata['records'])
    old_threshold = json.loads((PREVIOUS / 'REFINE_REFERENCE.json').read_text())['thresholds']['old_native']
    old_counts = alarm_counts(pack, scores['old_native'], old_threshold, clean, first, answers)
    budgets = dict(normal_token_budget=old_counts['fp'], normal_answer_budget=old_counts['normal_answer_alarms'])
    report, thresholds = {}, {}
    for name, values in scores.items():
        healthy = clean | first
        row = dict(all_tokens=ranking(pack['labels'], values),
            within_answer_auroc=pairwise_within(pack['labels'], values, pack['answer_index']),
            strict_first_error=ranking(first[healthy].astype(int), values[healthy]),
            fixed_compatibility95=alarm_counts(pack, values, .95, clean, first, answers))
        for policy, budget in budgets.items():
            normals = values[pack['labels'] == 0] if policy == 'normal_token_budget' else [
                values[a['start']:a['stop']].max() for a in answers if a['first'] is None]
            threshold = threshold_for_budget(normals, budget)
            row[policy] = dict(alarm_counts(pack, values, threshold, clean, first, answers),
                oracle_diagnostic=True, label_assisted=True, allowed_budget=budget)
            if policy == 'normal_token_budget':
                thresholds[name] = threshold
        report[name] = row
    changed = transitions(pack['labels'], scores['old_native'], scores[PRIMARY],
                          thresholds['old_native'], thresholds[PRIMARY])
    write_tokens(directory, pack, metadata, scores, thresholds)
    write_json(directory / 'evaluation.json', dict(primary=PRIMARY, methods=report,
        baseline_to_primary=changed, freeze_sha256=freeze_hash,
        tokens=len(pack['labels']), positives=int(pack['labels'].sum()),
        no_independently_calibrated_new_threshold=True, historical_exposure=True))
    print(json.dumps({name: dict(auc=row['all_tokens']['auroc'],
        tp=row['normal_token_budget']['tp'], fp=row['normal_token_budget']['fp'])
        for name, row in report.items()}, indent=2), flush=True)


def write_tokens(directory, pack, metadata, scores, thresholds):
    columns = ['id', 'target', 'word', 'label', 'onset', 'first']
    columns += [field for name in METHODS for field in (name, name + '__alarm')]
    columns += ['enter', 'continue_state', 'exit', 'gate_mass']
    with np.load(directory / 'states.npz') as states, (directory / 'tokens.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in metadata['records']:
            key = record['id']
            response = json.loads((CACHE / record['directory'] / 'response.json').read_text())
            for index in range(record['packed_start'], record['packed_stop']):
                target = int(pack['target'][index])
                row = dict(id=key, target=target, word=response['token_text'][target],
                    label=int(pack['labels'][index]), onset=int(pack['onsets'][index]),
                    first=int(pack['firsts'][index]), gate_mass=float(states[key + '__raw_weights'][target].sum()))
                for name in METHODS:
                    row[name] = float(scores[name][index])
                    row[name + '__alarm'] = int(scores[name][index] > thresholds[name])
                for field in ('enter', 'continue_state', 'exit'):
                    row[field] = float(states[key + '__' + field][target - 1]) if target else ''
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('score', 'evaluate'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    (score if args.stage == 'score' else evaluate)(args.output)


if __name__ == '__main__':
    main()
