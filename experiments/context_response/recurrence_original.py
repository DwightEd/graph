"""Keep the original strong baseline as seeds; change only recurrence coverage."""
import argparse
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json
from .recurrence import propagate
from .recurrence_barrier import corroborate
from .restore import original_threshold
from .score import calibrate

EXTRA = ('original_recurrence', 'original_corroborated')


def score(directory, previous):
    with np.load(previous/'scores.npz') as saved:
        values = {name: saved[name] for name in saved.files}
    with np.load(previous/'recurrence.npz') as saved:
        graph = {name: saved[name] for name in saved.files}
    edges = [graph[f'lag_{lag}'] for lag in range(1, 9)]
    seed = values['original_full_reference_fixed']
    values[EXTRA[0]], graph[EXTRA[0]] = propagate(seed, edges)
    values[EXTRA[1]], graph[EXTRA[1]] = corroborate(seed, edges)
    directory.mkdir(parents=True)
    np.savez_compressed(directory/'scores.npz', **values)
    np.savez_compressed(directory/'recurrence.npz', **graph)
    return values


def pilot(args):
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    manifest['iteration_provenance'] = 'added after first full QA/Summary response readout results; exploratory'
    write_json(args.output/'manifest.json', manifest)
    scores = {}
    for row in manifest['records']:
        key = row['key']
        scores[key] = score(args.output/key, args.previous/key)
        for name in ('responses.npz', 'context.npz'):
            (args.output/key/name).symlink_to((args.previous/key/name).resolve())
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        thresholds[task] = calibrate(args.output, rows, {r['key']: scores[r['key']] for r in rows})
        thresholds[task]['original_full_reference_fixed'] = original_threshold(task)
    write_json(args.output/'thresholds.json', thresholds)
    selection = read_json(args.previous/'scores_frozen.json')
    selection.update(methods=selection['methods']+list(EXTRA), main='original_corroborated',
                     iteration_provenance=manifest['iteration_provenance'])
    write_json(args.output/'scores_frozen.json', selection)


def full(args):
    read_json(args.previous/'evaluation_complete.json')
    read_json(args.pilot/'score_verification.json')
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest.update(methods=manifest['methods']+list(EXTRA), main='original_corroborated',
        previous=str(args.previous.resolve()), pilot=str(args.pilot.resolve()),
        iteration_provenance='baseline seed restoration after original full-test results; exploratory')
    write_json(args.output/'manifest.json', manifest)
    thresholds = read_json(args.previous/'thresholds.json')
    pilot_thresholds = read_json(args.pilot/'thresholds.json')
    for task in thresholds:
        thresholds[task].update({name: pilot_thresholds[task][name] for name in EXTRA})
    write_json(args.output/'thresholds.json', thresholds)
    for i, row in enumerate(manifest['records']):
        score(args.output/'responses'/row['id'], args.previous/'responses'/row['id'])
        if (i+1)%300==0:
            print('original seed scored', i+1, flush=True)
    write_json(args.output/'test_frozen.json', dict(status='complete', answers=len(manifest['records']),
        tokens=sum(r['tokens'] for r in manifest['records']), labels_used=False))
    evaluate_added(args, manifest, thresholds)


def evaluate_added(args, manifest, thresholds):
    """Evaluate new scores; retain byte-equivalent controls and their metrics."""
    from experiments.unsupervised_graph.data import load_inputs
    from experiments.probabilistic_detection.data import evaluation_labels
    from experiments.probabilistic_detection.evaluation import evaluate_all, source_bootstrap
    for task in ('QA', 'Summary', 'Data2txt'):
        pack, metadata = load_inputs(args.packs, task, 'test')
        with np.load(args.previous/task/'test_scores.npz') as saved:
            scores = {name: saved[name] for name in saved.files}
        added = {name: np.empty(len(pack['target'])) for name in EXTRA}
        for row in metadata['records']:
            region = slice(row['packed_start'], row['packed_stop'])
            selected = pack['target'][region]
            with np.load(args.output/'responses'/row['id']/'scores.npz') as saved:
                assert np.array_equal(saved['token_ids'][selected], pack['token_id'][region])
                for name in EXTRA:
                    added[name][region] = saved[name][selected]
                for name in scores:
                    assert np.array_equal(saved[name][selected], scores[name][region])
        scores.update(added)
        directory = args.output/task
        directory.mkdir()
        np.savez_compressed(directory/'test_scores.npz', **scores)
        pack.update(evaluation_labels(Path(manifest['source']), pack, metadata))
        metrics = read_json(args.previous/task/'metrics.json')
        new_metrics = evaluate_all(pack, added, thresholds[task])
        for metric in new_metrics.values():
            metric['threshold_rule'] = 'four source-equal unlabeled dev mixtures95 strict gt'
        metrics.update(new_metrics)
        candidate = scores[manifest['main']]
        uncertainty = dict(same_reference=source_bootstrap(pack, candidate, scores['source_route_fixed']),
            original_full_reference=source_bootstrap(pack, candidate, scores['original_full_reference_fixed']))
        write_json(directory/'metrics.json', metrics)
        write_json(directory/'bootstrap.json', uncertainty)
        print(task, {name: metric['auroc'] for name, metric in new_metrics.items()}, flush=True)
    write_json(args.output/'evaluation_complete.json', dict(status='complete', tasks=list(thresholds),
               unchanged_controls='exact packed score equality checked; previous metrics reused'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'full'), required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('outputs/context_response_recurrence_20260929_v3'))
    parser.add_argument('--packs', type=Path, default=Path('outputs/probabilistic_detection_20260928_full/packs'))
    args = parser.parse_args()
    {'pilot': pilot, 'full': full}[args.stage](args)
