"""Freeze pilot readout, capture every official test answer, then evaluate."""
import argparse
from pathlib import Path
from time import perf_counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'teaching/state_audit/src'))
import joblib
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import fit_cdf, percentile, valid_tokens
from experiments.head_state_readout.score import OLD
from experiments.native_support.dual_state.scoring import window_mean
from .sparse import robust_deviation, sparse_mean
from .stream import capture_answer

TASKS = ('QA', 'Summary', 'Data2txt')
NEW = ('sparse_response', 'sparse_address', 'sparse_joint')
METHODS = ('source_route_fixed', *NEW, *(name+'_fused' for name in NEW))


def raw_features(values):
    values = values.transpose(2, 0, 1, 3).reshape(values.shape[2], 1024, -1)
    effects = np.sign(values[..., :2])*np.log1p(np.abs(values[..., :2]))
    return np.concatenate((effects, values[..., 8:10]), axis=-1)


def measure_scores(values, reference):
    deviation = robust_deviation(reference, raw_features(values))
    response = np.log1p(deviation[..., :2]).max(-1)
    address = np.log1p(deviation[..., 2:]).max(-1)
    return dict(sparse_response=sparse_mean(response), sparse_address=sparse_mean(address),
                sparse_joint=sparse_mean(np.sqrt(response*address)))


def freeze(args):
    args.output.mkdir(exist_ok=False)
    pilot = read_json(args.pilot/'manifest.json')
    population = read_json(args.source/'manifest.json')
    records = [row for row in population['records'] if row['split']=='test']
    test_sources = {r['source_id'] for r in records}
    reference_sources = {r['source_id'] for r in pilot['records'] if r['role'] in ('fit', 'dev')}
    assert not test_sources & reference_sources
    assert len(records) == 2700 and len(test_sources) == 450
    thresholds = read_json(args.pilot/'thresholds.json')
    fitted = {}
    for task in TASKS:
        rows = [r for r in pilot['records'] if r['task']==task and r['role']=='fit']
        references, weights = [], []
        calibration = {name: [] for name in (*NEW, 'source_pair', 'raw_route_offline_mean')}
        for row in rows:
            valid = valid_tokens(row, args.pilot, OLD)
            indices = np.flatnonzero(valid)
            chosen = indices[np.linspace(0, len(indices)-1, 64).astype(int)]
            values = np.load(args.pilot/row['key']/'responses.npz')['values']
            references.append(raw_features(values)[chosen])
            weights.append(np.full(valid.sum(), 1/valid.sum()))
            with np.load(args.pilot/row['key']/'scores.npz') as saved:
                for name in NEW:
                    calibration[name].append(saved[name][valid])
            with np.load(Path(pilot['route_base'])/row['key']/'scores.npz') as saved:
                for name in ('source_pair', 'raw_route_offline_mean'):
                    calibration[name].append(saved[name][valid])
        fitted[task] = dict(reference=np.concatenate(references),
            ranks={name: fit_cdf(np.concatenate(value), np.concatenate(weights))
                   for name, value in calibration.items()},
            thresholds={name: thresholds[task][name] for name in METHODS})
    joblib.dump(fitted, args.output/'frozen_readout.joblib')
    write_json(args.output/'manifest.json', dict(records=records, model=pilot['model'],
        source=str(args.source.resolve()), pilot=str(args.pilot.resolve()),
        methods=METHODS, main='sparse_joint_fused', labels_used=False,
        reference_sources=sorted(reference_sources), test_sources=sorted(test_sources),
        fit_sources_per_task=4, dev_sources_per_task=4,
        status='frozen before full-test capture; exploratory known test population',
        storage='all 1024 physical-head response/JS coordinates; dense key arrays transient in RAM',
        scope='observer fixed-past native replay, not original generator states',
        selection='sparse joint fused chosen after exposed regression; no full test tuning'))
    write_json(args.output/'thresholds.json', {task: fitted[task]['thresholds'] for task in TASKS})


def score_answer(directory, response, task, fitted, source_directory):
    values = np.load(directory/'responses.npz')['values']
    scores = measure_scores(values, fitted['reference'])
    raw_route = np.load(directory/'readouts.npz')['raw_route']
    with np.load(source_directory/'scores.npz') as saved:
        pair = saved['source_pair_unit_mean']
    source_rank = percentile(pair, fitted['ranks']['source_pair'])
    route_rank = percentile(window_mean(raw_route, 16, offline=True), fitted['ranks']['raw_route_offline_mean'])
    scores['source_route_fixed'] = .75*source_rank+.25*route_rank
    for name in NEW:
        rank = percentile(scores[name], fitted['ranks'][name])
        scores[name+'_fused'] = .75*scores['source_route_fixed']+.25*rank
    for values in scores.values():
        assert np.isfinite(values).all() and len(values)==len(response['answer_ids'])
    np.savez_compressed(directory/'scores.npz', **scores, token_ids=response['answer_ids'])


def capture(args):
    from experiments.decision_risk_flow.run import load_model
    manifest = read_json(args.output/'manifest.json')
    fitted = joblib.load(args.output/'frozen_readout.joblib')
    model = load_model(manifest['model'])
    started = perf_counter()
    newly_completed = 0
    for index, row in enumerate(manifest['records']):
        directory = args.output/'responses'/row['id']
        if (directory/'scores.npz').exists():
            continue
        source_directory = Path(manifest['source'])/row['directory']
        response = read_json(source_directory/'response.json')
        source = read_json(Path(manifest['source'])/row['source_file'])
        if not (directory/'complete.json').exists():
            directory.parent.mkdir(exist_ok=True)
            # Failed partial runs are retained; the caller must use a new output.
            capture_answer(model, source['prompt_with_source'], response['answer_ids'],
                source['source_mask'], response['special_ids'], directory, args.batch)
        score_answer(directory, response, row['task'], fitted[row['task']], source_directory)
        newly_completed += 1
        write_json(args.output/'progress.json', dict(status='running', completed=index+1,
            newly_completed_this_session=newly_completed,
            total=len(manifest['records']), last_id=row['id'], seconds=perf_counter()-started))
        print(dict(done=index+1, total=len(manifest['records']), id=row['id'], task=row['task'],
                   seconds=round(perf_counter()-started, 1)), flush=True)
    write_json(args.output/'test_frozen.json', dict(status='complete', answers=len(manifest['records']),
        tokens=sum(r['tokens'] for r in manifest['records']), last_session_seconds=perf_counter()-started,
        record_capture_seconds=sum(read_json(args.output/'responses'/r['id']/'complete.json')['seconds'] for r in manifest['records']),
        labels_used=False))


def evaluate(args):
    from experiments.unsupervised_graph.data import load_inputs
    from experiments.probabilistic_detection.data import evaluation_labels
    from experiments.probabilistic_detection.evaluation import evaluate_all, source_bootstrap
    read_json(args.output/'test_frozen.json')
    manifest = read_json(args.output/'manifest.json')
    thresholds = read_json(args.output/'thresholds.json')
    methods = manifest['methods']
    for task in TASKS:
        pack, metadata = load_inputs(args.packs, task, 'test')
        scores = {name: np.empty(len(pack['target'])) for name in methods}
        for row in metadata['records']:
            region = slice(row['packed_start'], row['packed_stop'])
            selected = pack['target'][region]
            with np.load(args.output/'responses'/row['id']/'scores.npz') as saved:
                assert np.array_equal(saved['token_ids'][selected], pack['token_id'][region])
                for name in methods:
                    scores[name][region] = saved[name][selected]
        directory = args.output/task
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/'test_scores.npz', **scores)
        pack.update(evaluation_labels(Path(manifest['source']), pack, metadata))
        metrics = evaluate_all(pack, scores, thresholds[task])
        for metric in metrics.values():
            metric['threshold_rule'] = 'four source-equal unlabeled dev mixtures, 95th percentile strict gt'
        old_directory = Path('outputs/unsupervised_graph_20260928')/task
        with np.load(old_directory/'test_scores.npz') as saved:
            original = saved['fixed_unsupervised']
        old_threshold = read_json(old_directory/'selection.json')['mixed_thresholds']['fixed_unsupervised']
        metrics.update(evaluate_all(pack, {'original_full_reference_fixed': original},
                                    {'original_full_reference_fixed': old_threshold}))
        metrics['original_full_reference_fixed']['threshold_rule'] = 'previous full training reference and unlabeled development mixture95'
        uncertainty = dict(same_reference=source_bootstrap(pack, scores[manifest['main']], scores['source_route_fixed']),
            original_full_reference=source_bootstrap(pack, scores[manifest['main']], original))
        write_json(directory/'metrics.json', metrics)
        write_json(directory/'bootstrap.json', uncertainty)
        print(task, {name: metric['auroc'] for name, metric in metrics.items()}, flush=True)
    write_json(args.output/'evaluation_complete.json', dict(status='complete', tasks=TASKS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('freeze', 'capture', 'evaluate', 'all', 'resume'), default='all')
    parser.add_argument('--pilot', type=Path, default=Path('outputs/context_response_20260928_v2'))
    parser.add_argument('--source', type=Path, default=Path('outputs/native_support_ragtruth_all/source_first_v1'))
    parser.add_argument('--packs', type=Path, default=Path('outputs/probabilistic_detection_20260928_full/packs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()
    if args.stage in ('freeze', 'all'):
        freeze(args)
    if args.stage in ('capture', 'all', 'resume'):
        capture(args)
    if args.stage in ('evaluate', 'all', 'resume'):
        evaluate(args)


if __name__ == '__main__':
    main()
