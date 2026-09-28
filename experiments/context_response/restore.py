"""Keep the original full-reference unsupervised baseline instead of refitting it on four sources."""
import argparse
from pathlib import Path
import numpy as np
import joblib

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import percentile
from experiments.source_relation.refine import fit_rank
from .score import calibrate
from .full import evaluate

METHODS = ('source_route_fixed', 'original_full_reference_fixed', 'sparse_joint', 'strong_fused')
ORIGINAL = Path('outputs/unsupervised_graph_20260928')
PACKS = Path('outputs/probabilistic_detection_20260928_full/packs')


def cached_baselines(wanted, train=True):
    result = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        for split in ('train', 'test') if train else ('test',):
            metadata = read_json(PACKS/f'{task}_{split}.json')
            with np.load(PACKS/f'{task}_{split}.npz') as saved:
                positions = saved['target']
            files = {'fit': 'fit_scores', 'dev': 'development_scores'} if split=='train' else {'test': 'test_scores'}
            arrays = {part: np.load(ORIGINAL/task/(name+'.npz'))['fixed_unsupervised'] for part, name in files.items()}
            cursor = {part: 0 for part in files}
            for row in metadata['records']:
                part = row['partition']
                count = row['packed_stop']-row['packed_start']
                region = slice(cursor[part], cursor[part]+count)
                if row['id'] in wanted:
                    values = np.full(row['tokens'], np.nan)
                    selected = positions[row['packed_start']:row['packed_stop']]
                    values[selected] = arrays[part][region]
                    result[row['id']] = values
                cursor[part] += count
            assert all(cursor[part]==len(array) for part, array in arrays.items())
    return result


def original_threshold(task):
    return read_json(ORIGINAL/task/'selection.json')['mixed_thresholds']['fixed_unsupervised']


def pilot(args):
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output/'manifest.json', manifest)
    wanted = {r['id'] for r in manifest['records'] if r['kind']=='observer'}
    original = cached_baselines(wanted)
    scores = {}
    for row in manifest['records']:
        key = row['key']
        directory = args.output/key
        directory.mkdir()
        for name in ('responses.npz', 'context.npz'):
            (directory/name).symlink_to((args.previous/key/name).resolve())
        with np.load(args.previous/key/'scores.npz') as saved:
            scores[key] = {name: saved[name] for name in ('source_route_fixed', 'sparse_joint')}
        scores[key]['original_full_reference_fixed'] = original[row['id']] if row['kind']=='observer' else np.full(len(scores[key]['sparse_joint']), np.nan)
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        selected = {r['key']: scores[r['key']] for r in rows}
        rank = fit_rank(selected, 'sparse_joint', rows, args.output)
        for key, values in selected.items():
            values['strong_fused'] = .75*values['original_full_reference_fixed']+.25*rank[key]
            np.savez_compressed(args.output/key/'scores.npz', **values)
        thresholds[task] = calibrate(args.output, rows, selected)
        thresholds[task]['original_full_reference_fixed'] = original_threshold(task)
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(status='complete', methods=METHODS, main='strong_fused',
        labels_used=False, baseline='unchanged original fixed_unsupervised scores and original global unlabeled-dev threshold',
        new_threshold='four source-equal unlabeled dev mixtures95; baseline threshold is an explicit exception',
        reason='four-source empirical CDF saturated at one for QA; preserve previously effective full-reference baseline'))


def freeze_full(args):
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest.update(previous=str(args.previous.resolve()), pilot=str(args.pilot.resolve()),
        methods=METHODS, main='strong_fused', status='strong baseline fusion frozen before new full-test evaluation')
    write_json(args.output/'manifest.json', manifest)
    write_json(args.output/'thresholds.json', read_json(args.pilot/'thresholds.json'))


def score_full(args):
    read_json(args.previous/'test_frozen.json')
    manifest = read_json(args.output/'manifest.json')
    original = cached_baselines({r['id'] for r in manifest['records']}, train=False)
    fitted = joblib.load(args.previous/'frozen_readout.joblib')
    (args.output/'responses').mkdir(exist_ok=True)
    for row in manifest['records']:
        with np.load(args.previous/'responses'/row['id']/'scores.npz') as saved:
            baseline = saved['source_route_fixed']
            rank = percentile(saved['sparse_joint'], fitted[row['task']]['ranks']['sparse_joint'])
            joint, ids = saved['sparse_joint'], saved['token_ids']
        directory = args.output/'responses'/row['id']
        directory.mkdir(exist_ok=True)
        fused = .75*original[row['id']]+.25*rank
        np.savez_compressed(directory/'scores.npz', source_route_fixed=baseline,
            original_full_reference_fixed=original[row['id']], sparse_joint=joint, strong_fused=fused, token_ids=ids)
    write_json(args.output/'test_frozen.json', dict(status='complete', labels_used=False,
        answers=len(manifest['records']), tokens=sum(r['tokens'] for r in manifest['records'])))
    evaluate(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'freeze-full', 'score-full'), required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('outputs/context_response_20260928_v4'))
    parser.add_argument('--packs', type=Path, default=PACKS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'pilot': pilot, 'freeze-full': freeze_full, 'score-full': score_full}[args.stage](args)


if __name__ == '__main__':
    main()
