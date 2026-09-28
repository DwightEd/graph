"""Separate frozen conditional-calibration version reusing full new measurements."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'teaching/state_audit/src'))
import joblib
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import fit_cdf, percentile, valid_tokens
from experiments.head_state_readout.score import OLD
from experiments.head_state_readout.features import token_context
from .conditional import condition_rank
from .full import evaluate

METHODS = ('source_route_fixed', 'conditional_sparse', 'conditional_sparse_fused')
CONTEXT = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]


def freeze(args):
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    pilot = read_json(args.pilot/'manifest.json')
    thresholds = read_json(args.pilot/'thresholds.json')
    fitted = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        references, values, sources, scores, weights = [], [], [], [], []
        rows = [r for r in pilot['records'] if r['task']==task and r['role']=='fit']
        for row in rows:
            valid = valid_tokens(row, args.pilot, OLD)
            references.append(np.load(args.pilot/row['key']/'context.npz')['values'][valid][:, CONTEXT])
            with np.load(args.pilot/row['key']/'scores.npz') as saved:
                values.append(saved['sparse_joint'][valid])
                scores.append(saved['conditional_sparse'][valid])
            weights.append(np.full(valid.sum(), 1/valid.sum()))
            sources.extend([row['source_id']]*valid.sum())
        fitted[task] = dict(context=np.concatenate(references), values=np.concatenate(values),
            sources=np.asarray(sources), rank=fit_cdf(np.concatenate(scores), np.concatenate(weights)))
    joblib.dump(fitted, args.output/'frozen_readout.joblib')
    manifest.update(previous=str(args.previous.resolve()), pilot=str(args.pilot.resolve()),
        methods=METHODS, main='conditional_sparse_fused',
        status='conditional readout frozen before first full-test evaluation')
    write_json(args.output/'manifest.json', manifest)
    write_json(args.output/'thresholds.json', {task: {name: thresholds[task][name] for name in METHODS} for task in fitted})


def score(args):
    read_json(args.previous/'test_frozen.json')
    manifest = read_json(args.output/'manifest.json')
    fitted = joblib.load(args.output/'frozen_readout.joblib')
    root = Path(manifest['source'])
    (args.output/'responses').mkdir(exist_ok=True)
    for row in manifest['records']:
        previous = args.previous/'responses'/row['id']
        directory = args.output/'responses'/row['id']
        directory.mkdir(exist_ok=True)
        with np.load(previous/'readouts.npz') as saved:
            confidence, route = saved['confidence'], saved['raw_route']
        context = token_context(dict(row, root=str(root), kind='observer'), manifest, confidence, route)[:, CONTEXT]
        reference = fitted[row['task']]
        with np.load(previous/'scores.npz') as saved:
            baseline = saved['source_route_fixed']
            value = saved['sparse_joint']
            ids = saved['token_ids']
        conditional = condition_rank(context, value, reference['context'], reference['values'],
                                     reference['sources'], row['source_id'])
        fused = .75*baseline+.25*percentile(conditional, reference['rank'])
        np.savez_compressed(directory/'scores.npz', source_route_fixed=baseline,
            conditional_sparse=conditional, conditional_sparse_fused=fused, token_ids=ids)
    write_json(args.output/'test_frozen.json', dict(status='complete', labels_used=False,
        answers=len(manifest['records']), tokens=sum(r['tokens'] for r in manifest['records'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('freeze', 'score-evaluate'), required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('outputs/context_response_20260928_v3'))
    parser.add_argument('--packs', type=Path, default=Path('outputs/probabilistic_detection_20260928_full/packs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage=='freeze':
        freeze(args)
    else:
        score(args)
        evaluate(args)


if __name__ == '__main__':
    main()
