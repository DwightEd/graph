"""Calibrate channel tails separately before fusion; no label-based fitting."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import log_expit, logsumexp
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.source_relation.refine import fit_rank
from experiments.context_response.score import calibrate
from experiments.context_response.restore import original_threshold
from experiments.message_js.score import valid_tokens
from experiments.head_state_readout.score import OLD


def smooth_tail(value, reference, weights):
    """Logistic-mixture survival with one reference standard deviation bandwidth."""
    weights = weights/weights.sum()
    mean = weights@reference
    bandwidth = np.sqrt(weights@((reference-mean)**2))
    assert bandwidth>0
    log_survival = log_expit((reference[None,:]-value[:,None])/bandwidth)
    return -logsumexp(log_survival+np.log(weights), axis=-1)


def fit_smooth(scores, name, rows, output):
    arrays, weights = [], []
    for row in rows:
        if row['role']=='fit':
            valid = valid_tokens(row, output, OLD)
            arrays.append(scores[row['key']][name][valid])
            weights.append(np.full(valid.sum(), 1/valid.sum()))
    reference, weight = np.concatenate(arrays), np.concatenate(weights)
    return {key:smooth_tail(value[name],reference,weight) for key,value in scores.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smooth', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.first/'manifest.json')
    manifest['balance_input'] = str(args.first.resolve())
    write_json(args.output/'manifest.json', manifest)
    for name in ('features_complete.json', 'fit.json', 'interventions.json'):
        (args.output/name).symlink_to((args.first/name).resolve())
    scores = {}
    for row in manifest['records']:
        key = row['key']
        directory = args.output/key
        directory.mkdir()
        for name in ('graph.npz', 'head_readouts.npz', 'context.npz', 'responses.npz'):
            (directory/name).symlink_to((args.first/key/name).resolve())
        with np.load(args.first/key/'scores.npz') as saved:
            scores[key] = {name:saved[name] for name in saved.files}
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        task_scores = {r['key']:scores[r['key']] for r in rows}
        for graph in ('selective', 'attention', 'matched', 'direct'):
            channels = ('source_z','rejection_z') if graph=='direct' else (graph+'_source',graph+'_rejection')
            fitted = fit_smooth if args.smooth else fit_rank
            ranks = [fitted(task_scores, name, rows, args.output) for name in channels]
            for key in task_scores:
                values = np.stack([rank[key] for rank in ranks], -1)
                # Missing source observations stay unavailable, including natural traces.
                values[~np.isfinite(task_scores[key][channels[0]]),0] = np.nan
                prefix = 'smooth_' if args.smooth else 'balanced_'
                task_scores[key][prefix+graph] = np.max(values, axis=-1)
        thresholds[task] = calibrate(args.output, rows, task_scores)
        thresholds[task]['original_full_reference_fixed'] = original_threshold(task)
        for key, values in task_scores.items():
            np.savez_compressed(args.output/key/'scores.npz', **values)
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(status='complete', main=prefix+'selective',
        methods=list(next(iter(scores.values()))), labels_used=False, prior_discovery_label_informed=True,
        meaning='equal-source mixture tail scores, not normal-null p-values or factuality probabilities',
        calibration='logistic mixture, bandwidth=source-equal std' if args.smooth else 'empirical CDF',
        change='one fixed follow-up: separate channel tail calibration; no weight/threshold search'))


if __name__=='__main__':
    main()
