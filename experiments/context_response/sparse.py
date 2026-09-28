"""Sparse head readout without assigning factual meaning to derivative signs."""
import argparse
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import valid_tokens
from experiments.head_state_readout.score import OLD
from experiments.source_relation.refine import fit_rank
from .score import calibrate


def robust_deviation(reference, values):
    center = np.median(reference, axis=0)
    scale = np.quantile(reference, .75, axis=0)-np.quantile(reference, .25, axis=0)
    # No extrapolated percentile ties: retain distances beyond the fit range.
    scale = np.maximum(scale, 1e-3)
    return np.abs((values-center)/scale)


def sparse_mean(values, heads=8):
    # A fixed sparse alternative; all physical heads participate in selection.
    return np.partition(values, -heads, axis=1)[:, -heads:].mean(axis=1)


def score_task(output, previous, rows):
    raw, scores = {}, {}
    for row in rows:
        key = row['key']
        values = np.load(previous/key/'responses.npz')['values']
        values = values.transpose(2, 0, 1, 3).reshape(values.shape[2], 1024, -1)
        # Each head keeps both directed effects and its two address JS values.
        effects = np.sign(values[..., :2])*np.log1p(np.abs(values[..., :2]))
        raw[key] = np.concatenate((effects, values[..., 8:10]), axis=-1)
        with np.load(previous/key/'scores.npz') as saved:
            scores[key] = {name: saved[name] for name in saved.files}
    references = []
    for row in rows:
        if row['role'] == 'fit':
            available = np.flatnonzero(valid_tokens(row, output, OLD))
            # Equal number of deterministic positions per source.
            indices = available[np.linspace(0, len(available)-1, 64).astype(int)]
            references.append(raw[row['key']][indices])
    reference = np.concatenate(references)
    for key, value in raw.items():
        deviation = robust_deviation(reference, value)
        response = np.log1p(deviation[..., :2]).max(-1)
        address = np.log1p(deviation[..., 2:]).max(-1)
        scores[key]['sparse_response'] = sparse_mean(response)
        scores[key]['sparse_address'] = sparse_mean(address)
        scores[key]['sparse_joint'] = sparse_mean(np.sqrt(response*address))
    for name in ('sparse_response', 'sparse_address', 'sparse_joint'):
        rank = fit_rank(scores, name, rows, output)
        for key in scores:
            scores[key][name+'_fused'] = .75*scores[key]['source_route_fixed']+.25*rank[key]
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output/'manifest.json', manifest)
    for row in manifest['records']:
        directory = args.output/row['key']
        directory.mkdir()
        for name in ('responses.npz', 'context.npz'):
            (directory/name).symlink_to((args.previous/row['key']/name).resolve())
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in manifest['records'] if r['task'] == task]
        scores = score_task(args.output, args.previous, rows)
        thresholds[task] = calibrate(args.output, rows, scores)
        for key, values in scores.items():
            np.savez_compressed(args.output/key/'scores.npz', **values)
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(status='complete', main='sparse_joint',
        methods=list(next(iter(scores.values()))), labels_used=False, head_selection='top8 per token, no fixed label-picked heads',
        threshold='source-equal unlabeled dev mixture95 strict greater',
        meaning='unusual response and address change in the same head, not factual support'))


if __name__ == '__main__':
    main()
