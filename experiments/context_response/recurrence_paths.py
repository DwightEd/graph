"""Recover measured head contributions along post-score max-min paths."""
import argparse
import csv
from pathlib import Path
import joblib
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json
from .recurrence import head_features, normalize, STEPS


def trace(seed, edges):
    value = seed.copy()
    paths = [[t] for t in range(len(seed))]
    for _ in range(STEPS):
        updated, next_paths = value.copy(), [p.copy() for p in paths]
        for lag, edge in enumerate(edges, 1):
            for t, weight in enumerate(edge):
                for receiver, sender in ((t+lag, t), (t, t+lag)):
                    candidate = min(weight, value[sender])
                    if candidate>updated[receiver]:
                        updated[receiver] = candidate
                        next_paths[receiver] = [receiver]+paths[sender]
        value, paths = updated, next_paths
    return value, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    fitted = joblib.load(args.output/'frozen_readout.joblib')
    with (args.output/'propagation_tokens.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['method']=='recurrence_offline'
                and r['change'] in ('recovered', 'new_false_alarm')]
    results = []
    for row in manifest['records']:
        selected = [r for r in rows if r['key']==row['key']]
        if not selected:
            continue
        directory = args.output/row['key']
        raw = head_features(directory/'responses.npz')
        reference = fitted[row['task']]
        z = normalize(raw, reference['center'], reference['scale'])
        with np.load(directory/'recurrence.npz') as saved:
            edges = [saved[f'lag_{lag}'] for lag in range(1, 9)]
        with np.load(directory/'scores.npz') as saved:
            value, paths = trace(saved['strong_fused'], edges)
            np.testing.assert_allclose(value, saved['recurrence_offline'], atol=1e-12, rtol=0)
        for token in selected:
            t = int(token['token'])
            details = []
            for receiver, sender in zip(paths[t][:-1], paths[t][1:]):
                contribution = (z[receiver]*z[sender]).sum(-1)
                heads = np.argsort(contribution)[-8:][::-1]
                details.append(dict(receiver=receiver, sender=sender,
                    reliability=float(edges[abs(receiver-sender)-1][min(receiver,sender)]),
                    similarity=float(contribution.sum()),
                    top_contributing_heads=[dict(layer=int(h//32), head=int(h%32),
                        cosine_contribution=float(contribution[h]),
                        receiver_fields=raw[receiver,h].tolist(), sender_fields=raw[sender,h].tolist()) for h in heads]))
            results.append(dict(**token, path_from_receiver_to_seed=paths[t], edges=details))
    write_json(args.output/'propagation_head_paths.json', dict(tokens=results,
        scope='post-score audit; raw fields signed-log backward/forward response and backward/forward JS; no selected heads in scoring'))
    print('Verified and saved', len(results), 'changed-token paths with physical head contributions')


if __name__=='__main__':
    main()
