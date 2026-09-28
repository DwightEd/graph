"""Frozen route comparisons; labels never enter scores or calibration."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import inputs, read_json, write_json
from experiments.message_js.score import fit_cdf, percentile, valid_tokens
from experiments.native_support.dual_state.scoring import window_mean

METHODS = ('net_route_offline_mean', 'net_route', 'raw_route', 'raw_route_offline_mean',
           'equal_head_route', 'payload_use_js', 'source_pair', 'source_net', 'source_route',
           'source_net_fixed', 'source_route_fixed')
FIELDS = ('norm_route', 'net_route', 'source_cancel', 'history_cancel', 'read_payload_js', 'payload_use_js')


def route(mass):
    denominator = mass.sum((1, 2))
    difference = (mass[:, 1] - mass[:, 0]).sum(1)
    return np.mean(difference / np.maximum(denominator, 1e-30), axis=0)


def measure_scores(output, row):
    directory = output / row['key']
    with np.load(directory / 'measurements.npz') as saved:
        mass, divergence = saved['mass'].astype(np.float64), saved['source_js']
    gram = np.load(directory / 'head_gram.npy', mmap_mode='r')
    net = np.sqrt(np.maximum(np.diagonal(gram, axis1=-2, axis2=-1), 0)).transpose(0, 1, 3, 2)
    raw, integrated = route(mass), route(net)
    head_route = (mass[:, 1] - mass[:, 0]) / np.maximum(mass.sum(1), 1e-30)
    head_net = (net[:, 1] - net[:, 0]) / np.maximum(net.sum(1), 1e-30)
    cancellation = 1 - np.divide(net, mass, out=np.ones_like(mass), where=mass > 0)
    heads = np.stack((head_route, head_net, cancellation[:, 0], cancellation[:, 1],
                      divergence[..., 0], divergence[..., 1]), -1)
    np.savez_compressed(directory / 'head_features.npz', values=heads, fields=FIELDS)
    score = dict(raw_route=raw, net_route=integrated,
        raw_route_offline_mean=window_mean(raw, 16, offline=True),
        net_route_offline_mean=window_mean(integrated, 16, offline=True),
        equal_head_route=head_route.mean((0, 1)),
        payload_use_js=np.nanquantile(divergence[..., 1].reshape(1024, len(raw)), .9, axis=0))
    if row['kind'] == 'observer':
        with np.load(Path(row['root']) / row['directory'] / 'scores.npz') as old:
            score['source_pair'] = old['source_pair_unit_mean'].copy()
            score['original_cached_route'] = old['raw_route'].copy()
    else:
        score['source_pair'] = np.full(len(raw), np.nan)
    return score


def reference(rows, scores, output, old, field):
    values, weights = [], []
    for row in rows:
        if row['role'] != 'fit':
            continue
        valid = valid_tokens(row, output, old)
        values.append(scores[row['key']][field][valid])
        weights.append(np.full(valid.sum(), 1 / valid.sum()))
    return fit_cdf(np.concatenate(values), np.concatenate(weights))


def fuse_source(rows, scores, output, old):
    source = reference(rows, scores, output, old, 'source_pair')
    for name, channel in (('source_net', 'net_route_offline_mean'), ('source_route', 'raw_route_offline_mean')):
        fitted = reference(rows, scores, output, old, channel)
        for row in rows:
            score = scores[row['key']]
            rank = percentile(score[channel], fitted)
            source_rank = percentile(score['source_pair'], source)
            centered = rank.copy()
            if row['kind'] == 'observer':
                _, response = inputs(row)
                for unit in response['units']:
                    region = slice(unit['start'], unit['stop'])
                    centered[region] -= rank[region].mean()
            score[name] = source_rank + .1 * centered
            score[name + '_fixed'] = .75 * source_rank + .25 * rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old', type=Path, default=Path('outputs/transport_topology_cases_20260928'))
    args = parser.parse_args()
    read_json(args.output / 'capture_complete.json')
    records = read_json(args.output / 'manifest.json')['records']
    scores = {row['key']: measure_scores(args.output, row) for row in records}
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [row for row in records if row['task'] == task]
        fuse_source(rows, scores, args.output, args.old)
        values, weights = {name: [] for name in METHODS}, []
        for row in rows:
            np.savez_compressed(args.output / row['key'] / 'scores.npz', **scores[row['key']])
            if row['role'] != 'dev':
                continue
            valid = valid_tokens(row, args.output, args.old)
            weights.append(np.full(valid.sum(), 1 / valid.sum()))
            for name in METHODS:
                values[name].append(scores[row['key']][name][valid])
        thresholds[task] = {}
        for name in METHODS:
            ordered, cumulative = fit_cdf(np.concatenate(values[name]), np.concatenate(weights))
            thresholds[task][name] = float(ordered[np.searchsorted(cumulative[1:], .95)])
    write_json(args.output / 'thresholds.json', thresholds)
    write_json(args.output / 'scores_frozen.json', dict(status='complete', methods=METHODS,
        main='net_route_offline_mean', labels_used=False, head_selection=False,
        threshold='4 dev sources per task; source-equal unlabeled mixture .95',
        source_fusion='pair rank + .1 centered route rank; fixed: .75 pair rank + .25 route rank; 4 fit sources per task',
        natural_source_likelihood='unavailable, preserve NaN'))


if __name__ == '__main__':
    main()
