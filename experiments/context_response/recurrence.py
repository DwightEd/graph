"""Lag-calibrated full-head recurrence and bounded max-min risk propagation."""
import argparse
from pathlib import Path
import joblib
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.head_state_readout.score import OLD
from experiments.message_js.score import valid_tokens, fit_cdf, percentile
from .score import calibrate
from .restore import original_threshold

LAGS = 8
STEPS = 3
GATE = .95
BASE = ('original_full_reference_fixed', 'strong_fused', 'sparse_joint')
METHODS = BASE + ('recurrence_offline', 'recurrence_causal', 'recurrence_shuffled',
                  'distance_offline', 'internal_recurrence')


def head_features(path):
    raw = np.load(path)['values']
    value = raw.transpose(2, 0, 1, 3).reshape(raw.shape[2], 1024, -1)
    effect = np.sign(value[..., :2])*np.log1p(np.abs(value[..., :2]))
    return np.concatenate((effect, value[..., 8:10]), axis=-1)


def normalize(value, center, scale):
    z = np.arcsinh((value-center)/scale)
    return z / np.maximum(np.linalg.norm(z.reshape(len(z), -1), axis=1), 1e-12)[:, None, None]


def similarities(z, shuffle=False):
    if shuffle:
        # Independent head permutations erase consistent physical identity,
        # retaining the four coordinates within each head and all magnitudes.
        rng = np.random.default_rng(713)
        z = np.stack([row[rng.permutation(1024)] for row in z])
    return [np.einsum('thd,thd->t', z[lag:], z[:-lag]) for lag in range(1, LAGS+1)]


def build_reference(rows, raw, output):
    selected = []
    for row in rows:
        if row['role']=='fit':
            valid = np.flatnonzero(valid_tokens(row, output, OLD))
            selected.append(raw[row['key']][valid[np.linspace(0, len(valid)-1, 64).astype(int)]])
    reference = np.concatenate(selected)
    center = np.median(reference, axis=0)
    scale = np.maximum(np.quantile(reference, .75, axis=0)-np.quantile(reference, .25, axis=0), 1e-3)
    pairs = {mode: [] for mode in ('native', 'shuffled')}
    for row in rows:
        if row['role']!='fit':
            continue
        key = row['key']
        valid = valid_tokens(row, output, OLD)
        z = normalize(raw[key], center, scale)
        for mode in pairs:
            values = similarities(z, mode=='shuffled')
            pairs[mode].append(dict(source=row['source_id'], values=[
                value[valid[lag:] & valid[:-lag]] for lag, value in enumerate(values, 1)]))
    return dict(center=center, scale=scale, pairs=pairs)


def recurrence_edges(value, fitted, source=None, shuffle=False, gate=GATE):
    z = normalize(value, fitted['center'], fitted['scale'])
    result = []
    references = fitted['pairs']['shuffled' if shuffle else 'native']
    for index, similarity in enumerate(similarities(z, shuffle)):
        ranks = []
        for reference in references:
            if reference['source']==source:
                continue
            values = np.sort(reference['values'][index])
            if len(values):
                ranks.append((np.searchsorted(values, similarity, 'left')+
                              np.searchsorted(values, similarity, 'right'))/(2*len(values)))
        rank = np.mean(ranks, axis=0)
        result.append(np.where(rank>=gate, rank, 0.))
    return result


def propagate(seed, edges, causal=False, steps=STEPS):
    """Max-min paths: no path can amplify its strongest seed or weakest edge."""
    if not np.isfinite(seed).all():
        return np.full_like(seed, np.nan), np.full(len(seed), -1, dtype=int)
    value = seed.copy()
    origin = np.arange(len(seed))
    for _ in range(steps):
        updated, updated_origin = value.copy(), origin.copy()
        for lag, edge in enumerate(edges, 1):
            candidate = np.minimum(edge, value[:-lag])
            better = candidate>updated[lag:]
            updated[lag:][better] = candidate[better]
            updated_origin[lag:][better] = origin[:-lag][better]
            if not causal:
                candidate = np.minimum(edge, value[lag:])
                better = candidate>updated[:-lag]
                updated[:-lag][better] = candidate[better]
                updated_origin[:-lag][better] = origin[lag:][better]
        value, origin = updated, updated_origin
    return value, origin


def score_one(raw, base, fitted, source=None):
    edges = recurrence_edges(raw, fitted, source)
    shuffled = recurrence_edges(raw, fitted, source, shuffle=True)
    distance = [np.full(max(0, len(raw)-lag), np.exp(-lag/32)) for lag in range(1, LAGS+1)]
    result = {name: base[name] for name in BASE}
    origins = {}
    for name, graph, causal, seed in (
        ('recurrence_offline', edges, False, base['strong_fused']),
        ('recurrence_causal', edges, True, base['strong_fused']),
        ('recurrence_shuffled', shuffled, False, base['strong_fused']),
        ('distance_offline', distance, False, base['strong_fused']),
        ('internal_recurrence', edges, False, percentile(base['sparse_joint'], fitted['internal_rank']))):
        result[name], origins[name] = propagate(seed, graph, causal)
    return result, origins, edges


def pilot(args):
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output/'manifest.json', manifest)
    raw, base = {}, {}
    for row in manifest['records']:
        key = row['key']
        directory = args.output/key
        directory.mkdir()
        for name in ('responses.npz', 'context.npz'):
            (directory/name).symlink_to((args.previous/key/name).resolve())
        raw[key] = head_features(directory/'responses.npz')
        with np.load(args.previous/key/'scores.npz') as saved:
            base[key] = {name: saved[name] for name in BASE}
    fitted, thresholds = {}, {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        fitted[task] = build_reference(rows, raw, args.output)
        values, weights = [], []
        for row in rows:
            if row['role']=='fit':
                valid = valid_tokens(row, args.output, OLD)
                values.append(base[row['key']]['sparse_joint'][valid])
                weights.append(np.full(valid.sum(), 1/valid.sum()))
        fitted[task]['internal_rank'] = fit_cdf(np.concatenate(values), np.concatenate(weights))
        scores = {}
        for row in rows:
            key = row['key']
            scores[key], origins, edges = score_one(raw[key], base[key], fitted[task], row['source_id'])
            np.savez_compressed(args.output/key/'scores.npz', **scores[key])
            np.savez_compressed(args.output/key/'recurrence.npz', **origins,
                                **{f'lag_{lag}': edge for lag, edge in enumerate(edges, 1)})
        thresholds[task] = calibrate(args.output, rows, scores)
        thresholds[task]['original_full_reference_fixed'] = original_threshold(task)
        print('scored', task, flush=True)
    joblib.dump(fitted, args.output/'frozen_readout.joblib')
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(status='complete', methods=METHODS,
        main='recurrence_offline', labels_used=False, physical_heads=1024, lag=LAGS, steps=STEPS,
        gate=GATE, causal_primary=False, interpretation='bounded max-min paths, not hallucination probabilities'))


def full(args):
    from .full import evaluate
    read_json(args.previous/'evaluation_complete.json')
    args.output.mkdir(exist_ok=False)
    pilot_manifest = read_json(args.pilot/'manifest.json')
    selection = read_json(args.pilot/'scores_frozen.json')
    manifest = read_json(args.previous/'manifest.json')
    manifest.update(previous=str(args.previous.resolve()), pilot=str(args.pilot.resolve()),
                    methods=tuple(selection['methods'])+('source_route_fixed',), main=selection['main'])
    write_json(args.output/'manifest.json', manifest)
    thresholds = read_json(args.pilot/'thresholds.json')
    previous_thresholds = read_json(args.previous/'thresholds.json')
    for task in thresholds:
        thresholds[task]['source_route_fixed'] = previous_thresholds[task]['source_route_fixed']
    write_json(args.output/'thresholds.json', thresholds)
    fitted = joblib.load(args.pilot/'frozen_readout.joblib')
    # Both geometry and score thresholds were frozen before these predictions.
    for index, row in enumerate(manifest['records']):
        task = row['task']
        assert row['source_id'] not in {r['source_id'] for r in pilot_manifest['records']
                                       if r['role'] in ('fit', 'dev')}
        directory = args.output/'responses'/row['id']
        directory.mkdir(parents=True)
        raw = head_features(args.measurements/'responses'/row['id']/'responses.npz')
        with np.load(args.previous/'responses'/row['id']/'scores.npz') as saved:
            base = {name: saved[name] for name in BASE}
            ids = saved['token_ids']
            small_reference = saved['source_route_fixed']
        scores, origins, edges = score_one(raw, base, fitted[task], row['source_id'])
        if 'barrier_corroborated' in selection['methods']:
            from .recurrence_barrier import score_extra
            extra, extra_origins = score_extra(raw, base, fitted[task], row['source_id'])
            scores.update(extra)
            origins.update(extra_origins)
        scores['source_route_fixed'] = small_reference
        np.savez_compressed(directory/'scores.npz', **scores, token_ids=ids)
        np.savez_compressed(directory/'recurrence.npz', **origins,
                            **{f'lag_{lag}': edge for lag, edge in enumerate(edges, 1)})
        if (index+1)%100==0 or index+1==len(manifest['records']):
            write_json(args.output/'progress.json', dict(completed=index+1, total=len(manifest['records']),
                                                        task=task, labels_used=False))
            print('recurrence scored', index+1, '/', len(manifest['records']), task, flush=True)
    write_json(args.output/'test_frozen.json', dict(status='complete', answers=len(manifest['records']),
               tokens=sum(r['tokens'] for r in manifest['records']), labels_used=False))
    evaluate(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'full'), required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('outputs/context_response_recurrence_20260929_v1'))
    parser.add_argument('--measurements', type=Path, default=Path('outputs/context_response_full_20260928_v1'))
    parser.add_argument('--packs', type=Path, default=Path('outputs/probabilistic_detection_20260928_full/packs'))
    args = parser.parse_args()
    {'pilot': pilot, 'full': full}[args.stage](args)


if __name__=='__main__':
    main()
