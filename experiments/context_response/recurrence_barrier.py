"""State-reset barriers and distinct-origin corroboration for recurrence paths."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json
from .recurrence import BASE, METHODS, STEPS, head_features, recurrence_edges, propagate
from .score import calibrate
from .restore import original_threshold

EXTRA = ('barrier_only', 'corroborated_only', 'barrier_corroborated')


def cut_edges(edges, adjacent_rank):
    cuts = adjacent_rank<.05
    prefix = np.r_[0, np.cumsum(cuts)]
    return [np.where(prefix[lag:]-prefix[:-lag]>0, 0., edge)
            for lag, edge in enumerate(edges, 1)]


def corroborate(seed, edges):
    if not np.isfinite(seed).all():
        return np.full_like(seed, np.nan), np.full(len(seed), -1, dtype=int)
    count = len(seed)
    support = np.zeros((count, count))
    np.fill_diagonal(support, seed)
    for _ in range(STEPS):
        updated = support.copy()
        for lag, edge in enumerate(edges, 1):
            updated[lag:] = np.maximum(updated[lag:], np.minimum(edge[:, None], support[:-lag]))
            updated[:-lag] = np.maximum(updated[:-lag], np.minimum(edge[:, None], support[lag:]))
        support = updated
    if count<2:
        return seed.copy(), np.arange(count)
    # Columns identify original token seeds. A cycle cannot create a new seed.
    origin = np.argsort(support, axis=1, kind='stable')[:, -2]
    second = support[np.arange(count), origin]
    better = second>seed
    return np.maximum(seed, second), np.where(better, origin, np.arange(count))


def score_extra(raw, base, fitted, source):
    edges = recurrence_edges(raw, fitted, source)
    rank = recurrence_edges(raw, fitted, source, gate=0.)[0]
    barrier = cut_edges(edges, rank)
    seed = base['strong_fused']
    values, origins = {}, {}
    values['barrier_only'], origins['barrier_only'] = propagate(seed, barrier)
    values['corroborated_only'], origins['corroborated_only'] = corroborate(seed, edges)
    values['barrier_corroborated'], origins['barrier_corroborated'] = corroborate(seed, barrier)
    return values, origins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.previous/'manifest.json')
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output/'manifest.json', manifest)
    fitted = joblib.load(args.previous/'frozen_readout.joblib')
    scores = {}
    for row in manifest['records']:
        key = row['key']
        directory = args.output/key
        directory.mkdir()
        for name in ('responses.npz', 'context.npz'):
            (directory/name).symlink_to((args.previous/key/name).resolve())
        with np.load(args.previous/key/'scores.npz') as saved:
            scores[key] = {name: saved[name] for name in saved.files}
        with np.load(args.previous/key/'recurrence.npz') as saved:
            graph = {name: saved[name] for name in saved.files}
        raw = head_features(directory/'responses.npz')
        extra, origins = score_extra(raw, scores[key], fitted[row['task']], row['source_id'])
        scores[key].update(extra)
        graph.update(origins)
        np.savez_compressed(directory/'scores.npz', **scores[key])
        np.savez_compressed(directory/'recurrence.npz', **graph)
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        thresholds[task] = calibrate(args.output, rows, {r['key']: scores[r['key']] for r in rows})
        thresholds[task]['original_full_reference_fixed'] = original_threshold(task)
    joblib.dump(fitted, args.output/'frozen_readout.joblib')
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(status='complete', methods=METHODS+EXTRA,
        main='barrier_corroborated', labels_used=False, physical_heads=1024,
        boundary='lowest 5 percent adjacent similarity relative to unlabeled fit',
        corroboration='second distinct origin capacity, not independence of random variables'))


if __name__=='__main__':
    main()
