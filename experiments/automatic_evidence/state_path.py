"""Unsupervised source-address paths; latent states are not truth categories."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .model import odds


def source_path(profile, hidden_change, message_change):
    """Viterbi for a time-varying sticky-source factor model, without risk pooling.

    Emission factors are verified attribution profiles, not calibrated generative
    likelihoods. Refresh hazard = 1 - (1-hidden_change)*(1-message_change).
    States name source addresses; no state is designated correct/incorrect.
    """
    count, sources = profile.shape
    hazard = 1 - (1-np.clip(hidden_change, 0, 1)) * (1-np.clip(message_change, 0, 1))
    hazard = np.clip(hazard, 1e-8, 1-1e-8)
    emission = np.log(np.maximum(profile, 1e-12))
    value = emission[0] - np.log(sources)
    parents = np.zeros((count, sources), dtype=int)
    indices = np.arange(sources)
    for target in range(1, count):
        transition = np.full((sources, sources), hazard[target]/sources)
        transition[indices, indices] += 1-hazard[target]
        candidates = value[:, None] + np.log(transition)
        parents[target] = candidates.argmax(0)
        value = candidates[parents[target], indices] + emission[target]
    path = np.empty(count, dtype=int)
    path[-1] = value.argmax()
    for target in range(count-1, 0, -1):
        path[target-1] = parents[target, path[target]]
    return path, hazard


def path_spans(path):
    starts = np.r_[0, np.flatnonzero(path[1:] != path[:-1])+1]
    return [dict(start=int(start), stop=int(stop), source=int(path[start]))
            for start, stop in zip(starts, np.r_[starts[1:], len(path)])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = read_json(args.input/'manifest.json')['records']
    write_json(args.output/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input=str(args.input), primary='path_localized_odds', truth_state=False,
        motivation='v2 robust two-channel threshold collapsed to whole-answer regimes',
        algorithm='Viterbi; verified source factors; hidden/message-dependent refresh hazard',
        labels_read=False, previous_development_results_seen=True, fitted_parameters=0,
        risk='original independent token source-ablation contrast at selected path address; never pooled',
        threshold='within-answer unlabeled mixture95; no normal-FPR guarantee'))
    thresholds = {}
    for row in records:
        directory = args.output/row['key']
        directory.mkdir()
        with np.load(args.input/row['key']/'scores.npz') as previous, np.load(args.input/row['key']/'effects.npz') as effects:
            changes = previous['state_changes']
            path, hazard = source_path(previous['verified_profile'], changes[:, 1], changes[:, 2])
            independent, _ = source_path(previous['verified_profile'], np.ones(len(path)), np.ones(len(path)))
            contrast = odds(effects['masked_logp'])-odds(effects['original_logp'][:, None])
            score = contrast[np.arange(len(path)), path]
            scores = dict(path_localized_odds=score,
                          independent_verified=contrast[np.arange(len(path)), independent])
            thresholds[row['key']] = {name: float(np.quantile(value, .95)) for name, value in scores.items()}
            np.savez_compressed(directory/'scores.npz', **scores, source_path=path, refresh_hazard=hazard,
                                token_ids=previous['token_ids'])
        write_json(directory/'output_spans.json', path_spans(path))
    write_json(args.output/'thresholds.json', thresholds)
    write_json(args.output/'scores_frozen.json', dict(status='complete', labels_read=False,
        keys=[row['key'] for row in records], created_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
