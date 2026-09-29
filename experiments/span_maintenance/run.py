"""Measure cached heads and freeze token scores before separate evaluation."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from experiments.anchored_flow.edges import layer_arrays
from experiments.anchored_flow.score import rank
from experiments.native_support.dual_state.scoring import window_mean
from .state import FIELDS, maintain_layer, signed_readout

PREVIOUS = Path('outputs/anchored_flow_20260929_v1')
METHODS = ('base', 'raw', 'causal16', 'offline16', 'state_route', 'dependency_route',
           'state_both', 'head_shuffle')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def measure(row, directory):
    count = len(row['response']['answer_ids'])
    states, effects, stabilities, null_states = [], [], [], []
    weights = {name: np.zeros((count, count)) for name in ('state', 'dependency', 'shuffle')}
    rng = np.random.default_rng(713)
    for layer in range(32):
        attention, effect = layer_arrays(row, layer)
        state, kernel, dependency = maintain_layer(attention, len(row['prompt']), row['response']['answer_ids'])
        signed, stability = signed_readout(effect, len(row['prompt']))
        shuffled = np.empty_like(attention)
        for target in range(count):
            shuffled[:, target] = attention[rng.permutation(32), target]
        null, null_kernel, _ = maintain_layer(shuffled, len(row['prompt']), row['response']['answer_ids'])
        states.append(state.astype(np.float32))
        effects.append(signed.astype(np.float32))
        stabilities.append(stability.astype(np.float32))
        null_states.append(null.astype(np.float32))
        weights['state'] += kernel / 32
        weights['dependency'] += dependency / 32
        weights['shuffle'] += null_kernel / 32
    np.savez_compressed(directory / 'states.npz', fields=FIELDS, state=np.stack(states),
        signed=np.stack(effects), signed_stability=np.stack(stabilities), shuffled=np.stack(null_states))
    np.savez_compressed(directory / 'weights.npz', **weights)
    return weights


def score(row, weights, reference, directory):
    previous = PREVIOUS / row['key']
    with np.load(previous / 'observations_ranked.npz') as saved:
        source = saved['token'].copy()
        route = saved['raw_route'].copy()
        base = saved['base'].copy()
    routes = dict(raw=route, causal16=window_mean(route, 16), offline16=window_mean(route, 16, offline=True),
        state_route=weights['state'] @ route, dependency_route=weights['dependency'] @ route,
        state_both=weights['state'] @ route, head_shuffle=weights['shuffle'] @ route)
    sources = {name: source for name in routes}
    sources['state_both'] = weights['state'] @ source
    scores = {name: .75 * rank(sources[name], reference['pair']) + .25 * rank(values, reference['route'])
              for name, values in routes.items()}
    scores['base'] = base
    np.savez_compressed(directory / 'scores.npz', **scores)
    np.savez_compressed(directory / 'route_scores.npz', **routes)


def verify_weights(weights):
    for matrix in weights.values():
        np.testing.assert_allclose(matrix.sum(-1), 1, atol=1e-12)
        assert np.all(matrix >= 0)
        assert not np.any(np.triu(matrix, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    manifest = read_json(PREVIOUS / 'manifest.json')
    manifest['records'] = [row for row in manifest['records'] if row['role'] == 'case']
    write_json(args.output / 'manifest.json', manifest)
    references = joblib.load(PREVIOUS / 'references.joblib')
    write_json(args.output / 'thresholds.json', read_json(PREVIOUS / 'thresholds.json'))
    for row in manifest['records']:
        directory = args.output / row['key']
        directory.mkdir()
        weights = measure(row, directory)
        verify_weights(weights)
        score(row, weights, references[row['task']], directory)
        print('measured', row['key'], len(row['response']['answer_ids']), round(perf_counter() - started, 1), flush=True)
    write_json(args.output / 'scores_frozen.json', dict(status='complete', methods=METHODS,
        primary='state_both', labels_used=False, heads=1024, seconds=perf_counter() - started,
        semantic_state_validated=False, new_model_forward=False, causal_state=True,
        scope='18 exposed development answers; old rank scales and thresholds, not matched FPR',
        derivative_scope='current-query margin, fixed original past KV, not root provenance',
        head_control='independent per-target head permutation within each layer, seed 713'))


if __name__ == '__main__':
    main()
