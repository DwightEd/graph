"""Punctuation baseline and internal confirmation, without semantic/gold boundaries."""
import argparse
import re
import shutil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from experiments.anchored_flow.edges import RAG, GSM, scatter_self
from experiments.anchored_flow.score import rank
from .run import PREVIOUS, read_json, write_json, verify_weights
from .state import FIELDS, maintain_layer

BOUNDARY = re.compile(r'(?:\n+|[.!?;:。！？；：]+["”’\)\]]*\s*)$')


def punctuation_boundaries(token_text):
    """Use only already generated text; do not reset repeatedly on trailing spaces."""
    result = np.zeros(len(token_text), dtype=bool)
    prefix = ''
    last_boundary = -1
    for target in range(1, len(token_text)):
        prefix += token_text[target - 1]
        match = BOUNDARY.search(prefix)
        if match is not None and match.start() != last_boundary:
            result[target] = True
            last_boundary = match.start()
    return result


def offline_punctuation_weights(boundaries):
    count = len(boundaries)
    weights = np.zeros((count, count))
    edges = np.r_[0, np.flatnonzero(boundaries), count]
    for start, stop in zip(edges[:-1], edges[1:]):
        weights[start:stop, start:stop] = 1 / (stop - start)
    return weights


def causal_punctuation_weights(boundaries):
    weights = np.zeros((len(boundaries), len(boundaries)))
    start = 0
    for target in range(len(boundaries)):
        if boundaries[target]:
            start = target
        weights[target, start:target + 1] = 1 / (target - start + 1)
    return weights


def attention_layer(row, layer):
    root = RAG if row['dataset'] == 'ragtruth' else GSM
    attention = np.load(root / row['key'] / 'attention.npy', mmap_mode='r')[layer]
    if row['dataset'] == 'gsm8k':
        return scatter_self(attention, len(row['prompt']))
    return attention


def measure(row, directory):
    boundaries = punctuation_boundaries(row['response']['token_text'])
    count = len(boundaries)
    kernels = {name: np.zeros((count, count)) for name in ('hard', 'soft')}
    states = {name: [] for name in kernels}
    for layer in range(32):
        attention = attention_layer(row, layer)
        for mode in kernels:
            state, kernel, _ = maintain_layer(attention, len(row['prompt']),
                row['response']['answer_ids'], boundaries, mode)
            states[mode].append(state.astype(np.float32))
            kernels[mode] += kernel / 32
    verify_weights(kernels)
    kernels['punctuation_offline'] = offline_punctuation_weights(boundaries)
    kernels['punctuation_causal'] = causal_punctuation_weights(boundaries)
    np.savez_compressed(directory / 'boundary_states.npz', fields=FIELDS, boundaries=boundaries,
        **{name: np.stack(values) for name, values in states.items()})
    np.savez_compressed(directory / 'boundary_weights.npz', **kernels)
    return kernels


def score(row, old, directory, kernels, reference):
    with np.load(old / 'scores.npz') as saved:
        scores = {name: saved[name].copy() for name in saved.files}
    with np.load(PREVIOUS / row['key'] / 'observations_ranked.npz') as saved:
        source, route = saved['token'].copy(), saved['raw_route'].copy()
    for mode, kernel in kernels.items():
        scores[mode + '_both'] = .75 * rank(kernel @ source, reference['pair']) + .25 * rank(kernel @ route, reference['route'])
    np.savez_compressed(directory / 'scores.npz', **scores)
    return list(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.previous / 'scores_frozen.json')
    manifest = read_json(args.previous / 'manifest.json')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('manifest.json', 'thresholds.json'):
        shutil.copyfile(args.previous / name, args.output / name)
    references = joblib.load(PREVIOUS / 'references.joblib')
    started = perf_counter()
    for row in manifest['records']:
        directory, old = args.output / row['key'], args.previous / row['key']
        directory.mkdir()
        for name in ('states.npz', 'weights.npz', 'route_scores.npz'):
            (directory / name).symlink_to((old / name).resolve())
        kernels = measure(row, directory)
        methods = score(row, old, directory, kernels, references[row['task']])
        print('boundary', row['key'], round(perf_counter() - started, 1), flush=True)
    write_json(args.output / 'scores_frozen.json', dict(status='complete', methods=methods,
        primary='soft_both', labels_used=False, seconds=perf_counter() - started,
        new_model_forward=False, no_semantic_boundary_annotations=True,
        caveat='v1 states symlinked; new hard/soft states in boundary_states.npz',
        protocol='punctuation is a fallible cue; soft confirmation squares retention only at punctuation'))


if __name__ == '__main__':
    main()
