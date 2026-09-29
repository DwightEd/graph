"""Keep a signed source-address memory in every physical head at every token."""
import argparse
from pathlib import Path

import numpy as np

from experiments.anchored_flow.edges import layer_arrays
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.span_maintenance.state import FIELDS
from .measure import centered_effect


MEMORY_FIELDS = ('address_reuse', 'signed_reuse', 'continued_source_control')


def normalize(values):
    mass = values.sum(-1, keepdims=True)
    return np.divide(values, mass, out=np.zeros_like(values), where=mass > 0)


def update_memory(memory, current, continuation, age):
    """Axes [head, sign, prompt-key]; no cross-head or sign cancellation."""
    profile = normalize(current.reshape(len(current), -1)).reshape(current.shape)
    previous = normalize(memory.reshape(len(memory), -1)).reshape(memory.shape)
    signed = np.sqrt(profile * previous).sum((1, 2))
    address = np.sqrt(profile.sum(1) * previous.sum(1)).sum(-1)
    next_memory = (memory * (age - 1)[:, None, None] + profile) / age[:, None, None]
    observed = np.stack((address, signed, address * continuation), axis=-1)
    return next_memory, observed


def source_memory(effect, state):
    heads, count, keys = effect.shape
    memory = np.zeros((heads, 2, keys))
    observed = np.empty((heads, count, len(MEMORY_FIELDS)), dtype=np.float32)
    carriers = np.empty((heads, count), dtype=np.int32)
    for target in range(count):
        current = np.stack((np.maximum(effect[:, target], 0), np.maximum(-effect[:, target], 0)), axis=1)
        memory, observed[:, target] = update_memory(memory, current,
            state[:, target, FIELDS.index('continuation')], state[:, target, FIELDS.index('age')])
        carriers[:, target] = memory.sum(1).argmax(-1)
    return observed, carriers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for row in read_json(args.output / 'manifest.json')['records']:
        path = Path('outputs/span_maintenance_20260929_v1') / row['key'] / 'states.npz'
        with np.load(path) as saved:
            state = saved['state']
        measured, carriers = [], []
        for layer in range(32):
            attention, effect = layer_arrays(row, layer)
            centered = centered_effect(attention, effect)[..., :len(row['prompt'])]
            observed, keys = source_memory(centered, state[layer])
            measured.append(observed)
            carriers.append(keys)
        np.savez_compressed(args.output / row['key'] / 'memory.npz',
            measured=measured, carrier_keys=carriers, fields=MEMORY_FIELDS)
        print('source memory', row['key'], flush=True)
    write_json(args.output / 'memory_complete.json', dict(status='complete', fields=MEMORY_FIELDS,
        labels_used=False, note='joint reading/choice-source persistence; changing margins are not semantic truth'))


if __name__ == '__main__':
    main()
