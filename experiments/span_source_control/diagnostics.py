"""Explicit post-hoc paired-position probes and numerical dose checks, not scoring."""
import argparse
from pathlib import Path

import numpy as np

from experiments.anchored_flow.edges import layer_arrays
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from .finite import measure_probe
from .measure import MODEL, group_treatments

# Selected after reading case text/gold: diagnostics, never detector input or validation.
TARGETS = {'00005': [35, 54], '00006': [36, 43], '00012': [128], '00013': [97],
           '15604': [100], 'gsm8k-49': [93, 101], 'gsm8k-243': [217]}


def prepare_probes(row, directory):
    with np.load(directory / 'heads.npz') as saved:
        measured = saved['measured']
        continuity = saved['continuation'].mean((0, 1))
    groups = read_json(directory / 'sources.json')
    probes = []
    for target in TARGETS.get(row['key'], []):
        layer, head = np.unravel_index(np.argmax(np.abs(measured[:, :, target, 5])), (32, 32))
        attention, effect = layer_arrays(row, int(layer))
        probes.append(dict(target=target, layer=int(layer), head=int(head), regime='posthoc_diagnostic',
            token=row['response']['token_text'][target], continuation=float(continuity[target]),
            treatments=group_treatments(attention[head, target], effect[head, target], groups)))
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = read_json(args.output / 'manifest.json')['records']
    probes = {row['key']: prepare_probes(row, args.output / row['key']) for row in records}
    write_json(args.output / 'diagnostic_plan.json', dict(probes=probes, labels_used_for_positions=True,
        source_selection='unchanged automatic lexical readout, no evidence labels', dose_check=.005))
    model = load_model(MODEL)
    count = 0
    for row in records:
        directory = args.output / row['key']
        with np.load(directory / 'baseline.npz') as saved:
            baseline = {name: saved[name] for name in saved.files}
        for index, probe in enumerate(probes[row['key']]):
            measure_probe(model, row, probe, baseline, directory, f'diagnostic_{index}')
            count += 1
            print('diagnostic', row['key'], probe['target'], flush=True)
        if row['key'] in ('00012', 'gsm8k-49'):
            probe = read_json(directory / 'probes.json')[1]
            measure_probe(model, row, probe, baseline, directory, 'small_dose', dose_size=.005)
            print('small dose', row['key'], probe['target'], flush=True)
    write_json(args.output / 'diagnostic_complete.json', dict(status='complete', posthoc_probes=count,
        dose_checks=2, forward_passes=(count + 2) * 9, labels_used_for_positions=True))


if __name__ == '__main__':
    main()
