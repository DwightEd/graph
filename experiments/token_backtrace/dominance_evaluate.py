"""Evaluate conditional channel necessity on the frozen matched native cohort."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .grounded_projection_data import write_json
from .repetition_analysis import source_bootstrap


CHANNELS = ('source', 'history', 'remote', 'both', 'important_source', 'important_history')


def evaluate_case(case, directory):
    metadata = json.loads((directory / 'metadata.json').read_text())
    saved = np.load(directory / 'observations.npz')
    names = metadata['names']
    rows, measures = [], {}
    for layer in metadata['layers']:
        margins = saved[f'{layer}:margin']
        base = margins[0]
        actual = [case['token_ids'][case['prompt_length']+case[position]] for position in ('target', 'control')]
        eligible = (base > 0) & (saved[f'{layer}:top'][0] == actual)
        measures[layer] = dict(baseline=base, eligible=eligible)
        for channel in CHANNELS:
            index = names.index('drop_' + channel + '_1')
            drop = margins[index]
            measures[layer][channel] = dict(loss=base-drop, necessary=eligible & (drop < 0),
                quarter=eligible & (margins[names.index('drop_' + channel + '_0.25')] < 0),
                random=eligible & (margins[names.index('random_' + channel)] < 0),
                top_changed=saved[f'{layer}:top'][index] != saved[f'{layer}:top'][0])
        both = margins[names.index('drop_both_1')]
        measures[layer]['source_rescue'] = (both < 0) & (margins[names.index('drop_history_1')] > 0)
        measures[layer]['history_rescue'] = (both < 0) & (margins[names.index('drop_source_1')] > 0)
        for index, world in enumerate(names):
            for role, position in enumerate(('error', 'normal')):
                rows.append(dict(id=case['id'], source_id=case['source_id'], kind=case['kind'],
                    layer=layer, role=position, world=world, margin=float(margins[index, role]),
                    logp=float(saved[f'{layer}:logp'][index, role]),
                    top=int(saved[f'{layer}:top'][index, role]), baseline=float(base[role]),
                    loss=float(base[role]-margins[index, role])))
    return measures, rows


def summarize(cases, measurements):
    result = {}
    for role, label in enumerate(('error', 'normal')):
        eligible = np.array([[m['eligible'][role] for m in measurements[c['id']].values()] for c in cases])
        result[label] = dict(eligible_positions=int(eligible.any(axis=1).sum()),
                            eligible_layer_positions=int(eligible.sum()))
        for channel in CHANNELS:
            summary = {}
            for field in ('necessary', 'quarter', 'random', 'top_changed'):
                values = np.array([[m[channel][field][role] for m in measurements[c['id']].values()] for c in cases])
                summary[field] = dict(layer_positions=int((values & eligible).sum()),
                    positions_any_layer=int((values & eligible).any(axis=1).sum()))
            losses = [np.mean([m[channel]['loss'][role] for m in measurements[c['id']].values()]) for c in cases]
            summary['mean_layer_loss'] = source_bootstrap([c['source_id'] for c in cases], losses)
            result[label][channel] = summary
        for field in ('source_rescue', 'history_rescue'):
            values = np.array([[m[field][role] for m in measurements[c['id']].values()] for c in cases])
            result[label][field] = dict(layer_positions=int((values & eligible).sum()),
                positions_any_layer=int((values & eligible).any(axis=1).sum()))
    differences = []
    for case in cases:
        layers = measurements[case['id']].values()
        difference = np.mean([(m['history']['loss'][0]-m['source']['loss'][0])
            - (m['history']['loss'][1]-m['source']['loss'][1]) for m in layers])
        differences.append(difference)
    result['paired_history_minus_source_loss'] = source_bootstrap([c['source_id'] for c in cases], differences)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-directory', default='native')
    args = parser.parse_args()
    directory = args.output / args.native_directory
    cases = json.loads((directory / 'inputs.json').read_text())['cases']
    measurements, rows = {}, []
    for case in cases:
        measured, records = evaluate_case(case, directory / case['id'])
        measurements[case['id']] = measured
        rows.extend(records)
    result = {'all': summarize(cases, measurements)}
    for kind in sorted({case['kind'] for case in cases}):
        result[kind] = summarize([c for c in cases if c['kind'] == kind], measurements)
    write_json(directory / 'results.json', result)
    with (directory / 'interventions.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
