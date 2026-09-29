"""Evaluate a frozen automatic source-state path against unchanged token labels."""
import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .evaluate import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output/'scores_frozen.json')
    thresholds = read_json(args.output/'thresholds.json')
    methods = ('path_localized_odds', 'independent_verified')
    with (args.input/'tokens.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    localization = {}
    for key in thresholds:
        subset = [item for item in rows if item['key'] == key]
        with np.load(args.output/key/'scores.npz') as saved, np.load(args.input/key/'effects.npz') as effect, np.load(args.input/key/'scores.npz') as previous:
            np.testing.assert_allclose(saved['independent_verified'], previous['verified_localized_odds'])
            indices = np.arange(len(subset))
            path_effect = effect['output_js'][indices, saved['source_path']]
            original_effect = effect['output_js'][indices, previous['confirmed']]
            localization[key] = dict(path_js_mean=float(path_effect.mean()), independent_js_mean=float(original_effect.mean()),
                output_spans=len(read_json(args.output/key/'output_spans.json')),
                selected_address_changed=int(np.sum(saved['source_path'] != previous['confirmed'])))
            for target, item in enumerate(subset):
                item['gold'] = int(item['gold'])
                item['valid'] = item['valid']=='True'
                item['path_source'] = int(saved['source_path'][target])
                for name in methods:
                    item[name] = float(saved[name][target])
                    item[name+'_alarm'] = bool(saved[name][target] > thresholds[key][name])
    with (args.output/'tokens.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(pooled=summarize(rows, methods), localization=localization,
        answers={key: summarize([row for row in rows if row['key']==key], methods) for key in thresholds},
        state_identity='source address, not truth category', readout_future_scope='offline Viterbi')
    write_json(args.output/'evaluation.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
