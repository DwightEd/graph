"""Evaluate already-frozen layer pilot on all exposed original and manual controls."""
import argparse
from pathlib import Path
import numpy as np
from .run import read_json, write_json
from experiments.probabilistic_detection.cases import build_case, render_html
from experiments.native_support.ragtruth_benchmark.data import annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.capture/'READOUT_FREEZE.json')
    roster = read_json(args.capture/'manifest.json')['records']
    selection = read_json(args.capture/'selection.json')
    cases, controls = [], []
    for record in roster:
        role = record['role']
        if role not in ('regression', 'manual_positive'):
            continue
        with np.load(args.capture/f"{record['key']}_scores.npz") as saved:
            positions = saved['positions']
            scores = {name: saved[name] for name in saved.files if name != 'positions'}
        thresholds = selection[record['task']]['thresholds']
        if role == 'manual_positive':
            controls.append(dict(key=record['key'], original_id=record['id'], task=record['task'],
                methods={name: dict(false_alarms=int((value > thresholds[name]).sum()), tokens=len(value))
                         for name, value in scores.items()}))
            continue
        root = Path(record['root'])
        truth = annotations(root, read_json(root/'manifest.json'), [record])[record['id']]
        response = read_json(root/record['directory']/'response.json')
        case = build_case(record, response, truth, positions, np.array(response['answer_ids'])[positions], scores, thresholds, 8)
        case.update(in_sample=False, new_fit_source_excluded=True,
            partition_note='已知回归；整来源排除本轮拟合与校准；层间读出小样本pilot。',
            selected=selection[record['task']]['selected'])
        cases.append(case)
    report = dict(cases=cases, manual_controls=controls)
    write_json(args.capture/'cases.json', report)
    (args.capture/'cases.html').write_text(render_html(report), encoding='utf-8')
    print('LAYER_CASES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
