"""Independent official-span metrics for an already measured score iteration."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.verify import official_targets, independent_metrics
from .score import valid_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output / 'manifest.json')
    records = [row for row in manifest['records'] if row['role'] == 'regression']
    dataset = Path(read_json(Path(records[0]['root']) / 'manifest.json')['dataset'])
    wanted = {row['id'] for row in records}
    official = {}
    for line in (dataset / 'response.jsonl').open():
        row = json.loads(line)
        if str(row['id']) in wanted:
            official[str(row['id'])] = row
    methods = read_json(args.output / 'scores_frozen.json')['methods']
    thresholds = read_json(args.output / 'thresholds.json')
    claimed = {(row['key'], row['method']): row for row in read_json(args.output / 'evaluation.json')['cases']}
    count = 0
    for row in records:
        target, valid, _ = official_targets(row, official)
        with np.load(args.output / row['key'] / 'scores.npz') as saved:
            for method in methods:
                selected = valid & np.isfinite(saved[method])
                result = independent_metrics(target[selected], saved[method][selected], thresholds[row['task']][method])
                expected = claimed[row['key'], method]
                for name in ('auroc', 'ap', 'false_alarms', 'detected_errors'):
                    value = result[name]
                    assert (value is None and expected[name] is None) or np.isclose(value, expected[name], atol=1e-12, rtol=0), (row['key'], method, name)
                count += 1
    calibration_checks = 0
    for task in ('QA', 'Summary', 'Data2txt'):
        dev = [row for row in manifest['records'] if row['role'] == 'dev' and row['task'] == task]
        for method in methods:
            values, weights = [], []
            for row in dev:
                valid = valid_tokens(row, args.output, Path('outputs/transport_topology_cases_20260928'))
                with np.load(args.output / row['key'] / 'scores.npz') as saved:
                    selected = saved[method][valid]
                values.extend(selected)
                weights.extend(np.full(len(selected), 1 / len(selected)))
            values, weights = np.asarray(values), np.asarray(weights)
            available = np.isfinite(values)
            values, weights = values[available], weights[available]
            order = np.argsort(values, kind='stable')
            cumulative = np.cumsum(weights[order])
            cumulative /= cumulative[-1]
            expected = values[order[np.flatnonzero(cumulative >= .95)[0]]]
            assert expected == thresholds[task][method], (task, method)
            calibration_checks += 1
    operator = Path(manifest.get('operator', args.output))
    assert read_json(operator / 'verification.json')['status'] == 'passed'
    result = dict(status='passed', checked_case_method_pairs=count, same_agent_numeric_verification=True,
        independently_checked_thresholds=calibration_checks,
        native_measurement_verification=str(operator / 'verification.json'))
    write_json(args.output / 'score_verification.json', result)
    print(result)


if __name__ == '__main__':
    main()
