"""Audit restored original thresholds separately from new small-reference thresholds."""
import argparse
import json
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.verify import official_targets, independent_metrics
from experiments.message_js.score import valid_tokens
from experiments.head_state_readout.score import OLD
from .restore import original_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    methods = read_json(args.output/'scores_frozen.json')['methods']
    thresholds = read_json(args.output/'thresholds.json')
    records = [r for r in manifest['records'] if r['role']=='regression']
    root = Path(records[0]['root'])
    dataset = Path(read_json(root/'manifest.json')['dataset'])
    wanted = {r['id'] for r in records}
    official = {}
    for line in (dataset/'response.jsonl').open():
        row = json.loads(line)
        if str(row['id']) in wanted:
            official[str(row['id'])] = row
    expected = {(r['key'], r['method']): r for r in read_json(args.output/'evaluation.json')['cases']}
    checked = 0
    for row in records:
        target, valid, _ = official_targets(row, official)
        with np.load(args.output/row['key']/'scores.npz') as saved:
            for method in methods:
                result = independent_metrics(target[valid], saved[method][valid], thresholds[row['task']][method])
                for name in ('auroc', 'ap', 'false_alarms', 'detected_errors'):
                    value = result[name]
                    claim = expected[row['key'], method][name]
                    assert (value is None and claim is None) or np.isclose(value, claim, atol=1e-12, rtol=0)
                checked += 1
    for task in thresholds:
        for method in methods:
            if method=='original_full_reference_fixed':
                assert thresholds[task][method]==original_threshold(task)
                continue
            values, weights = [], []
            for row in manifest['records']:
                if row['role']=='dev' and row['task']==task:
                    valid = valid_tokens(row, args.output, OLD)
                    values.extend(np.load(args.output/row['key']/'scores.npz')[method][valid])
                    weights.extend(np.full(valid.sum(), 1/valid.sum()))
            values, weights = np.asarray(values), np.asarray(weights)
            order = np.argsort(values, kind='stable')
            cumulative = np.cumsum(weights[order])/weights.sum()
            expected_threshold = values[order[np.flatnonzero(cumulative>=.95)[0]]]
            assert expected_threshold==thresholds[task][method], (task, method)
    write_json(args.output/'score_verification.json', dict(status='passed', case_method_groups=checked,
        independent_small_reference_thresholds=3*(len(methods)-1), preserved_original_thresholds=3, same_agent=True))
    print('verified', checked, 'case-method groups;', 3*(len(methods)-1), 'new thresholds and 3 preserved original thresholds')


if __name__=='__main__':
    main()
