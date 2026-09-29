"""Recompute causal and factorial checks from saved raw arrays; same-agent audit."""
import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .analyze import read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controlled', type=Path, required=True)
    parser.add_argument('--historical', type=Path, default=Path('outputs/span_source_control_20260929_v1'))
    args = parser.parse_args()
    cases = read_json(args.output / 'plan.json')['cases']
    errors = {}
    for case in cases:
        directory = args.output / case['name']
        runs = {name: read_run(directory, name) for name in (
            'clamp_zero', 'clamp_plus', 'clamp_minus', 'force_zero', 'force_plus')}
        if not case['role_repaired']:
            historical = read_run(args.historical / case['key'], 'baseline')
            start = case['target']
            errors[case['name']] = float(np.max(np.abs(
                historical['logp'][start:start + 32] - runs['clamp_zero']['logp'])))
        interaction = ((runs['force_plus']['margin'] - runs['clamp_plus']['margin'])
                       - (runs['force_zero']['margin'] - runs['clamp_zero']['margin']))
        np.testing.assert_allclose(read_run(directory, 'responses')['source_token_interaction'],
                                   interaction, atol=1e-12, rtol=0.)
        assert all(np.isfinite(run['kv']).all() for run in runs.values())
    assert max(errors.values()) < .005

    rows = read_json(args.controlled / 'measurements.json')
    conditions = {(row['family'], row['pair'], row['reverse'], row['swap'], row['query']) for row in rows}
    assert len(rows) == len(conditions) == 128
    for row in rows:
        correct = row['margin'] * (-1 if row['query'] else 1) > 0
        assert correct == row['correct']
        assert row['candidates'][0] != row['candidates'][1]
    result = dict(status='PASS', reviewer='same agent, separate raw-array recomputation',
        historical_native_logp_max_errors=errors, factorial_interaction_checks=len(cases),
        controlled_conditions=128, controlled_correct=sum(row['correct'] for row in rows),
        all_controlled_source_effects_negative=all(row['source_effect'] < 0 for row in rows),
        metadata_correction='v1 complete.json independent_cases=2 counts answers; independent problems=1',
        model_provenance='Llama3.1-8B observer; originals: GSM49 Llama3-70B, GSM243 Qwen2.5-Math-72B')
    write_json(args.output / 'verification.json', result)
    print(result)


if __name__ == '__main__':
    main()
