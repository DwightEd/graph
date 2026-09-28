"""Independently recompute official targets, ranks, and alarm counts."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.verify import official_targets, independent_metrics
from .score import METHODS


def verify(output):
    records = read_json(output / 'manifest.json')['records']
    evaluation = read_json(output / 'evaluation.json')
    sources = {role: {row['source_id'] for row in records if row['role'] == role}
               for role in ('fit', 'dev', 'regression', 'natural')}
    roles = list(sources)
    for index, role in enumerate(roles):
        for other in roles[index+1:]:
            assert not sources[role] & sources[other], (role, other)
    root = Path(next(row['root'] for row in records if row['kind'] == 'observer'))
    dataset = Path(read_json(root / 'manifest.json')['dataset'])
    wanted = {row['id'] for row in records if row['role'] == 'regression'}
    official = {}
    for line in (dataset / 'response.jsonl').open():
        row = json.loads(line)
        if str(row['id']) in wanted:
            official[str(row['id'])] = row
    claimed = {(row['key'], row['method']): row for row in evaluation['cases']}
    thresholds = read_json(output / 'thresholds.json')
    checked, errors, tokens, max_mass_error = 0, [], 0, 0.
    for row in records:
        with np.load(output / row['key'] / 'readouts.npz') as saved:
            errors.extend(saved['replay_errors'])
            tokens += len(saved['token_ids'])
            count = len(saved['token_ids'])
            prompt = int(saved['prompt_length'])
        for name in ('attention', 'derivative'):
            matrix = np.load(output / row['key'] / (name + '.npy'), mmap_mode='r')
            assert matrix.shape == (32, 32, count, prompt + count - 1)
            for layer in matrix:
                assert np.isfinite(layer).all()
                if name == 'attention':
                    mass_error = float(np.abs(layer.sum(-1, dtype=np.float64) - 1.).max())
                    max_mass_error = max(max_mass_error, mass_error)
                    # FP32 native softmax plus reduction rounding; float64 audit sum.
                    assert mass_error <= 32 * np.finfo(np.float32).eps, (row['key'], mass_error)
            del matrix
        if row['role'] != 'regression':
            continue
        target, valid, _ = official_targets(row, official)
        saved = np.load(output / row['key'] / 'scores.npz')
        for method in METHODS:
            available = valid & np.isfinite(saved[method])
            result = independent_metrics(target[available], saved[method][available], thresholds[row['task']][method])
            expected = claimed[row['key'], method]
            for name in ('auroc', 'ap', 'false_alarms', 'detected_errors'):
                value = result[name]
                assert (value is None and expected[name] is None) or np.isclose(value, expected[name], atol=1e-12, rtol=0), (row['key'], method, name)
            checked += 1
    maxima = np.max(errors, axis=0)
    assert maxima[0] <= .005 and maxima[1] <= .0005
    result = dict(status='passed', checked_case_method_pairs=checked, full_answers=len(records), tokens=tokens,
        source_counts={name: len(value) for name, value in sources.items()},
        max_logit_error=float(maxima[0]), max_state_relative_error=float(maxima[1]),
        max_attention_mass_error=max_mass_error, mass_tolerance=32 * float(np.finfo(np.float32).eps),
        original_official_spans=True, full_physical_axes=True, same_agent_numeric_verification=True)
    write_json(output / 'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    verify(parser.parse_args().output)
