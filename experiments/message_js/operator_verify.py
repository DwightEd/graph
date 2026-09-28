"""Check native decomposition, full response axes, and independent case metrics."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.verify import official_targets, independent_metrics


def verify(output):
    manifest = read_json(output / 'manifest.json')
    records, base = manifest['records'], Path(manifest['base'])
    methods = read_json(output / 'scores_frozen.json')['methods']
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
    claimed = {(row['key'], row['method']): row for row in read_json(output / 'evaluation.json')['cases']}
    thresholds = read_json(output / 'thresholds.json')
    checks, tokens, max_replay, max_relative, min_eigenvalue = 0, 0, 0., 0., 0.
    diagnostics = []
    for row in records:
        with np.load(base / row['key'] / 'readouts.npz') as original:
            count, prompt = len(original['token_ids']), int(original['prompt_length'])
            token_ids = original['token_ids'].copy()
        with np.load(output / row['key'] / 'operator.npz') as saved:
            np.testing.assert_array_equal(saved['token_ids'], token_ids)
            assert saved['final_state'].shape == (count, 4096)
            assert saved['tangent'].shape == (count, 3, 4096)
            for name in ('final_state', 'tangent', 'margin', 'gram', 'prompt_cancel', 'history_cancel'):
                assert np.isfinite(saved[name]).all(), (row['key'], name)
            for name in ('prompt_cancel', 'history_cancel'):
                assert saved[name].min() >= -1e-6 and saved[name].max() <= 1 + 1e-6
            gram = saved['gram'].astype(np.float64)
            np.testing.assert_allclose(gram, gram.transpose(0, 2, 1), atol=1e-6, rtol=1e-5)
            eigenvalues = np.linalg.eigvalsh(gram)
            scale = np.maximum(np.abs(gram).max((1, 2)), 1.)
            assert np.all(eigenvalues[:, 0] >= -1e-5 * scale)
            min_eigenvalue = min(min_eigenvalue, float(eigenvalues.min()))
            max_replay = max(max_replay, float(saved['replay_errors'].max()))
        original = np.load(base / row['key'] / 'derivative.npy', mmap_mode='r')
        residual = np.load(output / row['key'] / 'residual.npy', mmap_mode='r')
        transformed = np.load(output / row['key'] / 'ffn.npy', mmap_mode='r')
        assert original.shape == residual.shape == transformed.shape == (32, 32, count, prompt + count - 1)
        squared_error, squared_original, absolute_error = 0., 0., 0.
        for layer in range(32):
            assert np.isfinite(residual[layer]).all() and np.isfinite(transformed[layer]).all()
            expected = original[layer].astype(np.float64)
            difference = residual[layer].astype(np.float64) + transformed[layer] - expected
            squared_error += np.square(difference).sum()
            squared_original += np.square(expected).sum()
            absolute_error = max(absolute_error, float(np.abs(difference).max()))
        relative = float(np.sqrt(squared_error / max(squared_original, 1e-30)))
        # Independent FP32 replay uses different query batches; not a bitwise identity.
        assert relative <= .001, (row['key'], relative, absolute_error)
        max_relative = max(max_relative, relative)
        diagnostics.append(dict(key=row['key'], decomposition_relative_error=relative,
                                decomposition_max_absolute_error=absolute_error))
        tokens += count
        if row['role'] != 'regression':
            continue
        target, valid, _ = official_targets(row, official)
        with np.load(output / row['key'] / 'scores.npz') as scores:
            for method in methods:
                available = valid & np.isfinite(scores[method])
                result = independent_metrics(target[available], scores[method][available], thresholds[row['task']][method])
                expected = claimed[row['key'], method]
                for name in ('auroc', 'ap', 'false_alarms', 'detected_errors'):
                    value = result[name]
                    assert (value is None and expected[name] is None) or np.isclose(value, expected[name], atol=1e-12, rtol=0), (row['key'], method, name)
                checks += 1
    assert max_replay <= .0005
    finite = read_json(output / records[0]['key'] / 'finite_validation.json')
    assert max(row['relative_error'] for row in finite['results']) <= .03
    result = dict(status='passed', answers=len(records), tokens=tokens,
        checked_case_method_pairs=checks, source_counts={key: len(value) for key, value in sources.items()},
        max_state_relative_error=max_replay, max_decomposition_relative_error=max_relative,
        minimum_gram_eigenvalue=min_eigenvalue, finite_validation=finite,
        full_response_dimension=4096, intervention_directions=3, full_gate_jacobian=False,
        same_agent_numeric_verification=True, diagnostics=diagnostics)
    write_json(output / 'verification.json', result)
    print({key: value for key, value in result.items() if key != 'diagnostics'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
