"""Independent NumPy readback of source-program readout scores and all worlds."""
import json
from pathlib import Path

import numpy as np
import torch

from .matrix_audit import sha256, write_json
from .source_prechoice_pilot import OUTPUT


def independent_scores(states, saved):
    weights = saved['state_dict']
    coordinate = weights['coordinate'].numpy().astype(np.float64)
    sites = weights['site_weight'].numpy().astype(np.float64)
    candidates = saved['candidate_embeddings'].numpy().astype(np.float64)
    mean = saved['mean'].numpy().astype(np.float64)
    scale = saved['scale'].numpy().astype(np.float64)
    output = []
    for start in range(0, len(states), 8):
        values = (states[start:start + 8].astype(np.float64) - mean) / scale
        coefficients = sites[:, :, None] * coordinate
        vector = (values * coefficients).sum(axis=(1, 2))
        output.append(vector @ candidates.T / np.sqrt(states.shape[-1]) + float(weights['bias']))
    return np.concatenate(output)


def verify_role_worlds(controls, scores, native):
    result = {}
    source_sets = {}
    for cohort in ('fit', 'dev'):
        selected = [row['source_id'] for row in controls if row['cohort'] == cohort]
        source_sets[cohort] = set(selected)
        per_source = []
        for identity in sorted(source_sets[cohort]):
            positions = [i for i, row in enumerate(controls) if row['source_id'] == identity]
            rows = [controls[i] for i in positions]
            assert len(rows) == 4
            assert {(row['world'], row['role']) for row in rows} == {
                ('original', 'left'), ('original', 'right'), ('swapped', 'left'), ('swapped', 'right')}
            truth = np.array([row['correct'] for row in rows])
            assert truth.sum() == 2 and truth[0] != truth[2] and truth[1] != truth[3]
            assert rows[0]['path'] == rows[2]['path'] and rows[1]['path'] == rows[3]['path']
            predicted = scores[positions].argmax(1)
            native_predicted = (native[positions] < 0).astype(int)
            per_source.append((float((predicted == truth).mean()), float((native_predicted == truth).mean())))
        result[cohort] = dict(sources=len(per_source), source_mean_readout_pair_accuracy=float(np.mean(
            [row[0] for row in per_source])), source_mean_native_pair_accuracy=float(np.mean(
            [row[1] for row in per_source])), all_four_worlds_correct=sum(row[0] == 1 for row in per_source))
    assert not source_sets['fit'] & source_sets['dev']
    return result


def main():
    controls = json.loads((OUTPUT / 'controls.json').read_text())
    protocol = json.loads((OUTPUT / 'PROTOCOL.json').read_text())
    freeze = json.loads((OUTPUT / 'SCORE_FREEZE.json').read_text())
    states = np.load(OUTPUT / 'states.npy', mmap_mode='r')
    saved = torch.load(OUTPUT / 'model.pt', weights_only=False, map_location='cpu')
    scores = np.load(OUTPUT / 'frozen_scores.npy')
    native = np.load(OUTPUT / 'native_candidate_margins.npy')
    assert sha256(OUTPUT / 'frozen_scores.npy') == freeze['scores_sha256']
    assert sha256(OUTPUT / 'model.pt') == freeze['model_sha256']
    assert sha256(OUTPUT / 'PROTOCOL.json') == freeze['protocol_sha256']
    for path, identity in protocol['hashes'].items():
        assert sha256(Path(path)) == identity
    assert list(states.shape) == [len(controls), 32, 3, 4096]
    rebuilt = independent_scores(states, saved)
    error = float(np.max(np.abs(scores - rebuilt)))
    assert error < 1e-4
    worlds = verify_role_worlds(controls, rebuilt, native)
    recorded = json.loads((OUTPUT / 'RESULTS.json').read_text())['metrics']
    for cohort, values in worlds.items():
        assert values['source_mean_readout_pair_accuracy'] == recorded[cohort]['readout_pair_accuracy']
        assert values['source_mean_native_pair_accuracy'] == recorded[cohort]['native_pair_accuracy']
        assert values['all_four_worlds_correct'] == recorded[cohort]['all_four_source_role_worlds_correct']
    report = dict(status='PASS', scope='same-agent independent NumPy verification; not fresh scientific external audit',
                  new_llm_forwards=0, new_fits=0, score_max_abs_error=error, worlds=worlds,
                  raw_states_bytes=states.nbytes, parameter_count=sum(value.numel() for value in saved['state_dict'].values()),
                  timing='post-run artifact hash inventory',
                  audit_code_sha256=sha256(Path(__file__)),
                  hashes={str(path): sha256(path) for path in OUTPUT.iterdir()
                          if path.is_file() and path.name != 'NUMERIC_AUDIT.json'})
    write_json(OUTPUT / 'NUMERIC_AUDIT.json', report)
    print(json.dumps(dict(status='PASS', error=error, worlds=worlds, states_bytes=states.nbytes), indent=2))


if __name__ == '__main__':
    main()
