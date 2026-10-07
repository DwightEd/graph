"""Add one capacity-matched sequential node control without refitting frozen graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
from .data import digest_files
from .freeze_cache import manifest
from .run import train, score
from experiments.token_backtrace.grounded_projection_data import write_json


def control(capture, base, output, seed):
    manifest(capture, verify=True)
    frozen = json.loads((base / 'frozen_scores.json').read_text())
    assert digest_files([Path(p) for p in frozen]) == frozen
    protocol = json.loads((base / 'protocol.json').read_text())
    expected_capture = Path(protocol['raw_capture'])
    assert capture.resolve() == expected_capture.resolve(), 'Control cache differs from frozen graph cache'
    assert (capture / 'inputs.json').read_bytes() == (base / 'inputs.json').read_bytes()
    assert digest_files([capture / 'content_manifest.json']) == digest_files([expected_capture / 'content_manifest.json'])
    model_bytes = (base / 'native_model.pkl').read_bytes()
    base_model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    assert protocol['seed'] == seed and protocol['version'] == 3
    assert (protocol['node_dimensions'], protocol['context_dimensions'], protocol['factor_rank']) == (128, 128, 8)
    output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((base / 'inputs.json').read_text())
    write_json(output / 'inputs.json', inputs)
    started = time.time()
    # Load exactly the model bytes hashed before fitting; no second file read.
    original = pickle.loads(model_bytes)
    fitted = train(capture, inputs['cases'], 'lag_pair', 3, seed, original['node_projection'], 128, 8)
    for key in ('coefficients', 'covariances', 'weights', 'loadings', 'noise'):
        assert np.shape(fitted['model'][key]) == np.shape(original['model'][key]), key
    assert fitted['context_projection']['axes'].shape == original['context_projection']['axes'].shape
    with (output / 'lag_pair_model.pkl').open('wb') as stream:
        pickle.dump(fitted, stream)
    snapshot = output / 'code_snapshot'
    snapshot.mkdir()
    files = [Path(__file__).with_name(n) for n in ('control.py', 'run.py', 'model.py', 'evaluate.py', 'freeze_cache.py')]
    for path in files:
        (snapshot / path.name).write_bytes(path.read_bytes())
    protocol.update(lag_pair_control=True, base_scores=str(base),
        control='two preceding native head nodes; no explicit graph attributes',
        control_code_hashes=digest_files(files),
        cache_manifest_hash=digest_files([capture / 'content_manifest.json']),
        base_native_model_sha256=base_model_sha256,
        base_frozen_manifest_sha256=digest_files([base / 'frozen_scores.json']),
        capacity_matched=True, finite_iteration_budget=True)
    write_json(output / 'protocol.json', protocol)
    scores = {}
    for case in inputs['cases']:
        row = dict(np.load(base / case['id'] / 'scores.npz'))
        risk, posterior, mean, covariance = score(capture, case, 'lag_pair', 3, fitted)
        assert np.isfinite(risk).all()
        row.update(lag_pair=risk, lag_pair_posterior=posterior,
                   lag_pair_factor_mean=mean, lag_pair_factor_covariance=covariance)
        scores[case['id']] = row
    thresholds = json.loads((base / 'thresholds.json').read_text())
    for task in thresholds:
        reference = [scores[c['id']]['lag_pair'] for c in inputs['cases']
                     if c['cohort'] == 'reference' and c['task'] == task]
        thresholds[task]['lag_pair'] = float(np.quantile(np.concatenate(reference), .95))
    write_json(output / 'thresholds.json', thresholds)
    paths = [output / n for n in ('inputs.json', 'thresholds.json', 'protocol.json')]
    for case in inputs['cases']:
        directory = output / case['id']
        directory.mkdir()
        path = directory / 'scores.npz'
        np.savez(path, **scores[case['id']])
        saved = np.load(path)
        old = np.load(base / case['id'] / 'scores.npz')
        assert all(np.array_equal(saved[k], old[k]) for k in old.files)
        paths.append(path)
    write_json(output / 'diagnostics.json', dict(lag_pair=dict(trace=fitted['model']['trace'],
        context_variance=fitted['context_projection']['variance_retained']),
        capacity_shapes={k:list(np.shape(fitted['model'][k])) for k in ('coefficients','covariances','loadings')},
        original_score_arrays_identical=True))
    write_json(output / 'frozen_scores.json', digest_files(paths))
    write_json(output / 'execution.json', dict(status='DONE', seconds=time.time()-started,
        cases=len(inputs['cases']), models_added=1, original_models_refitted=False, labels_read_for_scoring=False))
    print(f'CONTROL DONE seed={seed} seconds={time.time()-started:.1f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    control(args.capture, args.base, args.output, args.seed)
