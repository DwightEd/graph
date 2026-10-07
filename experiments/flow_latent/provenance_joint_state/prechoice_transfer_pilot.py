"""Frozen boolean-readout transfer to two old manual pre-error cases, CPU only.

    These candidate words are outside the program's True/False training support.
    Scores are exploratory ranks, not calibrated risks or natural token AUROC.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .matrix_audit import sha256, write_json
from .native_prefix_pilot import INPUTS
from .prechoice_case_audit import weight_slice
from .prechoice_readout import CandidateCompatibility
from .prechoice_states_pilot import OUTPUT as STATES
from .source_prechoice_pilot import OUTPUT as PROGRAM
from .source_prechoice_audit import independent_scores


OUTPUT = Path('outputs/source_transfer_boolean_to_oldcases_20261008')


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    inputs = json.loads(INPUTS.read_text())
    model_path = Path(inputs['model'])
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']
    saved = torch.load(PROGRAM / 'model.pt', weights_only=False, map_location='cpu')
    readout = CandidateCompatibility().eval()
    readout.load_state_dict(saved['state_dict'])
    dependencies = [PROGRAM / 'model.pt', PROGRAM / 'SCORE_FREEZE.json', Path(__file__), INPUTS]
    dependencies.extend(STATES / case['id'] / 'states.npy' for case in inputs['cases'])
    write_json(OUTPUT / 'PROTOCOL.json', dict(
        scope='adapted after source-control pilot; two manual-discovery candidate pairs only',
        readout='frozen source-program parameters and normalization; no natural fitting/selection',
        input='native pre-choice state excluding current word; static candidate embeddings only',
        unsupported='candidate words outside True/False training support; no risk probability or deployment claim',
        new_llm_forwards=0, new_fits=0,
        hashes={str(path): sha256(path) for path in dependencies}))
    results = []
    for case in inputs['cases']:
        target = case['prompt_length'] + case['target']
        ids = [case['token_ids'][target], case['rival_token']]
        candidates = torch.stack([weight_slice(model_path, index, 'lm_head.weight', identity) for identity in ids])
        candidates = torch.nn.functional.normalize(candidates, dim=-1)
        raw = np.load(STATES / case['id'] / 'states.npy', mmap_mode='r')
        states = torch.from_numpy(np.array(raw[-1:]))
        normalized = (states - saved['mean']) / saved['scale']
        with torch.no_grad():
            scores = readout(normalized, candidates).numpy()
        independent = independent_scores(states.numpy(), dict(saved, candidate_embeddings=candidates))
        error = float(np.max(np.abs(scores - independent)))
        assert error < 1e-4
        native = json.loads((STATES / case['id'] / 'RESULTS.json').read_text())['baseline']
        results.append(dict(id=case['id'], candidate_ids=ids,
            candidate_order=['manual_incompatible', 'manual_compatible'],
            raw_compatibility_scores=scores[0].tolist(),
            compatible_minus_incompatible=float(scores[0, 1] - scores[0, 0]),
            correctly_orders_manual_pair=bool(scores[0, 1] > scores[0, 0]),
            native_compatible_margin=native['compatible_margin'],
            native_greedy_token=native['greedy_token'],
            scope='exploratory unsupported-candidate transfer; no detection threshold', numeric_error=error))
    np.save(OUTPUT / 'frozen_scores.npy', np.array([row['raw_compatibility_scores'] for row in results]))
    write_json(OUTPUT / 'SCORE_FREEZE.json', dict(sha256=sha256(OUTPUT / 'frozen_scores.npy')))
    write_json(OUTPUT / 'RESULTS.json', dict(status='DONE', cases=results, natural_detector_fits=0,
        source_fit_reused=1, new_llm_forwards=0, new_fits=0,
        first_error_warning_proven=False, intervention_proven=False, typed_detection_proven=False))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
