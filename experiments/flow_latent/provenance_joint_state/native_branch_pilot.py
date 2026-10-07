"""Post-first-pilot revision: compare fixed incompatible and compatible word branches.

Manually verified candidate words are mechanism-diagnosis inputs only. Future
original words are preserved, but branch comparisons stop at the replaced word.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter

from experiments.decision_risk_flow.run import load_model
from experiments.token_backtrace.repetition_teacher import SelfReadout
from .matrix_audit import sha256, write_json
from .native_prefix_pilot import (
    INPUTS, OUTPUT as ORIGINAL, condition_list, replay_prefix, teacher_marker,
)


OUTPUT = Path('outputs/source_transfer_native_branches_20261008')


def summarize_branches(case, baseline, observations, teachers):
    original = np.load(ORIGINAL / f"{case['id']}_full_observations.npz")
    target = baseline['target']
    target_row = int(np.where(baseline['positions'] == target)[0][0])
    original_marker = teacher_marker(original['native/diagonal'], teachers)
    alternate_marker = teacher_marker(baseline['diagonal'], teachers)
    native_gap = float(original_marker[target_row] - alternate_marker[target_row])
    result = dict(id=case['id'], incompatible_token=case['token_ids'][target],
                  compatible_token=case['rival_token'],
                  native_incompatible_minus_compatible_marker=native_gap, conditions={})
    for name, observed in observations.items():
        wrong = teacher_marker(original[f'{name}/diagonal'], teachers)
        correct = teacher_marker(observed['diagonal'], teachers)
        gap = float(wrong[target_row] - correct[target_row])
        result['conditions'][name] = dict(
            incompatible_marker_change=float(wrong[target_row] - original_marker[target_row]),
            compatible_marker_change=float(correct[target_row] - alternate_marker[target_row]),
            marker_gap_change=gap - native_gap)
    result['shared_prechoice_hidden_max_abs_error'] = float(np.max(np.abs(
        original['native/final_hidden'][:target_row] - baseline['final_hidden'][:target_row])))
    result['source_direction_max_abs_error'] = float(np.max(np.abs(
        original['source_direction'] - baseline['source_message'].cpu().numpy())))
    return result


def main():
    OUTPUT.mkdir(exist_ok=False)
    inputs = json.loads(INPUTS.read_text())
    dependencies = [INPUTS, Path(__file__), Path('experiments/flow_latent/provenance_joint_state/native_prefix_pilot.py'),
                    ORIGINAL / 'PROTOCOL.json', ORIGINAL / 'RESULTS.json']
    dependencies.extend(ORIGINAL / f"{case['id']}_full_observations.npz" for case in inputs['cases'])
    write_json(OUTPUT / 'PROTOCOL.json', dict(
        scope='adapted after initial two-case result; no confirmatory claim',
        source='same manually verified Passage3 and rival words from earlier labelled discovery',
        observations='full raw arrays; interpreting alternate branch only up to target token',
        scoring='frozen naturally supervised marker, not an unsupervised detector',
        current_choice='use original branch margin; alternate replay margin is 0 by construction and unused',
        expected_prechoice='same hidden states and source direction before replaced word',
        planned_new_full_forwards=20, previous_full_forwards=20,
        hashes={str(path): sha256(path) for path in dependencies}))
    teachers = [SelfReadout(seed) for seed in (42, 123)]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = ModelAdapter(load_model(inputs['model']))
    started = time.time()
    results = []
    forwards = 0
    for case in inputs['cases']:
        alternate = dict(case, token_ids=case['token_ids'].copy())
        alternate['token_ids'][case['prompt_length'] + case['target']] = case['rival_token']
        baseline = replay_prefix(adapter, alternate)
        forwards += 1
        observations = {}
        for name, cut, delta in condition_list(baseline['source_message']):
            observations[name] = replay_prefix(adapter, alternate, cut, delta)
            forwards += 1
            print(f"BRANCH {case['id']} {name} {forwards}/20", flush=True)
        arrays = {f'{name}/{key}': row[key] for name, row in [('native', baseline)] + list(observations.items())
                  for key in ('diagonal', 'final_hidden', 'positions')}
        arrays['source_direction'] = baseline['source_message'].cpu().numpy()
        np.savez_compressed(OUTPUT / f"{case['id']}_full_observations.npz", **arrays)
        result = summarize_branches(case, baseline, observations, teachers)
        write_json(OUTPUT / f"{case['id']}_RESULTS.json", result)
        results.append(result)
    write_json(OUTPUT / 'RESULTS.json', dict(cases=results, new_full_forwards=forwards,
        both_stages_full_forwards=40, seconds=time.time() - started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), status='DONE', detector_fit_operations=0))
    print('DONE', json.dumps(dict(new_forwards=forwards, seconds=time.time() - started)), flush=True)


if __name__ == '__main__':
    main()
