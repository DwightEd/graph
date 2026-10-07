"""Independent CPU readback of saved native pre-choice states; no model forward."""
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from .matrix_audit import sha256, write_json
from .native_prefix_pilot import INPUTS
from .prechoice_states_pilot import OUTPUT


def weight_slice(model_path, index, name, token=None):
    """Read only the final norm or one vocabulary row from the existing weights."""
    with safe_open(model_path / index[name], framework='pt', device='cpu') as handle:
        if token is None:
            return handle.get_tensor(name).float()
        return handle.get_slice(name)[token:token + 1].float().squeeze(0)


def independently_decode_margin(model_path, index, epsilon, states, correct, wrong):
    # Raw sites are [prediction row, layer, residual/attention/MLP, coordinate].
    last = torch.from_numpy(np.array(states[-1, -1]))
    residual = (last[0] + last[1]) + last[2]
    normalized = residual * torch.rsqrt(residual.square().mean() + epsilon)
    normalized *= weight_slice(model_path, index, 'model.norm.weight')
    correct_row = weight_slice(model_path, index, 'lm_head.weight', correct)
    wrong_row = weight_slice(model_path, index, 'lm_head.weight', wrong)
    return float(normalized @ correct_row - normalized @ wrong_row)


def audit_case(case, result, model_path, index, epsilon):
    assert case['id'] == result['id']
    directory = OUTPUT / case['id']
    states = np.load(directory / 'states.npy', mmap_mode='r')
    expected_prefix = case['prompt_length'] + case['target']
    shape_ok = list(states.shape) == [case['target'] + 1, 32, 3, 4096]
    # The next layer's input independently checks reconstruction at 31 cuts.
    summed = (states[:, :-1, 0] + states[:, :-1, 1]) + states[:, :-1, 2]
    equation_error = float(np.max(np.abs(summed - states[:, 1:, 0])))
    qkv_shapes = {}
    for name, width in (('q', 4096), ('k', 1024), ('v', 1024)):
        values = np.load(directory / f'{name}_full_prefix_pre_rope.npy', mmap_mode='r')
        qkv_shapes[name] = list(values.shape)
        assert list(values.shape) == [32, expected_prefix, width]
    wrong = case['token_ids'][expected_prefix]
    margin = independently_decode_margin(model_path, index, epsilon, states, case['rival_token'], wrong)
    baseline = result['baseline']
    probability_margin = float(np.log(baseline['compatible_probability'] /
                                      baseline['incompatible_probability']))
    probability_errors = []
    delta_errors = []
    for row in result['interventions']:
        probability_errors.append(abs(np.log(row['compatible_probability'] /
                                             row['incompatible_probability']) - row['compatible_margin']))
        delta_errors.append(abs(row['compatible_margin'] - baseline['compatible_margin'] -
                               row['compatible_margin_change']))
    margin_error = abs(margin - baseline['compatible_margin'])
    assert shape_ok and result['prefix_length'] == expected_prefix
    assert equation_error == 0 and margin_error < 1e-4
    assert max(probability_errors) < 1e-5 and max(delta_errors) < 1e-12
    return dict(id=case['id'], states_shape=list(states.shape), qkv_shapes=qkv_shapes,
                prediction_prefix_length=expected_prefix, interlayer_equation_error=equation_error,
                independent_cpu_margin=margin, saved_margin=baseline['compatible_margin'],
                margin_error=margin_error, probability_margin_error=abs(probability_margin - baseline['compatible_margin']),
                interventions_checked=len(result['interventions']),
                intervention_probability_error=max(probability_errors), delta_error=max(delta_errors),
                greedy_flips=sum(row['greedy_token'] != baseline['greedy_token']
                                 for row in result['interventions']))


def main():
    torch.set_num_threads(4)
    inputs = json.loads(INPUTS.read_text())
    results = json.loads((OUTPUT / 'RESULTS.json').read_text())
    model_path = Path(inputs['model'])
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']
    epsilon = json.loads((model_path / 'config.json').read_text())['rms_norm_eps']
    cases = [audit_case(case, result, model_path, index, epsilon)
             for case, result in zip(inputs['cases'], results['cases'])]
    paths = list(OUTPUT.rglob('*.npy')) + [OUTPUT / 'RESULTS.json', OUTPUT / 'PROTOCOL.json',
        INPUTS, Path(__file__), model_path / 'config.json', model_path / 'model.safetensors.index.json']
    report = dict(status='PASS', scope='same-agent independent CPU arithmetic and saved-array readback',
                  timing='post-experiment verification, not a pre-run artifact manifest',
                  new_llm_forwards=0, new_fits=0, cases=cases,
                  artifact_hashes={str(path): sha256(path) for path in paths})
    write_json(OUTPUT / 'NUMERIC_AUDIT.json', report)
    print(json.dumps(dict(status='PASS', cases=cases), indent=2))


if __name__ == '__main__':
    main()
