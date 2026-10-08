"""Frozen eight-query pilot of signed native influence, without detector fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.decision_risk_flow.run import load_model
from .mechanism import (GROUPS, OBJECTIVES, finite_query, measure_query,
                        norm_preserving_control, prefill_past, signed_responses)


CASE_IDS = ('12693', '14397', '12219', '12297')
MEMORY_CAP_BYTES = 22 * 1024 ** 3


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def freeze_protocol(inputs_path, output):
    document = json.loads(inputs_path.read_text())
    indexed = {case['id']: case for case in document['cases']}
    cases = [indexed[case_id] for case_id in CASE_IDS]
    protocol = dict(model=document['model'], input_path=str(inputs_path.resolve()),
        input_sha256=hashlib.sha256(inputs_path.read_bytes()).hexdigest(), cases=cases,
        groups=GROUPS, objectives=OBJECTIVES, local_window=16, random_direction_seeds=list(range(37, 69)),
        selected_layers_for_summary=[0, 7, 15, 23, 31], natural_labels_used_for_fit=False,
        positions='historical gold-guided matched mechanism roster; not blind detection',
        past='native no-grad KV, separately prefilled for every query; fixed during gradients',
        source_mask='existing coarse source text mask; no relation applicability annotations',
        detector_score=None, entropy_role='saved diagnostic only; never a risk score',
        observer=document['observer'], generator=document['generator'])
    write_json(output / 'PROTOCOL.json', protocol)
    return protocol


def random_direction_responses(measured, seeds):
    effects = []
    for seed in seeds:
        changed = norm_preserving_control(measured['messages'], measured['output_gram'], seed)
        effects.append(signed_responses(measured['gradient'], changed))
    return torch.stack(effects)


def response_summary(measured, random_effects):
    rows = []
    for layer in (0, 7, 15, 23, 31):
        for group, name in enumerate(GROUPS):
            actual = measured['response'][1, layer, group]
            randomized = random_effects[:, 1, layer, group].sum(-1)
            row = dict(layer=layer, group=name, margin_effect=float(actual.sum()),
                absolute_head_effect=float(actual.abs().sum()), positive_heads=int((actual > 0).sum()),
                projected_norm_sum=float(measured['projected_norm'][layer, group].sum()),
                attention_mass_sum=float(measured['attention_mass'][layer, group].sum()),
                random_net_mean=float(randomized.mean()), random_net_std=float(randomized.std()),
                residual_effect=float(measured['response_residual'][1, layer, group].sum()),
                ffn_effect=float(measured['response_ffn'][1, layer, group].sum()))
            rows.append(row)
    return rows


def finite_check(model, past, query_token, measured):
    response = measured['response'][1, :, 0]
    chosen = int(response.abs().flatten().argmax())
    layer, head = divmod(chosen, response.shape[-1])
    direction = measured['messages'][layer, 0, head]
    arguments = (model, past, query_token, measured['actual_id'], measured['rival_id'], layer, head, direction)
    identity = finite_query(*arguments, dose=0.)
    reference = torch.tensor([measured['actual_logp'], measured['margin']])
    checks = []
    for dose in (.01, .001):
        positive = finite_query(*arguments, dose=dose)
        negative = finite_query(*arguments, dose=-dose)
        slope = (positive - negative) / (2 * dose)
        predicted = measured['response'][:, layer, 0, head]
        relative = (slope - predicted).abs() / predicted.abs().clamp_min(1e-6)
        checks.append(dict(dose=dose, measured_slope=slope.tolist(), predicted=predicted.tolist(),
                           absolute_error=(slope - predicted).abs().tolist(), relative_error=relative.tolist()))
    return dict(layer=layer, head=head, group='source', selection='largest native absolute margin response',
                identity_max_error=float((identity - reference).abs().max()), central_differences=checks)


def run_case(model, case, role, protocol, output, check_finite):
    target = case[role]
    position = case['prompt_length'] + target - 1
    query_token = case['token_ids'][position]
    actual_id = case['token_ids'][position + 1]
    past = prefill_past(model, case['token_ids'][:position])
    measured = measure_query(model, past, query_token, actual_id, case['source_mask'],
                             case['prompt_length'], protocol['local_window'])
    error = float(measured['reconstruction_error'].max())
    assert error < 2e-5, f'AV partition mismatch: {error}'
    random_effects = random_direction_responses(measured, protocol['random_direction_seeds'])
    finite = finite_check(model, past, query_token, measured) if check_finite else None
    if finite is not None:
        assert finite['identity_max_error'] <= 1e-3
        for check in finite['central_differences']:
            passed = [(absolute <= .02 or relative <= .02)
                      for absolute, relative in zip(check['absolute_error'], check['relative_error'])]
            assert all(passed), f'Finite directional check failed: {check}'
    arrays = {name: value.numpy() for name, value in measured.items() if isinstance(value, torch.Tensor)}
    arrays['random_direction_response'] = random_effects.numpy()
    arrays.pop('output_gram')
    key = case['id'] + '_' + role
    np.savez_compressed(output / (key + '.npz'), **arrays)
    result = dict(key=key, case_id=case['id'], role=role, text=case[role + '_text'],
        target_index=target, query_position=position, actual_id=actual_id, rival_id=measured['rival_id'],
        actual_logp=measured['actual_logp'], margin=measured['margin'], entropy=measured['entropy'],
        reconstruction_max_error=error, summary=response_summary(measured, random_effects), finite_check=finite,
        source_net_positive_layers=int((measured['response'][1, :, 0].sum(-1) > 0).sum()),
        local_net_positive_layers=int((measured['response'][1, :, 1].sum(-1) > 0).sum()),
        remote_net_positive_layers=int((measured['response'][1, :, 2].sum(-1) > 0).sum()))
    write_json(output / (key + '.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = freeze_protocol(args.inputs, args.output)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(protocol['model'])
    started = time.time()
    results = []
    selected_cases = protocol['cases'][:1] if args.profile_only else protocol['cases']
    for case in selected_cases:
        roles = ('target',) if args.profile_only else ('target', 'control')
        for role in roles:
            result = run_case(model, case, role, protocol, args.output, check_finite=not results)
            results.append(result)
            peak = torch.cuda.max_memory_allocated()
            print(json.dumps(dict(key=result['key'], margin=result['margin'],
                                  source_positive_layers=result['source_net_positive_layers'],
                                  peak_gib=peak / 1024 ** 3)), flush=True)
            assert peak <= MEMORY_CAP_BYTES, f'Observed allocation exceeds frozen 22 GiB cap: {peak}'
    write_json(args.output / 'RESULTS.json', dict(status='DONE', queries=len(results),
        backward_calls=2 * len(results), source_fits=0, detector_scores=0,
        seconds=time.time() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(), results=results))


if __name__ == '__main__':
    main()
