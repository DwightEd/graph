"""Two labelled-discovery cases: physical full-prefix message perturbations.

No detection training. All future native KV is recomputed; finite directional
responses are not exact JVPs, and the observer is not the original generator.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter

from experiments.decision_risk_flow.run import load_model
from experiments.token_backtrace.grounded_projection import capture_projection_inputs, observed_attention
from experiments.token_backtrace.repetition_teacher import SelfReadout, TEACHERS
from .matrix_audit import sha256, write_json


INPUTS = Path('outputs/repetition_passage3_20261007/native_all_layers/inputs.json')
OUTPUT = Path('outputs/source_transfer_native_prefix_20261008')
CUT = 17
SENDER_LAG = 2
DOSES = (.05, .1)


@contextmanager
def inject_message(model, layer, position, displacement):
    def replace(module, args):
        changed = args[0].clone()
        changed[0, position] += displacement.flatten()
        return (changed,) + args[1:]
    handle = model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(replace)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def replay_prefix(adapter, case, cut=None, displacement=None):
    prompt = case['prompt_length']
    target = prompt + case['target']
    sender = target - SENDER_LAG
    first = max(prompt, target - 8)
    ids = adapter.input_ids(case['token_ids'][:target + 3])

    with capture_projection_inputs(adapter.native, first + 1) as records:
        if cut is None:
            hidden = adapter.native.model(ids, use_cache=False).last_hidden_state[0]
        else:
            with inject_message(adapter.native, cut, sender, displacement):
                hidden = adapter.native.model(ids, use_cache=False).last_hidden_state[0]
    logits = adapter.native.lm_head(hidden[target - 1])
    actual = case['token_ids'][target]
    margin = float((logits[case['rival_token']] - logits[actual]).cpu())
    kept_hidden = hidden[first:].cpu().numpy()
    positions = torch.arange(len(ids[0]), device=ids.device)
    cosine, sine = adapter.native.model.rotary_emb(adapter.native.model.embed_tokens(ids), positions[None])
    selected = positions[first:]
    diagonal = []
    source_message = None
    for layer, record in enumerate(records):
        weights, values = observed_attention(adapter, layer, record, selected, cosine[0], sine[0])
        rows = torch.arange(len(selected), device=ids.device)
        diagonal.append(weights[rows, :, selected].cpu().numpy())
        if layer == CUT and cut is None:
            source = torch.zeros(len(ids[0]), dtype=torch.bool, device=ids.device)
            source[:prompt] = torch.tensor(case['source_mask'], device=ids.device)
            source_message = torch.einsum('hs,shd->hd', weights[sender - first, :, source], values[source])
    diagonal = np.stack(diagonal, axis=1)
    return dict(diagonal=diagonal, final_hidden=kept_hidden,
                compatible_margin=margin, source_message=source_message,
                positions=selected.cpu().numpy(), target=target, sender=sender)


def teacher_marker(diagonal, teachers):
    values = np.log1p(1000 * diagonal).reshape(len(diagonal), -1)
    scores = [((values - teacher.mean) / teacher.scale) @ teacher.tangent() for teacher in teachers]
    return np.mean(scores, axis=0)


def condition_list(message):
    generator = torch.Generator(device=message.device).manual_seed(73)
    random = torch.randn(message.shape, device=message.device, generator=generator)
    random *= message.norm(dim=-1, keepdim=True) / random.norm(dim=-1, keepdim=True)
    conditions = [('identity', CUT, message * 0)]
    for dose in DOSES:
        for sign in (-1, 1):
            conditions.append((f'source_{sign * dose:+.2f}', CUT, sign * dose * message))
    for sign in (-1, 1):
        conditions.append((f'random_{sign * .1:+.2f}', CUT, sign * .1 * random))
        conditions.append((f'last_{sign * .1:+.2f}', 31, sign * .1 * message))
    return conditions


def summarize_case(case, baseline, observations, teachers):
    markers = {name: teacher_marker(row['diagonal'], teachers) for name, row in observations.items()}
    native_marker = teacher_marker(baseline['diagonal'], teachers)
    target_row = int(np.where(baseline['positions'] == baseline['target'])[0][0])
    sender_row = int(np.where(baseline['positions'] == baseline['sender'])[0][0])
    result = dict(id=case['id'], target_text=case['target_text'], source_passage=3,
                  cut=CUT, sender_lag=SENDER_LAG, native_compatible_margin=baseline['compatible_margin'],
                  native_target_marker=float(native_marker[target_row]), conditions={})
    for name, observed in observations.items():
        result['conditions'][name] = dict(
            target_marker_change=float(markers[name][target_row] - native_marker[target_row]),
            compatible_margin_change=observed['compatible_margin'] - baseline['compatible_margin'],
            selfhead_max_abs_change=float(np.max(np.abs(observed['diagonal'] - baseline['diagonal']))),
            future_hidden_max_abs_change=float(np.max(np.abs(
                observed['final_hidden'][sender_row + 1:] - baseline['final_hidden'][sender_row + 1:]))))
    derivatives = {}
    for dose in DOSES:
        positive = observations[f'source_{dose:+.2f}']
        negative = observations[f'source_{-dose:+.2f}']
        derivatives[dose] = (positive['diagonal'] - negative['diagonal']) / (2 * dose)
    large, small = derivatives[.1], derivatives[.05]
    result['finite_selfhead_derivative_relative_disagreement'] = float(
        np.linalg.norm(large - small) / max(np.linalg.norm(small), 1e-30))
    result['future_selfhead_derivative_norm'] = float(np.linalg.norm(small[sender_row + 1:]))
    return result


def freeze_protocol(inputs):
    paths = [INPUTS, Path(__file__), Path('experiments/token_backtrace/grounded_projection.py'),
             Path('experiments/decision_risk_flow/run.py'), Path('experiments/decision_risk_flow/precision.py'),
             Path('experiments/token_backtrace/repetition_teacher.py'),
             Path('teaching/state_audit/src/state_audit/model/adapter.py'), Path('.aris/compute/env-spec.json')]
    paths.extend(TEACHERS / f'self_only_seed{seed}/model.pt' for seed in (42, 123))
    write_json(OUTPUT / 'PROTOCOL.json', dict(
        scope='two previously selected labelled mechanism cases; no unsupervised scores',
        observer=inputs['model'], generator='Llama2; not the measured observer',
        full_prefix=True, native_future_KV_recomputed=True, fixed_discrete_text=True,
        source_keys='manually verified Passage3 from prior discovery input', cut=CUT, sender_lag=SENDER_LAG,
        doses=list(DOSES), observations='all 1024 self-head axes and final residual4096 per local token',
        response='centered finite differences; no exact 8B JVP/VJP claim',
        supervision='manual candidates and frozen naturally supervised teacher only for diagnosis',
        cases=[{k: case[k] for k in ('id', 'source_id', 'target', 'target_text', 'rival_token')}
               for case in inputs['cases']], planned_full_forwards=20,
        hashes={str(path): sha256(path) for path in paths},
        model_weights='existing env/weight path reused; no new independent whole-weight hash audit'))


def main():
    OUTPUT.mkdir(exist_ok=False)
    inputs = json.loads(INPUTS.read_text())
    freeze_protocol(inputs)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teachers = [SelfReadout(seed) for seed in (42, 123)]
    started = time.time()
    adapter = ModelAdapter(load_model(inputs['model']))
    results = []
    forwards = 0
    for case in inputs['cases']:
        baseline = replay_prefix(adapter, case)
        forwards += 1
        observations = {}
        for name, cut, delta in condition_list(baseline['source_message']):
            observations[name] = replay_prefix(adapter, case, cut, delta)
            forwards += 1
            print(f"FORWARD {case['id']} {name} {forwards}/20", flush=True)
        arrays = {f'{name}/{key}': row[key] for name, row in [('native', baseline)] + list(observations.items())
                  for key in ('diagonal', 'final_hidden', 'positions')}
        arrays['source_direction'] = baseline['source_message'].cpu().numpy()
        np.savez_compressed(OUTPUT / f"{case['id']}_full_observations.npz", **arrays)
        result = summarize_case(case, baseline, observations, teachers)
        write_json(OUTPUT / f"{case['id']}_RESULTS.json", result)
        results.append(result)
    write_json(OUTPUT / 'RESULTS.json', dict(cases=results, full_forwards=forwards,
        seconds=time.time() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        status='DONE', detector_fit_operations=0))
    print('DONE', json.dumps(dict(forwards=forwards, seconds=time.time() - started)), flush=True)


if __name__ == '__main__':
    main()
