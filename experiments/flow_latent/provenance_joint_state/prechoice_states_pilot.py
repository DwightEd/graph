"""Pre-choice all-layer state capture and native MLP intervention discovery.

Only earlier tokens enter the model. Two manually verified candidate pairs are
used for causal diagnosis, not for training or claiming a predictive detector.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch
from state_audit.model.adapter import ModelAdapter

from experiments.decision_risk_flow.run import load_model
from .matrix_audit import sha256, write_json
from .native_prefix_pilot import INPUTS


OUTPUT = Path('outputs/source_transfer_prechoice_states_20261008')
DOSE = .1


def prefix_before_candidate(case):
    return case['token_ids'][:case['prompt_length'] + case['target']]


@contextmanager
def capture_states(model, start):
    records = [{} for _ in model.model.layers]
    handles = []

    def save_input(layer, name):
        def observe(module, args):
            records[layer][name] = args[0][0, start:].detach().cpu()
        return observe

    def save_output(layer, name, full=False):
        def observe(module, args, output):
            value = output[0] if full else output[0, start:]
            records[layer][name] = value.detach().cpu()
        return observe

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(save_input(index, 'residual')))
        handles.append(layer.self_attn.o_proj.register_forward_hook(save_output(index, 'attention_write')))
        handles.append(layer.mlp.register_forward_hook(save_output(index, 'mlp_write')))
        handles.append(layer.register_forward_hook(save_output(index, 'after')))
        for name in ('q', 'k', 'v'):
            module = {'q': layer.self_attn.q_proj, 'k': layer.self_attn.k_proj,
                      'v': layer.self_attn.v_proj}[name]
            handles.append(module.register_forward_hook(save_output(index, name, full=True)))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def change_mlp_write(model, layer, delta):
    def replace(module, args, output):
        changed = output.clone()
        changed[0, -1] += delta
        return changed
    handle = model.model.layers[layer].mlp.register_forward_hook(replace)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def candidate_readout(model, hidden, case):
    logits = model.lm_head(hidden)
    probabilities = logits.softmax(-1)
    wrong = case['token_ids'][case['prompt_length'] + case['target']]
    correct = case['rival_token']
    return dict(compatible_margin=float(logits[correct] - logits[wrong]),
                compatible_probability=float(probabilities[correct]),
                incompatible_probability=float(probabilities[wrong]),
                greedy_token=int(logits.argmax()), greedy_probability=float(probabilities.max()))


@torch.no_grad()
def lens_profile(model, records, case):
    profile = []
    for layer, row in enumerate(records):
        residual = row['residual'][-1].to(model.device)
        attention = row['attention_write'][-1].to(model.device)
        mlp = row['mlp_write'][-1].to(model.device)
        values = {}
        for name, state in [('before', residual), ('after_attention', residual + attention),
                            ('after_mlp', residual + attention + mlp)]:
            values[name] = candidate_readout(model, model.model.norm(state), case)['compatible_margin']
        profile.append(dict(layer=layer, **values))
    return profile


def save_states(directory, records):
    states = np.stack([np.stack([row[name].numpy() for name in
        ('residual', 'attention_write', 'mlp_write')], axis=1) for row in records], axis=1)
    np.save(directory / 'states.npy', states)
    for name in ('q', 'k', 'v'):
        np.save(directory / f'{name}_full_prefix_pre_rope.npy', np.stack([row[name].numpy() for row in records]))
    error = max(float((row['residual'] + row['attention_write'] + row['mlp_write'] - row['after']).abs().max())
                for row in records)
    return error


@torch.no_grad()
def diagnose_case(adapter, case):
    model = adapter.native
    directory = OUTPUT / case['id']
    directory.mkdir()
    ids = adapter.input_ids(prefix_before_candidate(case))
    start = case['prompt_length'] - 1
    with capture_states(model, start) as records:
        hidden = model.model(ids, use_cache=False).last_hidden_state[0, -1]
    baseline = candidate_readout(model, hidden, case)
    reconstruction_error = save_states(directory, records)
    profile = lens_profile(model, records, case)
    interventions = []
    for layer, row in enumerate(records):
        native_write = row['mlp_write'][-1].to(model.device)
        generator = torch.Generator(device=model.device).manual_seed(73 + layer)
        random = torch.randn(native_write.shape, device=model.device, generator=generator)
        random *= native_write.norm() / random.norm()
        for name, delta in [('attenuate', -DOSE * native_write), ('random', DOSE * random)]:
            with change_mlp_write(model, layer, delta):
                changed = model.model(ids, use_cache=False).last_hidden_state[0, -1]
            outcome = candidate_readout(model, changed, case)
            interventions.append(dict(layer=layer, condition=name,
                compatible_margin_change=outcome['compatible_margin'] - baseline['compatible_margin'],
                **outcome))
        print(f"PRECHOICE {case['id']} layer={layer} forwards={1 + 2 * (layer + 1)}/65", flush=True)
    result = dict(id=case['id'], target=case['target'], prefix_length=len(ids[0]),
                  current_wrong_word_not_in_input=True, baseline=baseline,
                  source_relative_candidates='manual earlier mechanism discovery',
                  states_axes=[case['target'] + 1, 32, 3, 4096], sites=['residual_before', 'attention_write', 'mlp_write'],
                  state_equation_max_abs_error=reconstruction_error,
                  lens_scope='raw uncalibrated final-norm/unembedding lens; not semantic decoding proof',
                  lens_profile=profile, interventions=interventions)
    write_json(directory / 'RESULTS.json', result)
    return result


def main():
    OUTPUT.mkdir(exist_ok=False)
    inputs = json.loads(INPUTS.read_text())
    paths = [INPUTS, Path(__file__), Path('experiments/decision_risk_flow/run.py'),
             Path('experiments/decision_risk_flow/precision.py')]
    write_json(OUTPUT / 'PROTOCOL.json', dict(
        scope='2 manual-error-operator discovery cases; no natural detector fits or typed classifier',
        observation='prediction position P+t-1; model input ends before target word y_t',
        stored='float32 full raw vectors; all past prediction rows and all32 layers, no window/SVD',
        source_target_alignment='not token t post-state; prompt-end predicts answer0 included',
        model=inputs['model'], original_generator_weights='Llama2 not present locally',
        primary_sites=['residual_before', 'attention_write', 'mlp_write'],
        auxiliary_QKV='complete prefix projection outputs before RoPE, not compressed attention statistics',
        interventions='all32 current-position MLP writes: subtract10percent and equalnorm random, seed73+layer',
        dose=DOSE, planned_full_forwards=130,
        evidence='native prefix-conditioned choice only; no decoding/generation prevention claim',
        hashes={str(path): sha256(path) for path in paths}))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    adapter = ModelAdapter(load_model(inputs['model']))
    results = [diagnose_case(adapter, case) for case in inputs['cases']]
    write_json(OUTPUT / 'RESULTS.json', dict(cases=results, full_forwards=130,
        seconds=time.time() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        status='DONE', detector_fits=0, source_relative_type_assignments=0))
    print('DONE prechoice states', flush=True)


if __name__ == '__main__':
    main()
