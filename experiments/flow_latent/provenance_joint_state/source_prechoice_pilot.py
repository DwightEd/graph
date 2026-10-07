"""Source-only boolean binding pilot, not natural QA detection or prevention."""
from contextlib import contextmanager
from collections import Counter
import ast
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.decision_risk_flow.run import load_model
from .matrix_audit import sha256, write_json
from .prechoice_readout import CandidateCompatibility, source_balanced_loss, standardize_fit
from .source_control_coverage import CONTROLS


OUTPUT = Path('outputs/source_transfer_boolean_prechoice_20261008')
MODEL = Path('../../models/Meta-Llama-3.1-8B-Instruct').resolve()
STEPS = 400
LEARNING_RATE = .01
RIDGE = 1e-4


def source_value(data, path):
    for key in path:
        data = data[key]
    return data


def set_source_value(data, path, value):
    owner = data
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value


def make_controls(rows):
    """Same relation query; source field swap reverses True/False compatibility."""
    all_sources = sorted({row['source_id'] for row in rows})
    fit_sources = set(all_sources[:24])
    controls = []
    for row in rows:
        if row['family'] != 'boolean_attribute':
            continue
        literal = row['original_prompt'].split('Structured data:\n', 1)[1].rsplit('\nOverview:', 1)[0]
        data = ast.literal_eval(literal)
        left, right = row['left'], row['right']
        first, second = source_value(data, left), source_value(data, right)
        assert type(first) is bool and type(second) is bool and first != second
        assert first == row['value_left'] and second == row['value_right']
        swapped = copy.deepcopy(data)
        set_source_value(swapped, left, second)
        set_source_value(swapped, right, first)
        for world, structured in [('original', data), ('swapped', swapped)]:
            for role, path in [('left', left), ('right', right)]:
                prompt = ('Read only the structured data below. Return the boolean value at the exact JSON path. '
                          'Answer with True or False only.\nStructured data:\n' +
                          json.dumps(structured, ensure_ascii=False) + '\nJSON path: ' +
                          json.dumps(path) + '\nValue:')
                controls.append(dict(source_id=row['source_id'], cohort='fit' if row['source_id'] in fit_sources else 'dev',
                                     world=world, role=role, path=path, prompt=prompt,
                                     correct=int(not source_value(structured, path))))
    return controls


def tokenize_controls(tokenizer, controls):
    candidate_ids = None
    for control in controls:
        prefix = tokenizer.encode(control['prompt'], add_special_tokens=True)
        candidates = []
        for word in (' True', ' False'):
            complete = tokenizer.encode(control['prompt'] + word, add_special_tokens=True)
            assert complete[:-1] == prefix, 'Candidate must add exactly one token without changing prefix'
            candidates.append(complete[-1])
        if candidate_ids is None:
            candidate_ids = candidates
        assert candidate_ids == candidates
        control['token_ids'] = prefix
    return candidate_ids


@contextmanager
def capture_prediction(model):
    records = [[None] * 3 for _ in model.model.layers]
    handles = []
    for index, layer in enumerate(model.model.layers):
        def observe_input(module, args, layer_index=index):
            records[layer_index][0] = args[0][0, -1].detach().cpu()
        def observe_attention(module, args, output, layer_index=index):
            records[layer_index][1] = output[0, -1].detach().cpu()
        def observe_mlp(module, args, output, layer_index=index):
            records[layer_index][2] = output[0, -1].detach().cpu()
        handles.extend([layer.register_forward_pre_hook(observe_input),
                        layer.self_attn.o_proj.register_forward_hook(observe_attention),
                        layer.mlp.register_forward_hook(observe_mlp)])
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def collect_states(model, controls, candidates):
    states, margins = [], []
    for index, control in enumerate(controls):
        ids = torch.tensor([control['token_ids']], device=model.device)
        with capture_prediction(model) as records:
            hidden = model.model(ids, use_cache=False).last_hidden_state[0, -1]
        states.append(torch.stack([torch.stack(row) for row in records]))
        logits = model.lm_head(hidden)
        margins.append(float(logits[candidates[0]] - logits[candidates[1]]))
        if index % 8 == 0 or index + 1 == len(controls):
            print(f'CONTROL {index + 1}/{len(controls)}', flush=True)
    return torch.stack(states), np.asarray(margins)


def train_readout(states, candidates, controls):
    fit_mask = torch.tensor([row['cohort'] == 'fit' for row in controls])
    standardized, mean, scale = standardize_fit(states, fit_mask)
    correct = torch.tensor([row['correct'] for row in controls])
    source_names = sorted({row['source_id'] for row in controls})
    sources = torch.tensor([source_names.index(row['source_id']) for row in controls])
    readout = CandidateCompatibility()
    optimizer = torch.optim.Adam(readout.parameters(), lr=LEARNING_RATE)
    history = []
    for step in range(STEPS):
        optimizer.zero_grad()
        scores = readout(standardized[fit_mask], candidates)
        loss = source_balanced_loss(scores, correct[fit_mask], sources[fit_mask], readout, RIDGE)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == STEPS - 1:
            history.append(dict(step=step, fit_loss=float(loss.detach())))
    with torch.no_grad():
        scores = readout(standardized, candidates).numpy()
    torch.save(dict(state_dict=readout.state_dict(), mean=mean, scale=scale,
                    candidate_embeddings=candidates, source_names=source_names, fit_mask=fit_mask,
                    history=history), OUTPUT / 'model.pt')
    return scores, history


def evaluate_frozen(scores, margins, controls):
    results = {}
    for cohort in ('fit', 'dev'):
        chosen = np.array([row['cohort'] == cohort for row in controls])
        truth = np.array([row['correct'] for row in controls])[chosen]
        prediction = scores[chosen].argmax(axis=1)
        native = (margins[chosen] < 0).astype(int)
        identities = sorted({row['source_id'] for row in controls if row['cohort'] == cohort})
        complete_flips = []
        for identity in identities:
            selected = np.array([row['source_id'] == identity for row in controls])
            complete_flips.append(bool(np.all(scores[selected].argmax(1) == np.array(
                [row['correct'] for row in controls])[selected])))
        results[cohort] = dict(sources=len(identities), prefixes=int(chosen.sum()),
                              readout_pair_accuracy=float((prediction == truth).mean()),
                              native_pair_accuracy=float((native == truth).mean()),
                              candidate_only_balanced_accuracy=.5,
                              all_four_source_role_worlds_correct=sum(complete_flips))
    return results


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    rows = json.loads(CONTROLS.read_text())
    controls = make_controls(rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    candidate_ids = tokenize_controls(tokenizer, controls)
    write_json(OUTPUT / 'controls.json', controls)
    write_json(OUTPUT / 'PROTOCOL.json', dict(
        scope='source-only synthetic boolean path-binding pilot; no natural hallucination training/evaluation',
        candidates=dict(True_token=candidate_ids[0], False_token=candidate_ids[1]),
        input='prefix ends at Value:; candidate tokens never enter the model',
        states='all32 layers x residual/attention_write/MLP_write x4096 float32; no QKV disk cache',
        worlds='original/swapped boolean roles, both left/right queries; original source value multiset preserved',
        split='sorted original all32 program sources first24 fit/rest8 dev; boolean family31source subset',
        steps=STEPS, learning_rate=LEARNING_RATE, ridge=RIDGE, seed=42,
        loss='source-equally averaged BCE + pair logistic + mean-square ridge, no dev selection',
        readout='all96 sites, shared4096 diagonal coordinate map, signed96 weights+bias=4193 parameters',
        standardization='fit-only per-coordinate mean/std, scale floor .001; candidate output rows normalized L2',
        supervision='structure-derived targets, not strict unsupervised; zero natural response/label file reads',
        planned_forwards=len(controls), model=str(MODEL),
        hashes={str(path): sha256(path) for path in [CONTROLS, Path(__file__),
            Path('experiments/flow_latent/provenance_joint_state/prechoice_readout.py'), OUTPUT / 'controls.json']}))
    started = time.time()
    model = load_model(MODEL)
    with torch.no_grad():
        candidates = model.lm_head.weight[candidate_ids].detach().float().cpu()
        candidates = torch.nn.functional.normalize(candidates, dim=-1)
    states, margins = collect_states(model, controls, candidate_ids)
    peak = torch.cuda.max_memory_allocated()
    del model
    torch.cuda.empty_cache()
    np.save(OUTPUT / 'states.npy', states.numpy())
    np.save(OUTPUT / 'native_candidate_margins.npy', margins)
    scores, history = train_readout(states, candidates, controls)
    np.save(OUTPUT / 'frozen_scores.npy', scores)
    write_json(OUTPUT / 'SCORE_FREEZE.json', dict(scores_sha256=sha256(OUTPUT / 'frozen_scores.npy'),
        model_sha256=sha256(OUTPUT / 'model.pt'), protocol_sha256=sha256(OUTPUT / 'PROTOCOL.json')))
    results = evaluate_frozen(scores, margins, controls)
    write_json(OUTPUT / 'RESULTS.json', dict(status='DONE', scope='synthetic source-held-out dev, not natural detection',
        metrics=results, history=history, full_forwards=len(controls), source_control_fits=1,
        natural_detector_fits=0, new_natural_detection_scores=0, seconds=time.time() - started,
        peak_gpu_bytes=peak, training_source_balance=dict(Counter(row['cohort'] for row in controls))))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
