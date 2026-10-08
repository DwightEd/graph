"""Collect source controls, then fit and freeze matched candidate readouts."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer
from state_audit.model.adapter import ModelAdapter
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.decision_risk_flow.run import load_model
from ..provenance_joint_state.matrix_audit import sha256, write_json
from ..provenance_joint_state.source_control_coverage import CONTROLS
from .controls import make_controls
from .observe import collect_control, WINDOW, SITES
from .readout import OrderedCompatibility, WideCurrentCompatibility, fit_statistics, row_permutations, source_loss, transform_view
from .qa_controls import make_qa_controls
from ..provenance_joint_state.source_control_coverage import ROSTER, SOURCE_FILE

MODEL = Path('../../models/Meta-Llama-3.1-8B-Instruct').resolve()
VARIANTS = ('single', 'mean', 'ordered', 'shuffled', 'drop_source', 'linear', 'wide_single')


def allocate_arrays(output, count):
    shapes = dict(states=(count, WINDOW, 32, 4, 4096),
                  source_heads=(count, WINDOW, 32, 32, 128),
                  source_mass=(count, WINDOW, 32, 32), candidates=(count, 2, 4096))
    return {name: np.lib.format.open_memmap(output / f'{name}.npy', mode='w+',
                                          dtype=np.float32, shape=shape) for name, shape in shapes.items()}


def reused_rows(path):
    if path is None:
        return {}, {}, []
    controls = json.loads((path / 'controls.json').read_text())
    index = {row['id']: i for i, row in enumerate(controls)}
    arrays = {name: np.load(path / f'{name}.npy', mmap_mode='r') for name in
              ('states', 'source_heads', 'source_mass', 'candidates')}
    return index, arrays, json.loads((path / 'observations.json').read_text())


def collect(args):
    args.output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    controls = make_controls(tokenizer, json.loads(CONTROLS.read_text()), args.cohort)
    if args.domain == 'mixed':
        controls += make_qa_controls(tokenizer)
    write_json(args.output / 'controls.json', controls)
    old_index, old_arrays, old_observations = reused_rows(args.reuse)
    planned = sum(row['id'] not in old_index for row in controls)
    code = list(Path(__file__).parent.glob('*.py'))
    data_paths = [CONTROLS]
    if args.domain == 'mixed':
        data_paths += [ROSTER, SOURCE_FILE]
    qa_inventory = None
    if args.domain == 'mixed':
        roster = json.loads(ROSTER.read_text())
        requested = {str(row['source_id']) for cohort, limit in [('fit', 12), ('dev', 4)]
                     for row in roster[cohort][:limit]}
        used = {row['source_id'] for row in controls if row['family'].startswith('qa_')}
        qa_inventory = dict(requested_sources=sorted(requested), used_sources=sorted(used),
            excluded_before_model=sorted(requested - used),
            exclusion_rule='first160-character quotes must be unique across all3 passages and length>=8')
    protocol = dict(model=str(MODEL), window=WINDOW, sites=SITES, cohort=args.cohort, domain=args.domain,
        planned_new_full_forwards=planned, reused_from=str(args.reuse),
        qa_inventory=qa_inventory,
        source_supervision='program boolean fields / literal quotes; zero natural label/answer reads',
        context='unverified draft cue; source swaps and role queries crossed with both cues',
        interpretation='prefix-conditioned choice, not self-generated clean-first-error prevention',
        source_message='native A_source V before W_O stored with physical heads; W_O output fourth site',
        input='all prior tokens, no current/future candidate appended',
        hashes={str(p): sha256(p) for p in code + data_paths + [args.output / 'controls.json']})
    write_json(args.output / 'PROTOCOL.json', protocol)
    arrays = allocate_arrays(args.output, len(controls))
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = ModelAdapter(load_model(MODEL))
    observations = collect_rows(adapter, controls, arrays, old_index, old_arrays, old_observations)
    peak = torch.cuda.max_memory_allocated()
    del adapter
    torch.cuda.empty_cache()
    for values in arrays.values():
        values.flush()
    write_json(args.output / 'observations.json', observations)
    error = max(row['errors']['head_reconstruction'] for row in observations)
    equation = max(row['errors']['state_equation'] for row in observations)
    assert error < 2e-4 and equation == 0, 'Native observation reconstruction failed'
    report = dict(status='DONE', controls=len(controls), new_full_forwards=planned,
        reused_controls=len(controls) - planned, seconds=time.time() - started, peak_gpu_bytes=peak,
        max_head_reconstruction_error=error, max_state_equation_error=equation,
        natural_detector_fits=0, native_metrics=native_summary(controls, observations))
    write_json(args.output / 'COLLECTION.json', report)
    print(json.dumps(report, indent=2), flush=True)


def collect_rows(adapter, controls, arrays, old_index, old_arrays, old_observations):
    observations = []
    with torch.no_grad():
        for index, control in enumerate(controls):
            if control['id'] in old_index:
                old = old_index[control['id']]
                for name, values in arrays.items():
                    values[index] = old_arrays[name][old]
                observation = old_observations[old]
                assert observation['token_ids'] == control['token_ids']
            else:
                states, heads, mass, native, errors = collect_control(adapter, control)
                arrays['states'][index] = states
                arrays['source_heads'][index] = heads
                arrays['source_mass'][index] = mass
                embedding = adapter.native.lm_head.weight[control['candidate_ids']].float()
                arrays['candidates'][index] = torch.nn.functional.normalize(embedding, dim=-1).cpu().numpy()
                observation = dict(id=control['id'], token_ids=control['token_ids'], native=native, errors=errors)
            observations.append(observation)
            if index % 8 == 0 or index + 1 == len(controls):
                print(f'COLLECT {index + 1}/{len(controls)} {control["id"]}', flush=True)
    return observations


def native_summary(controls, observations):
    predictions = np.array([np.argmax(row['native']['pair_logits']) for row in observations])
    truth = np.array([row['correct'] for row in controls])
    result = {}
    for cohort in ('fit', 'dev'):
        for family in ['all'] + sorted({row['family'] for row in controls}):
            chosen = np.array([row['cohort'] == cohort and
                               (family == 'all' or row['family'] == family) for row in controls])
            result[f'{cohort}/{family}'] = dict(count=int(chosen.sum()),
                native_pair_accuracy=float((predictions[chosen] == truth[chosen]).mean()),
                native_wrong=int((predictions[chosen] != truth[chosen]).sum()))
    return result


def get_batch(states, candidates, indices, variant, permutations):
    # States are standardized once on GPU; this changes storage, not the formula.
    selected = torch.tensor(indices, device='cuda')
    values = states.index_select(0, selected)
    values = transform_view(values, variant, permutations[indices].to('cuda'))
    embeddings = candidates.index_select(0, selected)
    return values, embeddings


def fit_one(args, states, candidates, controls, mean, scale, variant, seed, permutations):
    directory = args.output / f'{variant}_seed{seed}'
    directory.mkdir()
    torch.manual_seed(seed)
    model = OrderedCompatibility().to('cuda')
    if variant == 'wide_single':
        model = WideCurrentCompatibility().to('cuda')
    if variant == 'linear':
        model.transition.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    fit_indices = np.array([i for i, row in enumerate(controls) if row['cohort'] == 'fit'])
    truth = torch.tensor([row['correct'] for row in controls], device='cuda')
    identities = sorted({row['source_id'] for row in controls})
    sources = torch.tensor([identities.index(row['source_id']) for row in controls], device='cuda')
    generator = np.random.default_rng(seed)
    history = []
    for step in range(args.steps):
        indices = generator.choice(fit_indices, min(16, len(fit_indices)), replace=False)
        values, embeddings = get_batch(states, candidates, indices, variant, permutations)
        optimizer.zero_grad()
        loss = source_loss(model(values, embeddings), truth[indices], sources[indices], model)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step + 1 == args.steps:
            history.append(dict(step=step, loss=float(loss.detach())))
    scores = predict_all(model, states, candidates, variant, permutations)
    np.save(directory / 'scores.npy', scores)
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                    mean=mean.cpu(), scale=scale.cpu(), variant=variant, seed=seed,
                    steps=args.steps, history=history), directory / 'model.pt')
    write_json(directory / 'FREEZE.json', dict(scores_sha256=sha256(directory / 'scores.npy'),
        model_sha256=sha256(directory / 'model.pt'), cache=str(args.cache)))
    metrics = evaluate_scores(scores, controls, args.cache)
    write_json(directory / 'RESULTS.json', dict(metrics=metrics, history=history,
        trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
        source_program_fits=1, natural_detector_fits=0))
    print(f'FIT {variant} seed={seed} ' + json.dumps(metrics['dev/all']), flush=True)
    del model, optimizer
    return dict(variant=variant, seed=seed, metrics=metrics)


@torch.no_grad()
def predict_all(model, states, candidates, variant, permutations):
    output = []
    for first in range(0, len(states), 16):
        indices = np.arange(first, min(first + 16, len(states)))
        values, embeddings = get_batch(states, candidates, indices, variant, permutations)
        output.append(model(values, embeddings).cpu().numpy())
    return np.concatenate(output)


def evaluate_scores(scores, controls, cache):
    observations = json.loads((cache / 'observations.json').read_text())
    native = np.array([np.argmax(row['native']['pair_logits']) for row in observations])
    greedy = np.array([row['native']['greedy_token'] for row in observations])
    covered = np.array([token in row['candidate_ids'] for token, row in zip(greedy, controls)])
    confidence = np.array([row['native']['greedy_probability'] for row in observations])
    truth = np.array([row['correct'] for row in controls])
    predicted = scores.argmax(axis=1)
    risk = scores[np.arange(len(scores)), 1 - native] - scores[np.arange(len(scores)), native]
    results = {}
    for cohort in ('fit', 'dev'):
        for family in ['all'] + sorted({row['family'] for row in controls}):
            chosen = np.array([row['cohort'] == cohort and
                               (family == 'all' or row['family'] == family) for row in controls])
            wrong = chosen & (native != truth)
            right = chosen & (native == truth)
            errors = (native != truth)[chosen]
            risk_defined = bool(errors.any() and (~errors).any())
            results[f'{cohort}/{family}'] = dict(count=int(chosen.sum()),
                pair_accuracy=float((predicted[chosen] == truth[chosen]).mean()),
                native_accuracy=float((native[chosen] == truth[chosen]).mean()),
                native_wrong=int(wrong.sum()), corrected_native_wrong=int((predicted[wrong] == truth[wrong]).sum()),
                damaged_native_right=int((predicted[right] != truth[right]).sum()),
                greedy_candidate_coverage=int((chosen & covered).sum()),
                confident_native_wrong=int((wrong & covered & (confidence >= .5)).sum()),
                program_token_risk_defined=risk_defined,
                program_token_risk_auroc=float(roc_auc_score(errors, risk[chosen])) if risk_defined else None,
                program_token_risk_ap=float(average_precision_score(errors, risk[chosen])) if risk_defined else None,
                mean_signed_margin=float((np.where(truth[chosen] == 0, 1, -1) *
                    (scores[chosen, 0] - scores[chosen, 1])).mean()))
    return results


def fit(args):
    args.output.mkdir(exist_ok=False)
    controls = json.loads((args.cache / 'controls.json').read_text())
    states = np.load(args.cache / 'states.npy', mmap_mode='r')
    candidates = np.load(args.cache / 'candidates.npy', mmap_mode='r')
    fit_indices = [i for i, row in enumerate(controls) if row['cohort'] == 'fit']
    mean, scale = fit_statistics(states, fit_indices)
    permutations = row_permutations(len(states), WINDOW)
    torch.save(dict(mean=mean, scale=scale, permutations=permutations), args.output / 'statistics.pt')
    write_json(args.output / 'PROTOCOL.json', dict(cache=str(args.cache), steps=args.steps,
        seeds=args.seeds, variants=args.variants, learning_rate=.01, ridge=1e-4, batch_size=16,
        standardization='fit-only pooled time/site-coordinate; shared over lag',
        model='full-coordinate level + adjacent diagonal bilinear; wide_single has independent current linear/square maps',
        capacity='single/mean collapse lag weights; wide_single has8449 vs ordered6017 parameters',
        hashes={str(p): sha256(p) for p in [Path(__file__), Path(__file__).with_name('readout.py'),
                                           args.cache / 'PROTOCOL.json', args.output / 'statistics.pt']}))
    mean, scale = mean.to('cuda'), scale.to('cuda')
    states = torch.tensor(np.asarray(states), device='cuda')
    states.sub_(mean).div_(scale)
    candidates = torch.tensor(np.asarray(candidates), device='cuda')
    started = time.time()
    results = []
    for seed in args.seeds:
        for variant in args.variants:
            results.append(fit_one(args, states, candidates, controls, mean, scale, variant, seed, permutations))
    write_json(args.output / 'RESULTS.json', dict(status='DONE', fits=len(results),
        new_llm_forwards=0, natural_detector_fits=0, seconds=time.time() - started, runs=results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('collect', 'fit'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--cohort', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--domain', choices=('boolean', 'mixed'), default='boolean')
    parser.add_argument('--reuse', type=Path)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'collect':
        collect(args)
    else:
        fit(args)


if __name__ == '__main__':
    main()
