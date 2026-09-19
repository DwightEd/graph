"""One pipeline: audit original samples -> compile claims -> intervene -> report."""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import DEFAULT_SAMPLES, inventory, compile_cases, write_json


def experiments(layers, scopes, heads, repeats, seed):
    from .native import Intervention

    result = []
    groups = [('evidence',), ('value_source',), ('history',),
              ('evidence', 'value_source'), ('evidence', 'history'), ('mlp',)]
    for layer in layers:
        for scope in scopes:
            for group in groups:
                item = Intervention(layer, group, scope, tuple(heads))
                result.append(('cut_' + '_'.join(group), item))
            for group in ('value_source', 'history'):
                for number in range(repeats):
                    item = Intervention(layer, (group,), scope, tuple(heads), 'random', seed + number)
                    result.append(('random_' + group + '_' + str(number), item))
    return result


def save_run(path, record, trajectory, writes, changes):
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, record=np.asarray(json.dumps(record)),
            trajectory=np.asarray(json.dumps(trajectory)), writes=np.asarray(json.dumps(writes)),
            changes=np.asarray(json.dumps(changes)))
    temporary.replace(path)


def verify_replay(model, probe, tolerance, message_tolerance):
    from .native import forward

    logits, run = forward(model, probe['prefix_ids'], probe)
    last = logits[-1]
    saved_ids = probe['top_ids'].tolist()
    difference = last[saved_ids].cpu().numpy() - probe['top_logits']
    normalizer_error = abs(float(last.logsumexp(-1)) - probe['log_normalizer'])
    maximum = float(abs(difference).max())
    reconstruction = max(run.reconstruction)
    assert maximum <= tolerance and normalizer_error <= tolerance, 'Original logits do not replay: inspect model/tokenizer/runtime'
    assert reconstruction <= message_tolerance, 'A V W_O reconstruction differs from native attention output'
    measured = dict(top_logit_max_error=maximum, log_normalizer_error=normalizer_error,
        attention_write_relative_error=reconstruction, native_top1=int(last.argmax()),
        sampled_next=probe['sampled_next'], sampled_was_argmax=bool(last.argmax() == probe['sampled_next']))
    return measured, run.baseline_heads


def run_side(args, model, case, side, probe, schedule, output):
    from .native import evaluate

    identity = dict(case_id=case['case_id'], source_id=case['source_id'], side=side,
                    seed=probe['seed'], trace=probe['trace'], response_step=probe['response_step'])
    replay, baseline_heads = verify_replay(model, probe, args.replay_atol, args.message_rtol)
    results = output / 'runs'
    stem = case['case_id'] + '_' + side
    baseline_path = results / (stem + '_full.npz')
    if not baseline_path.exists():
        score, trajectory, writes, changes = evaluate(model, probe)
        save_run(baseline_path, dict(identity, variant='full', layer=-1, scope='none', **score), trajectory, writes, changes)
    for name, intervention in tqdm(schedule, desc=stem, unit='intervention'):
        filename = f'{stem}_L{intervention.layer}_{intervention.scope}_{name}.npz'
        path = results / filename
        if not path.exists():
            score, trajectory, writes, changes = evaluate(model, probe, (intervention,))
            save_run(path, dict(identity, variant=name, layer=intervention.layer, scope=intervention.scope, **score),
                     trajectory, writes, changes)
        if args.restore_layer is not None and intervention.operation == 'cut' and intervention.layer < args.restore_layer:
            restore_path = results / filename.replace('.npz', '_restore.npz')
            if not restore_path.exists():
                restore = dict(layer=args.restore_layer, heads=tuple(args.restore_heads), states=baseline_heads[args.restore_layer])
                score, trajectory, writes, changes = evaluate(model, probe, (intervention,), restore)
                save_run(restore_path, dict(identity, variant=name + '_restore', layer=intervention.layer,
                    restore_layer=args.restore_layer, scope=intervention.scope, **score), trajectory, writes, changes)
    return dict(identity, **replay)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['inventory', 'run', 'report'], default='run')
    parser.add_argument('--samples', default=DEFAULT_SAMPLES)
    parser.add_argument('--cases', default=str(Path(__file__).with_name('cases.json')))
    parser.add_argument('--output')
    parser.add_argument('--study', choices=['flow', 'focused', 'coarse'], default='flow')
    parser.add_argument('--recent-window', type=int, default=10)
    parser.add_argument('--flow-top-k', type=int, default=3)
    parser.add_argument('--flow-max-heads', type=int, default=12)
    parser.add_argument('--pairs-per-group', type=int, default=2)
    parser.add_argument('--supervised-head-roles', help='Optional head_rules.csv; geometry alone never aligns heads')
    parser.add_argument('--model', help='Same original checkpoint at a relocated local path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--layers', nargs='+', type=int, default=[8, 16, 24, 31])
    parser.add_argument('--heads', nargs='*', type=int, default=[], help='All heads by default; use one layer for exact L:H sweeps')
    parser.add_argument('--scopes', nargs='+', choices=['query', 'prefix'], default=['query', 'prefix'])
    parser.add_argument('--random', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--restore-layer', type=int)
    parser.add_argument('--restore-heads', nargs='*', type=int, default=[])
    parser.add_argument('--replay-atol', type=float, default=.25, help='Explicit bf16 cached-vs-prefill logit tolerance; error is saved')
    parser.add_argument('--message-rtol', type=float, default=.02, help='Relative A V W_O numerical reconstruction tolerance')
    args = parser.parse_args(argv)
    if args.output is None:
        names = {
            'flow': 'outputs/evidence_target_flow_v2',
            'focused': 'outputs/same_question_path_conflict_focused_v2',
            'coarse': 'outputs/same_question_path_conflict_coarse_v2',
        }
        args.output = names[args.study]
    if args.study == 'flow':
        from .flow import run_flow
        run_flow(args)
        return
    if args.study == 'focused':
        from .focused import run_focused
        run_focused(args)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'report':
        from .report import report
        report(output)
        return
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    evidence = output / 'inventory'
    evidence.mkdir(exist_ok=True)
    settings, samples, prompts = inventory(args.samples, evidence, cases)
    if args.stage == 'inventory':
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .report import report

    configuration = dict(vars(args), actual_model=args.model or settings['model'], cases=cases, sampling_settings=settings)
    configuration.pop('stage')
    saved_config = output / 'config.json'
    if saved_config.exists():
        assert json.loads(saved_config.read_text()) == configuration, 'Use a separate output for changed experiment settings'
    write_json(saved_config, configuration)
    tokenizer = AutoTokenizer.from_pretrained(configuration['actual_model'], local_files_only=True, use_fast=True)
    compiled = compile_cases(args.samples, tokenizer, samples, prompts, cases, output)
    model = AutoModelForCausalLM.from_pretrained(configuration['actual_model'], local_files_only=True,
        torch_dtype=getattr(torch, args.dtype), attn_implementation='eager').to(args.device).eval()
    assert model.config.model_type == 'llama', 'This native operator targets the saved Llama sampling run'
    assert all(0 <= layer < len(model.model.layers) for layer in args.layers)
    assert all(0 <= head < model.config.num_attention_heads for head in args.heads + args.restore_heads)
    assert args.restore_layer is None or 0 <= args.restore_layer < len(model.model.layers)
    model.requires_grad_(False)
    schedule = experiments(args.layers, args.scopes, args.heads, args.random, args.seed)
    (output / 'runs').mkdir(exist_ok=True)
    replays = []
    for compiled_case in compiled:
        for side, probe in compiled_case['sides'].items():
            row = run_side(args, model, compiled_case['case'], side, probe, schedule, output)
            row['identical_pair_prefix'] = compiled_case['identical_prefix']
            replays.append(row)
            write_json(output / 'replay.json', replays)
    report(output)


if __name__ == '__main__':
    main()
