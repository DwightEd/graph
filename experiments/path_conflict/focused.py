"""Fixed-head follow-up: source-role cuts, dose controls and same-prefix restoration."""

from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import inventory, compile_cases, write_json
from .focused_inputs import candidate_panels, source_rows
from .focused_plan import focused_trials, SOURCE_ROLES


PROTOCOL = 'path_conflict_focused_v2'


def restoration(trial, baseline):
    if trial.restore_layer is None:
        return None
    bank = baseline.baseline_mlps if trial.restore_site == 'mlp' else baseline.baseline_heads
    return dict(layer=trial.restore_layer, site=trial.restore_site, heads=trial.restore_heads,
                states=bank[trial.restore_layer])


def save_capture(path, identity, trial, score, run):
    from .main import save_run
    record = dict(identity, variant=trial.name, kind=trial.kind, parents=list(trial.parents),
                  actions=[asdict(item) for item in trial.actions], restore_layer=trial.restore_layer,
                  restore_site=trial.restore_site, restore_heads=list(trial.restore_heads), **score)
    save_run(path, record, run.trajectory, run.writes, run.changes)


def run_panel(args, model, tokenizer, case, side, probe, schedule, output):
    from .scoring import evaluate_candidates
    identity = dict(case_id=case['case_id'], source_id=case['source_id'], side=side, panel=probe['panel'],
        seed=probe['seed'], trace=probe['trace'], native_response_step=probe['response_step'],
        query=len(probe['prefix_ids']) - 1, forced_common_tokens=probe['forced_common_tokens'],
        candidates=probe['candidate_texts'],
        first_token_texts=[tokenizer.decode([candidate[0]], clean_up_tokenization_spaces=False)
                           for candidate in probe['candidates']],
        prefix_tail=tokenizer.decode(probe['prefix_ids'][-40:], clean_up_tokenization_spaces=False))
    stem = case['case_id'] + '_' + side + '_' + probe['panel']
    baseline_score, baseline = evaluate_candidates(model, probe)
    baseline_path = output / 'runs' / (stem + '_full.npz')
    if not baseline_path.exists():
        save_capture(baseline_path, identity, schedule[0], baseline_score, baseline)
    else:
        with np.load(baseline_path, allow_pickle=False) as saved:
            previous = json.loads(str(saved['record']))
        assert abs(previous['next_margin'] - baseline_score['next_margin']) < 1e-6, 'Baseline changed on resume'
    np.savez_compressed(output / 'routes' / (stem + '.npz'), **baseline.routes,
                        prefix_ids=probe['prefix_ids'], correct_ids=probe['candidates'][0],
                        wrong_ids=probe['candidates'][1])
    for trial in tqdm(schedule[1:], desc=stem, unit='trial'):
        path = output / 'runs' / (stem + '_' + trial.name + '.npz')
        if path.exists():
            continue
        if trial.kind == 'random':
            parent = output / 'runs' / (stem + '_' + trial.parents[0] + '.npz')
            trial = match_cut_norms(trial, parent)
        restore = restoration(trial, baseline)
        score, run = evaluate_candidates(model, probe, trial.actions, restore,
                                         reference_logp=baseline.prefix_log_prob)
        save_capture(path, identity, trial, score, run)
    return identity, source_rows(tokenizer, probe, identity)


def match_cut_norms(trial, parent_path):
    """Use the corresponding actual cut's layer-wise dose, even after upstream changes."""
    with np.load(parent_path, allow_pickle=False) as saved:
        changes = json.loads(str(saved['changes']))
    norms = {row['layer']: row['query_change_norm'] for row in changes}
    actions = tuple(replace(item, reference_norm=norms[item.layer]) for item in trial.actions)
    return replace(trial, actions=actions)


def setup(args, output):
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    directory = output / 'inventory'
    directory.mkdir(exist_ok=True)
    settings, samples, prompts = inventory(args.samples, directory, cases)
    schedule = focused_trials(args.random)
    config = dict(protocol=PROTOCOL, samples=args.samples, actual_model=args.model or settings['model'],
                  dtype=args.dtype, recent_window=args.recent_window, cases=cases, source_roles=SOURCE_ROLES,
                  sampling_settings=settings, schedule=[asdict(trial) for trial in schedule],
                  replay_atol=args.replay_atol, message_rtol=args.message_rtol)
    path = output / 'focused_config.json'
    if path.exists():
        # JSON normalizes dataclass tuples to lists.
        assert json.loads(path.read_text()) == json.loads(json.dumps(config)), 'Use a new output for changed conditions'
    write_json(path, config)
    return cases, settings, samples, prompts, schedule, config


def run_focused(args):
    from .focused_report import report_focused
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'report':
        report_focused(output)
        return
    cases, settings, samples, prompts, schedule, config = setup(args, output)
    if args.stage == 'inventory':
        return
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config['actual_model'], local_files_only=True, use_fast=True)
    compiled = compile_cases(args.samples, tokenizer, samples, prompts, cases, output)
    model = AutoModelForCausalLM.from_pretrained(config['actual_model'], local_files_only=True,
        torch_dtype=getattr(torch, args.dtype), attn_implementation='eager').to(args.device).eval()
    model.requires_grad_(False)
    assert model.config.model_type == 'llama' and len(model.model.layers) == 32
    assert model.config.num_attention_heads == 32, 'Registered head coordinates require the original Llama-3.1-8B'
    write_json(output / 'runtime.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        activation_dtype=str(next(model.parameters()).dtype), lm_head_multiply='float32',
        first_log_softmax='float64', protocol=PROTOCOL))
    execute_cases(args, model, tokenizer, compiled, schedule, output)
    report_focused(output)


def execute_cases(args, model, tokenizer, compiled, schedule, output):
    from .main import verify_replay
    for name in ('runs', 'routes'):
        (output / name).mkdir(exist_ok=True)
    sources, panels, replays = [], [], []
    for entry in compiled:
        case = entry['case']
        for side, native in entry['sides'].items():
            measured, _ = verify_replay(model, native, args.replay_atol, args.message_rtol)
            replays.append(dict(case_id=case['case_id'], side=side, **measured))
            write_json(output / 'replay.json', replays)
            for probe in candidate_panels(tokenizer, native, case, args.recent_window):
                identity, rows = run_panel(args, model, tokenizer, case, side, probe, schedule, output)
                panels.append(identity)
                sources.extend(rows)
                pd.DataFrame(sources).to_csv(output / 'source_tokens.csv.gz', index=False)
                write_json(output / 'panels.json', panels)
