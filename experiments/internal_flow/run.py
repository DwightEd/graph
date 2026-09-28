"""Small exposed-case native diagnostics; no source deletion and no model training."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .capture import capture, sequence_logp, state_comparison
from .inputs import prepare


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def groups_for(probe, length):
    return dict(probe['groups'], history=list(range(probe['answer_start'], length)),
                query_self=[length-1])


def collect_states(model, probe, directory):
    states, summaries = {}, {}
    variants = dict(before=probe['prefix_ids'], **probe['variants'])
    for side, ids in variants.items():
        result, run = capture(model, ids, groups_for(probe, len(ids)), probe['candidate_ids'])
        if result['reconstruction_max'] > .01:
            raise ValueError(f"Native message reconstruction failed: {result['reconstruction_max']}")
        states[side] = run.states
        summaries[side] = result
        pd.DataFrame(run.writes).to_csv(directory/f'{side}_heads.csv', index=False)
        pd.DataFrame(run.trajectory).to_csv(directory/f'{side}_lens.csv', index=False)
        np.savez_compressed(directory/f'{side}_states.npz', **run.states)
        del run
    contrasts = []
    for side in ('wrong', 'equivalent'):
        contrasts.extend(dict(comparison=f'correct_vs_{side}', **row)
                         for row in state_comparison(states['correct'], states[side]))
    pd.DataFrame(contrasts).to_csv(directory/'state_contrasts.csv', index=False)
    save_json(directory/'baselines.json', summaries)
    return summaries


def intervention_plan(directory, heads):
    frame = pd.read_csv(directory/'before_heads.csv')
    evidence = frame[(frame.source_group == 'evidence') & (frame['head'] >= 0)].copy()
    row = evidence.loc[evidence.local_linear_support.abs().idxmax()]
    selected = dict(layer=int(row.layer), head=int(row['head']))
    random_head = (selected['head'] + 1 + 42 % (heads-1)) % heads
    configs = [('sham', 'evidence', 0., 'reroute', selected['head']),
               ('evidence_025', 'evidence', .25, 'reroute', selected['head']),
               ('evidence_050', 'evidence', .5, 'reroute', selected['head']),
               ('other_source', 'other_source', .5, 'reroute', selected['head']),
               ('equal_norm_random', 'evidence', .5, 'random_direction', selected['head']),
               ('control_head', 'evidence', .5, 'reroute', random_head)]
    return [dict(name=name, layer=selected['layer'], head=head, target=target, dose=dose, kind=kind)
            for name, target, dose, kind, head in configs]


def run_interventions(model, probe, directory, baseline):
    plan = intervention_plan(directory, model.config.num_attention_heads)
    save_json(directory/'intervention_plan.json', plan)
    full = {side: sequence_logp(model, probe, side) for side in ('correct', 'wrong')}
    rows = []
    for patch in plan:
        result, _ = capture(model, probe['prefix_ids'], groups_for(probe, len(probe['prefix_ids'])),
                            probe['candidate_ids'], patch=patch, collect=False)
        changed = {side: sequence_logp(model, probe, side, patch) for side in ('correct', 'wrong')}
        rows.append(dict(**patch, **result,
                         sequence_margin=changed['correct']-changed['wrong'],
                         sequence_margin_change=(changed['correct']-changed['wrong'])-(full['correct']-full['wrong']),
                         correct_sequence_logp_change=changed['correct']-full['correct'],
                         wrong_sequence_logp_change=changed['wrong']-full['wrong'],
                         margin_change=result['margin']-baseline['margin'],
                         correct_logp_change=result['correct_first_logp']-baseline['correct_first_logp'],
                         wrong_logp_change=result['wrong_first_logp']-baseline['wrong_first_logp']))
    save_json(directory/'sequence_baseline.json', full)
    if max(abs(rows[0]['margin_change']), abs(rows[0]['sequence_margin_change'])) > 1e-5:
        raise ValueError('Same-world null patch changed candidate margin')
    pd.DataFrame(rows).to_csv(directory/'interventions.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    probes = prepare(tokenizer, args.dataset)
    if args.cases:
        probes = [probe for probe in probes if probe['case'] in args.cases]
    manifest = dict(model=str(args.model), dtype='bfloat16', attention='eager', probes=probes,
        labels_used_for_discovery=True, new_detector=False, source_deletion=False,
        exclusions='12045 unsupported procedures lack a matched mutually exclusive value; not measured by this candidate-margin pilot',
        caveats=['observer replay, not original-generator states',
                 'post-branch differences include forced-token identity and length',
                 'same-prefix pre-branch state is shared by correct/wrong alternatives',
                 'known local field corrections, not certified full-answer clean controls',
                 'full-string likelihood intervention is not free-generation repair'])
    save_json(args.output/'manifest.json', manifest)
    if args.prepare_only:
        print(json.dumps(dict(prepared=len(probes), max_tokens=max(len(v) for p in probes for v in p['variants'].values()))), flush=True)
        return
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True).to('cuda:0').eval()
    started = perf_counter()
    for probe in probes:
        directory = args.output/probe['case']
        directory.mkdir(exist_ok=True)
        if (directory/'completed.json').exists():
            continue
        baseline = collect_states(model, probe, directory)
        run_interventions(model, probe, directory, baseline['before'])
        record = dict(case=probe['case'], seconds=perf_counter()-started,
                      candidate_tokens=tokenizer.convert_ids_to_tokens(probe['candidate_ids']),
                      before_margin=baseline['before']['margin'])
        save_json(directory/'completed.json', record)
        print(json.dumps(record), flush=True)
    save_json(args.output/'completed.json', dict(cases=len(probes), seconds=perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), model_forward_calls_per_case=24,
        new_detection_auroc=None, full_string_likelihood_measured=True, free_generation_repair_measured=False))


if __name__ == '__main__':
    main()
