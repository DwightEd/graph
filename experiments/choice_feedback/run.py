"""Run exposed GSM mechanism cases; this is not a detection benchmark."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.run import load_model
from experiments.span_source_control.measure import MODEL
from .native import full_margin, stream


def prepare_cases(records, tokenizer):
    """Fix one head across roles. No head selection on the new measurements."""
    candidates = [tokenizer.encode(text, add_special_tokens=False) for text in ('24', '20')]
    assert all(len(ids) == 1 for ids in candidates)
    candidates = [ids[0] for ids in candidates]
    cases = []
    specifications = [('correct_result', 'gsm8k-49', 62, False),
                      ('wrong_base', 'gsm8k-49', 93, False),
                      ('role_repaired', 'gsm8k-49', 93, True),
                      ('normal_base', 'gsm8k-243', 217, False)]
    for name, key, target, repair in specifications:
        row = next(row for row in records if row['key'] == key)
        answer = row['response']['answer_ids']
        prefix = row['prompt'] + answer[:target]
        if repair:
            replacement = tokenizer.encode(' original', add_special_tokens=False)
            assert len(replacement) == 1 and row['response']['token_text'][83] == ' new'
            prefix[len(row['prompt']) + 83] = replacement[0]
        assert tokenizer.decode([prefix[11]]) == '20'
        cases.append(dict(name=name, key=key, target=target, prefix=prefix,
            continuation=answer[target:target + 32], candidates=candidates,
            other_token=candidates[1] if answer[target] == candidates[0] else candidates[0],
            probe=dict(layer=17, head=27, receiver=len(prefix) - 1),
            source_keys=[11], control_keys=[12], role_repaired=repair))
    return cases


def save_run(directory, name, result, tokenizer):
    np.savez_compressed(directory / f'{name}.npz', **result)
    return dict(name=name, tokens=len(result['tokens']), mass=result['mass'],
                first_margin=float(result['margin'][0]),
                text=tokenizer.decode(result['tokens'].tolist()))


def validate_full(model, case, dose, result):
    prefix = case['prefix']
    full = full_margin(model, prefix + case['continuation'][:-1], len(prefix) - 1,
                       case['candidates'], case['probe'], case['source_keys'], dose)
    error = float(np.max(np.abs(full - result['margin'])))
    assert error < .005, ('cached/full disagreement', case['name'], dose, error)
    return error


def run_case(model, tokenizer, case, output):
    directory = output / case['name']
    directory.mkdir()
    conditions = [('clamp_zero', 0., False, None, 'source_keys'),
                  ('clamp_minus', -.5, False, None, 'source_keys'),
                  ('clamp_plus', .5, False, None, 'source_keys'),
                  ('control_plus', .5, False, None, 'control_keys'),
                  ('force_zero', 0., False, case['other_token'], 'source_keys'),
                  ('force_plus', .5, False, case['other_token'], 'source_keys'),
                  ('greedy_zero', 0., True, None, 'source_keys'),
                  ('greedy_plus', .5, True, None, 'source_keys'),
                  ('greedy_force', 0., True, case['other_token'], 'source_keys')]
    audits = []
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    for name, dose, greedy, forced, keys in conditions:
        started = perf_counter()
        result = stream(model, case['prefix'], case['continuation'], case['candidates'],
                        case['probe'], case[keys], dose, greedy, forced, eos_ids)
        audit = save_run(directory, name, result, tokenizer)
        if name.startswith('clamp_'):
            audit['full_recompute_max_error'] = validate_full(model, case, dose, result)
        audit['seconds'] = perf_counter() - started
        audits.append(audit)
        write_json(directory / 'runs.json', audits)
        print(case['name'], name, round(audit['seconds'], 2), flush=True)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=Path('outputs/span_source_control_20260929_v1'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    cases = prepare_cases(read_json(args.input / 'manifest.json')['records'], tokenizer)
    write_json(args.output / 'plan.json', dict(cases=cases, dose=.5, seed=42,
        horizon=32, candidate_order=['24', '20'],
        scope='2 exposed answers, 4 prefixes; controlled mechanism diagnostic, not natural generalization',
        head_selection='L17H27 from prior wrong_base diagnostic, fixed across all four prefixes',
        manual_choices='positions, 24/20 contrast and new/original single-token repair',
        model=str(MODEL), precision='FP32 execution of frozen BF16 weights',
        source_dose='one prompt numeric-key attention logit, odds factor exp(+/-0.5)',
        control='neighboring and token, unmatched original attention mass'))
    model = load_model(MODEL)
    started = perf_counter()
    results = {}
    for case in cases:
        results[case['name']] = run_case(model, tokenizer, case, args.output)
    write_json(args.output / 'complete.json', dict(status='complete', runs=results,
        seconds=perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        default_detector_changed=False, answers=2, independent_problems=1, active_jobs=False))


if __name__ == '__main__':
    main()
