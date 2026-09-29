"""Second development iteration: restricted constraints, then full-test freeze."""

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.span_source_control.measure import MODEL
from .benchmark import TASKS, load_pack, evaluate_test
from .logic import token_constraints, apply_constraints


def source_text(tokenizer, source):
    tokens = np.asarray(source['prompt_with_source'])
    return tokenizer.decode(tokens[np.asarray(source['source_mask'], bool)].tolist())


def capture(tokenizer, source, response, native):
    status, claims = token_constraints(source_text(tokenizer, source), response['text'], response['offsets'])
    return apply_constraints(native, status), status, claims


def pilot(previous, output):
    output.mkdir(exist_ok=False, parents=True)
    rows = read_json(previous / 'pilot_frozen.json')['records']
    thresholds = read_json(previous / 'thresholds.json')
    for task in TASKS:
        thresholds[task]['logic_full'] = thresholds[task]['odds_full']
        thresholds[task]['logic_only'] = .5
    write_json(output / 'thresholds.json', thresholds)
    write_json(output / 'protocol.json', dict(primary='logic_full', previous=str(previous.resolve()),
        labels_used_for_fit_or_threshold=False, exposed_case_labels_informed_schema_choice=True,
        schemas=['bare copular polarity with exact role/tense', 'bounded duration with heuristic event alignment'],
        threshold='inherit frozen odds_full threshold; no recalibration or target FPR guarantee',
        unknown='preserve original native score', supported='0 only on recognized assertion core',
        contradiction='1 on recognized contradictory assertion scope',
        scope='hybrid partial symbolic check; not full automatic internal node/donor method',
        logic_only='diagnostic selective operating point; unknown gives no alarm, not proved correct; fixed threshold0.5',
        stopping_rule='no more rule/threshold changes after full-test predictions frozen'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for row in rows:
        with np.load(previous / 'pilot' / row['key'] / 'scores.npz') as arrays:
            scores = {name: arrays[name] for name in ('token_id', 'base', 'token_pair', 'odds_pair', 'odds_full')}
        scores['logic_full'], status, claims = capture(tokenizer, row['source'], row['response'], scores['odds_full'])
        scores['logic_only'] = (status == 1).astype(float)
        directory = output / 'pilot' / row['key']
        directory.mkdir(parents=True)
        np.savez_compressed(directory / 'scores.npz', **scores, logic_status=status)
        write_json(directory / 'constraints.json', claims)
    write_json(output / 'pilot_frozen.json', dict(records=rows, labels_accessed=False))


def score_test(previous, output):
    read_json(previous / 'test_frozen.json')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    coverage = {}
    for task in TASKS:
        pack, metadata = load_pack(task, 'test')
        root = Path(metadata['source_cache'])
        with np.load(previous / f'{task}_test.npz') as arrays:
            scores = {name: arrays[name] for name in ('token_id', 'target', 'answer_index', 'base', 'odds_full')}
        scores['logic_full'] = scores['odds_full'].copy()
        statuses = np.zeros(len(pack['token_id']), dtype=np.int8)
        source_cache, witnesses = {}, {}
        for row in metadata['records']:
            if row['source_id'] not in source_cache:
                source_cache[row['source_id']] = source_text(tokenizer, read_json(root / row['source_file']))
            response = read_json(root / row['directory'] / 'response.json')
            status, claims = token_constraints(source_cache[row['source_id']], response['text'], response['offsets'])
            region = slice(row['packed_start'], row['packed_stop'])
            statuses[region] = status[pack['target'][region]]
            witnesses[row['id']] = claims
        scores['logic_full'] = apply_constraints(scores['odds_full'], statuses)
        scores['logic_only'] = (statuses == 1).astype(float)
        np.savez_compressed(output / f'{task}_test.npz', **scores, logic_status=statuses)
        write_json(output / f'{task}_constraints.json', witnesses)
        coverage[task] = dict(answers=len(metadata['records']), valid_tokens=len(pack['token_id']),
            recognized_tokens=int(np.count_nonzero(statuses)),
            contradictory_tokens=int(np.sum(statuses == 1)), supported_tokens=int(np.sum(statuses == -1)))
        print(task, coverage[task], flush=True)
    assert sum(row['answers'] for row in coverage.values()) == 2700
    assert sum(row['valid_tokens'] for row in coverage.values()) == 424408
    write_json(output / 'test_frozen.json', dict(coverage=coverage, labels_accessed=False,
        new_llm_forwards=0, root_gradients_recomputed_on_full_test=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'score-test', 'evaluate-test'), required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'evaluate-test':
        evaluate_test(args.output, methods=('odds_full', 'logic_full', 'logic_only'), primary='logic_full')
    else:
        {'pilot': pilot, 'score-test': score_test}[args.stage](args.previous, args.output)


if __name__ == '__main__':
    main()
