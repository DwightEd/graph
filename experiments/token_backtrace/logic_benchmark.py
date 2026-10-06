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
    parser.add_argument('--stage', choices=('pilot', 'score-test', 'evaluate-test',
                                           'typed-capture', 'typed-evaluate'), required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/automatic_evidence_20260930_v2/manifest.json'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=1536)
    parser.add_argument('--case-limit', type=int)
    parser.add_argument('--constructed-only', action='store_true')
    args = parser.parse_args()
    if args.stage == 'typed-capture':
        typed_capture(args)
    elif args.stage == 'typed-evaluate':
        typed_evaluate(args.output)
    elif args.stage == 'evaluate-test':
        evaluate_test(args.output, methods=('odds_full', 'logic_full', 'logic_only'), primary='logic_full')
    else:
        {'pilot': pilot, 'score-test': score_test}[args.stage](args.previous, args.output)


def typed_cases():
    """32 program-verifiable interface cases; these are development controls."""
    rows = []
    for index, entity in enumerate(('store', 'school', 'library')):
        source = f'The {entity} is not open.'
        for wrong in (False, True):
            answer = f'The {entity} is {"" if wrong else "not "}open.'
            rows.append(dict(family='polarity', source=source, answer=answer))
    for first, second in (('Alice', 'Bob'), ('Carol', 'Dave'), ('Eve', 'Frank')):
        source = f'{first} is taller than {second}.'
        for wrong in (False, True):
            answer = f'{second} is taller than {first}.' if wrong else source
            rows.append(dict(family='role', source=source, answer=answer))
    for number in (3, 4, 5, 6):
        source = f'The sailor was stranded on the island for more than {number} days.'
        for wrong in (False, True):
            answer = f'The sailor was stranded on the island for {"" if wrong else "more than "}{number} days.'
            rows.append(dict(family='interval', source=source, answer=answer))
    for entity in ('store', 'school', 'library'):
        source = f'If it rains, the {entity} is closed.'
        for removed in (False, True):
            answer = f'The {entity} is closed.' if removed else source
            rows.append(dict(family='condition', source=source, answer=answer))
    for entity in ('store', 'school', 'library'):
        source = f'The {entity} is not open.'
        for quoted in (False, True):
            answer = f'The claim "the {entity} is open" is rejected.' if quoted else source
            rows.append(dict(family='quote_reject', source=source, answer=answer))
    return [dict(id=f'constructed-{index:02d}', **row) for index, row in enumerate(rows)]


def parser_inputs(manifest_path, tokenizer, case_limit=None, natural=True):
    cases = typed_cases()
    if case_limit is not None:
        cases = cases[:case_limit]
    documents = [dict(id=row['id'], source=row['source'], answer=row['answer'],
                      offsets=tokenizer(row['answer'], add_special_tokens=False,
                                        return_offsets_mapping=True)['offset_mapping']) for row in cases]
    if natural:
        for row in read_json(manifest_path)['records']:
            documents.append(dict(id=row['key'], source=source_text(tokenizer, row['source']),
                                  answer=row['response']['text'], offsets=row['response']['offsets']))
    return documents


def generate_relation_batch(model, tokenizer, texts, max_new_tokens):
    """Only one text per request: source and answer never share a prompt."""
    import torch
    from .logic import RELATION_PARSER_PROMPT, grounded_relations

    prompts = [tokenizer.apply_chat_template([
        dict(role='system', content=RELATION_PARSER_PROMPT),
        dict(role='user', content='TEXT:\n' + text)], tokenize=False, add_generation_prompt=True)
        for text in texts]
    encoded = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**encoded, do_sample=False, temperature=None, top_p=None,
                                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    results = []
    for text, tokens in zip(texts, output[:, encoded.input_ids.shape[1]:]):
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        records, rejected = grounded_relations(text, raw)
        # Invalid/truncated alternatives cannot be dropped to manufacture consensus.
        complete = bool(torch.isin(tokens, torch.tensor(model.generation_config.eos_token_id,
                                                       device=tokens.device)).any())
        if not complete:
            rejected.append(dict(reason='generation_truncated'))
        results.append(dict(text=text, raw=raw, records=records, rejected=rejected,
                            usable=not rejected, generated_tokens=int((tokens != tokenizer.pad_token_id).sum())))
    return results


def typed_capture(args):
    import torch
    from time import perf_counter
    from transformers import AutoModelForCausalLM
    from .logic import RELATION_PARSER_PROMPT

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    documents = parser_inputs(args.manifest, tokenizer, args.case_limit, not args.constructed_only)
    texts = list(dict.fromkeys(text for row in documents for text in (row['source'], row['answer'])))
    write_json(args.output / 'inputs.json', documents)
    write_json(args.output / 'protocol.json', dict(parser_prompt=RELATION_PARSER_PROMPT,
        model=args.model, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        seed=17, greedy=True, source_answer_parsed_independently=True,
        labels_used_for_parsing=False, old_natural_answers_exposed=True,
        invalid_alternative_policy='entire document pair unresolved; never discard failed alternatives',
        edit_coverage='NOT_MEASURED; TG1 cannot fully pass without opposite/equivalent edits'))
    torch.manual_seed(17)
    torch.set_num_threads(4)
    started = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
    results = []
    for start in range(0, len(texts), args.batch_size):
        results.extend(generate_relation_batch(model, tokenizer, texts[start:start + args.batch_size],
                                                args.max_new_tokens))
        write_json(args.output / 'parser_outputs.json', results)
        print(dict(parsed=len(results), total=len(texts), seconds=round(perf_counter()-started, 1)), flush=True)
    write_json(args.output / 'outputs_frozen.json', dict(status='complete', documents=len(documents),
        unique_parser_requests=len(texts), generate_calls=(len(texts)+args.batch_size-1)//args.batch_size,
        seconds=perf_counter()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        natural_labels_read=False, construction_expected_status_read=False))


def typed_evaluate(output):
    from .logic import parsed_token_constraints

    read_json(output / 'outputs_frozen.json')
    parsed = {row['text']: row for row in read_json(output / 'parser_outputs.json')}
    truths = {row['id']: row for row in typed_cases()}
    for index, truth in enumerate(truths.values()):
        if index % 2 == 0:
            truth['expected'] = 'supported'
        elif truth['family'] in ('condition', 'quote_reject'):
            truth['expected'] = 'unknown'
        else:
            truth['expected'] = 'contradiction'
    cases, natural = [], []
    for row in read_json(output / 'inputs.json'):
        source, answer = parsed[row['source']], parsed[row['answer']]
        usable = source['usable'] and answer['usable']
        if usable:
            statuses, events = parsed_token_constraints(source['records'], answer['records'], row['offsets'])
        else:
            statuses, events = ['unknown'] * len(row['offsets']), []
        resolved = sum(status != 'unknown' for status in statuses)
        record = dict(id=row['id'], tokens=len(statuses), nonunknown_tokens=resolved,
                      usable=usable, statuses=statuses, events=events,
                      source_rejected=source['rejected'], answer_rejected=answer['rejected'])
        if row['id'] in truths:
            truth = truths[row['id']]
            event_status = [event['status'] for event in events]
            predicted = event_status[0] if event_status and len(set(event_status)) == 1 else 'unknown'
            # Empty/failed parser is not a successful abstention on a quote/condition.
            correct = usable and bool(source['records']) and bool(answer['records']) and predicted == truth['expected']
            record.update(family=truth['family'], expected=truth['expected'], predicted=predicted, correct=correct)
            cases.append(record)
        else:
            natural.append(record)
    families = {}
    for family in sorted({row['family'] for row in cases}):
        members = [row for row in cases if row['family'] == family]
        families[family] = dict(total=len(members), correct=sum(row['correct'] for row in members),
            unknown=sum(row['predicted'] == 'unknown' for row in members),
            expected_unknown=sum(row['expected'] == 'unknown' for row in members),
            usable=sum(row['usable'] for row in members))
    tokens = sum(row['tokens'] for row in natural)
    resolved = sum(row['nonunknown_tokens'] for row in natural)
    correctness = sum(row['correct'] for row in cases)
    coverage = resolved / tokens if tokens else None
    summary = dict(constructed_total=len(cases), constructed_correct=correctness, by_family=families,
        natural_answers=len(natural), natural_tokens=tokens, nonunknown_tokens=resolved, nonunknown_fraction=coverage,
        constructed_gate=correctness >= 29 and len(cases) == 32,
        natural_coverage_gate=coverage is not None and coverage >= .5,
        edit_coverage_gate='NOT_MEASURED', tg1_pass=False,
        interpretation='parser interface development test only; no graph score or new detector performance')
    write_json(output / 'token_relations.json', dict(constructed=cases, natural=natural))
    write_json(output / 'evaluation.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
