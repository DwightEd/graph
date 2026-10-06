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
                                           'typed-capture', 'typed-evaluate',
                                           'grounding-capture', 'grounding-evaluate', 'witness-capture'), required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--manifest', type=Path, default=Path('outputs/automatic_evidence_20260930_v2/manifest.json'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=1536)
    parser.add_argument('--case-limit', type=int)
    parser.add_argument('--constructed-only', action='store_true')
    parser.add_argument('--keys', nargs='+')
    parser.add_argument('--target-limit', type=int)
    parser.add_argument('--grounding-version', choices=('direct', 'audit', 'pointer'), default='direct')
    parser.add_argument('--reuse', type=Path)
    parser.add_argument('--score-field', default='score', choices=('score', 'witness_score', 'strict_score', 'hybrid_score'))
    args = parser.parse_args()
    if args.stage == 'grounding-capture':
        grounding_capture(args)
    elif args.stage == 'grounding-evaluate':
        grounding_evaluate(args.output, args.score_field)
    elif args.stage == 'witness-capture':
        witness_capture(args.previous, args.output)
    elif args.stage == 'typed-capture':
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


def grounding_batch(model, tokenizer, requests, system_prompt, assistant_prefix=''):
    import torch
    from .logic import grounding_probabilities

    prompts = [tokenizer.apply_chat_template([
        dict(role='system', content=system_prompt), dict(role='user', content=request)],
        tokenize=False, add_generation_prompt=True) + assistant_prefix for request in requests]
    encoded = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False).to(model.device)
    positions = (encoded.attention_mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        hidden = model.model(**encoded, position_ids=positions, use_cache=False).last_hidden_state[:, -1]
        logits = model.lm_head(hidden).float()
        choice_ids = [tokenizer.encode(choice, add_special_tokens=False) for choice in ('A', 'B')]
        assert all(len(ids) == 1 for ids in choice_ids)
        score, choice_logp, mass = grounding_probabilities(logits, [ids[0] for ids in choice_ids])
        top_logp, top_ids = logits.log_softmax(-1).topk(min(10, logits.shape[-1]), dim=-1)
    return dict(score=score.cpu().numpy(), choice_logp=choice_logp.cpu().numpy(),
                choice_logits=logits[:, [ids[0] for ids in choice_ids]].cpu().numpy(),
                choice_mass=mass.cpu().numpy(), top_ids=top_ids.cpu().numpy(), top_logp=top_logp.cpu().numpy(),
                input_tokens=encoded.attention_mask.sum(-1).cpu().numpy())


def grounding_memo(model, tokenizer, source, answer, max_new_tokens):
    from time import perf_counter
    import torch
    from .logic import GROUNDING_AUDIT_PROMPT

    prompt = tokenizer.apply_chat_template([dict(role='system', content=GROUNDING_AUDIT_PROMPT),
        dict(role='user', content=f'SOURCE:\n{source}\n\nRESPONSE:\n{answer}')],
        tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
    calls = dict(forward=0)

    def count_forward(module, inputs):
        calls['forward'] += 1

    hook = model.register_forward_pre_hook(count_forward)
    started = perf_counter()
    try:
        with torch.no_grad():
            generated = model.generate(**encoded, do_sample=False, temperature=None, top_p=None,
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    finally:
        hook.remove()
    tokens = generated[0, encoded.input_ids.shape[1]:]
    complete = bool(torch.isin(tokens, torch.tensor(model.generation_config.eos_token_id,
                                                  device=tokens.device)).any())
    return dict(text=tokenizer.decode(tokens, skip_special_tokens=True), complete=complete,
                generated_tokens=len(tokens), max_new_tokens=max_new_tokens, forward_calls=calls['forward'],
                seconds=perf_counter()-started)


def grounding_answer(model, tokenizer, row, output, batch_size, target_limit, version, max_new_tokens):
    from time import perf_counter
    import torch
    from .logic import TOKEN_GROUNDING_PROMPT, TOKEN_GROUNDING_PROMPT_V2, grounding_request

    source = source_text(tokenizer, row['source'])
    response = row['response']
    targets = list(range(len(response['answer_ids'])))
    if target_limit is not None:
        targets = targets[:target_limit]
    directory = output / row['key']
    directory.mkdir()
    started = perf_counter()
    memo = dict(text='', complete=True, generated_tokens=0, forward_calls=0)
    if version == 'audit':
        memo = grounding_memo(model, tokenizer, source, response['text'], max_new_tokens)
        write_json(directory / 'memo.json', memo)
        assert memo['complete'], 'Truncated analyst memo saved; answer scoring not complete'
    requests = [grounding_request(source, response['text'], response['offsets'], target, version, memo['text'])
                for target in targets]
    system_prompt = TOKEN_GROUNDING_PROMPT if version == 'direct' else TOKEN_GROUNDING_PROMPT_V2
    assistant_prefix = '' if version == 'direct' else '{"decision":"'
    collected = {name: [] for name in ('score', 'choice_logits', 'choice_logp', 'choice_mass',
                                      'top_ids', 'top_logp', 'input_tokens')}
    for start in range(0, len(requests), batch_size):
        values = grounding_batch(model, tokenizer, requests[start:start + batch_size], system_prompt, assistant_prefix)
        for name in collected:
            collected[name].append(values[name])
        if start == 0:
            single = grounding_batch(model, tokenizer, requests[:1], system_prompt, assistant_prefix)
            batch_error = float(np.max(np.abs(values['choice_logp'][0] - single['choice_logp'][0])))
            # BF16 prefill may change logits with batching; preserve measured error.
            print(dict(key=row['key'], batch_single_choice_logp_max_error=batch_error), flush=True)
        if start % 25 == 0 or start+batch_size >= len(targets):
            print(dict(key=row['key'], measured=min(start+batch_size, len(targets)), total=len(targets)), flush=True)
    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    np.savez_compressed(directory / 'scores.npz', **values, target=targets,
                        token_ids=np.asarray(response['answer_ids'])[targets])
    seconds = perf_counter()-started
    write_json(directory / 'complete.json', dict(tokens=len(targets),
        forward_calls=(len(targets)+batch_size-1)//batch_size+1+memo['forward_calls'],
        new_forward_calls=(len(targets)+batch_size-1)//batch_size+1+memo['forward_calls'],
        classification_forward_calls=(len(targets)+batch_size-1)//batch_size+1,
        seconds=seconds, new_seconds=seconds, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        batch_single_choice_logp_max_error=batch_error, memo_generate_calls=int(version=='audit'),
        memo_generated_tokens=memo['generated_tokens'], labels_read=False))


def grounding_reuse_records(previous, records, current, model, max_new_tokens):
    old_protocol = read_json(previous / 'protocol.json')
    for field in ('prompt', 'memo_prompt', 'batch_size', 'assistant_prefix', 'grounding_version', 'threshold'):
        assert old_protocol[field] == current[field], field
    assert max_new_tokens >= old_protocol['max_new_tokens']
    old_manifest = read_json(previous / 'manifest.json')
    assert old_manifest['model'] == model
    assert old_manifest['records'] == records
    return {row['key']: row for row in old_manifest['records']}


def reuse_grounding_answer(previous, row, output, expect_memo=False):
    import shutil

    directory = previous / row['key']
    if not (directory / 'complete.json').exists():
        return False
    complete = read_json(directory / 'complete.json')
    assert complete['tokens'] == len(row['response']['answer_ids'])
    with np.load(directory / 'scores.npz') as values:
        np.testing.assert_array_equal(values['target'], np.arange(complete['tokens']))
        np.testing.assert_array_equal(values['token_ids'], row['response']['answer_ids'])
    if expect_memo:
        assert (directory / 'memo.json').exists()
        assert read_json(directory / 'memo.json')['complete']
    shutil.copytree(directory, output / row['key'])
    complete.update(reused_from=str(directory.resolve()), new_forward_calls=0, new_seconds=0.)
    write_json(output / row['key'] / 'complete.json', complete)
    print(dict(key=row['key'], reused_tokens=complete['tokens']), flush=True)
    return True


def grounding_capture(args):
    from time import perf_counter
    import torch
    from transformers import AutoModelForCausalLM
    from .logic import TOKEN_GROUNDING_PROMPT, TOKEN_GROUNDING_PROMPT_V2, GROUNDING_AUDIT_PROMPT

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = read_json(args.manifest)
    records = manifest['records']
    if args.keys is not None:
        records = [row for row in records if row['key'] in args.keys]
        assert set(args.keys) == {row['key'] for row in records}
    write_json(args.output / 'manifest.json', dict(records=records, model=args.model))
    system_prompt = TOKEN_GROUNDING_PROMPT if args.grounding_version == 'direct' else TOKEN_GROUNDING_PROMPT_V2
    write_json(args.output / 'protocol.json', dict(prompt=system_prompt, batch_size=args.batch_size,
        grounding_version=args.grounding_version, memo_prompt=GROUNDING_AUDIT_PROMPT if args.grounding_version=='audit' else None,
        max_new_tokens=args.max_new_tokens, assistant_prefix='' if args.grounding_version=='direct' else '{"decision":"',
        target_limit=args.target_limit, primary='grounding_score', threshold=.5,
        measurement='independent original-token external grounding; A/B conditional probability',
        labels_used_for_scores=False, exposed_development=True, model_is_observer=True,
        original_generator_causal_claim=False, token_risk_broadcast=False,
        outside_knowledge=False, full_answer_visible=True, new_graph_score=False,
        low_choice_mass_cutoff=.1, low_mass_rule='descriptive only; never remove targets or tune threshold',
        top10_added_after_M0=True))
    previous_records = {}
    if args.reuse is not None:
        assert args.target_limit is None, 'Only complete original answers can be reused'
        current = read_json(args.output / 'protocol.json')
        previous_records = grounding_reuse_records(args.reuse, records, current, args.model, args.max_new_tokens)
        current['reuse'] = str(args.reuse.resolve())
        current['reuse_policy'] = 'exact original records, protocol and complete EOS; larger cap affects only unfinished memos'
        write_json(args.output / 'protocol.json', current)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
    for row in records:
        if row['key'] in previous_records:
            if reuse_grounding_answer(args.reuse, row, args.output, args.grounding_version == 'audit'):
                continue
        torch.cuda.reset_peak_memory_stats()
        grounding_answer(model, tokenizer, row, args.output, args.batch_size, args.target_limit,
                         args.grounding_version, args.max_new_tokens)
    write_json(args.output / 'scores_frozen.json', dict(status='complete', answers=len(records),
        complete_answers=args.target_limit is None, labels_read=False, seconds=perf_counter()-started))


def grounding_evaluate(output, score_field='score'):
    import csv
    from experiments.decision_risk_flow.data import labels
    from experiments.automatic_evidence.evaluate import metrics, annotated_span_metrics

    freeze = read_json(output / 'scores_frozen.json')
    records = read_json(output / 'manifest.json')['records']
    truth = labels([row['original'] for row in records])
    tokens = []
    for row in records:
        with np.load(output / row['key'] / 'scores.npz') as arrays:
            if freeze['complete_answers']:
                np.testing.assert_array_equal(arrays['target'], np.arange(len(row['response']['answer_ids'])))
            for index, target in enumerate(arrays['target']):
                start, stop = row['response']['offsets'][target]
                assert arrays['token_ids'][index] == row['response']['answer_ids'][target]
                tokens.append(dict(key=row['key'], target=int(target), start=start, stop=stop,
                    text=row['response']['token_text'][target], gold=int(truth[row['key']][target]),
                    valid=bool(stop > start and arrays['token_ids'][index] not in row['response']['special_ids']),
                    grounding_score=float(arrays[score_field][index]),
                    grounding_score_alarm=bool(arrays[score_field][index] > .5),
                    choice_mass=float(arrays['choice_mass'][index])))
    suffix = '' if score_field == 'score' else '_' + score_field
    with (output / ('tokens' + suffix + '.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tokens[0]))
        writer.writeheader()
        writer.writerows(tokens)
    def summarize(rows):
        rows = [row for row in rows if row['valid']]
        return metrics(np.array([row['gold'] for row in rows]),
            np.array([row['grounding_score'] for row in rows]),
            np.array([row['grounding_score_alarm'] for row in rows]))

    result = dict(pooled=summarize(tokens), answers={row['key']: summarize([r for r in tokens if r['key']==row['key']])
        for row in records}, threshold=.5, score_field=score_field,
        scope='exposed development, external/symbolic check; not native graph detection',
        choice_mass_scope='original model A/B mass, not symbolic score confidence',
        low_choice_mass_tokens=sum(row['choice_mass'] < .1 for row in tokens),
        low_choice_mass_fraction=sum(row['choice_mass'] < .1 for row in tokens)/len(tokens),
        choice_mass_min=min(row['choice_mass'] for row in tokens),
        choice_mass_median=float(np.median([row['choice_mass'] for row in tokens])))
    if freeze['complete_answers']:
        result['span_metrics'] = annotated_span_metrics(tokens, records, ['grounding_score'])
    write_json(output / ('evaluation' + suffix + '.json'), result)
    print(result, flush=True)


def witness_capture(previous, output):
    from .logic import witness_token_constraints

    freeze = read_json(previous / 'scores_frozen.json')
    protocol = read_json(previous / 'protocol.json')
    manifest = read_json(previous / 'manifest.json')
    assert freeze['status'] == 'complete' and freeze['complete_answers']
    assert freeze['answers'] == len(manifest['records'])
    assert protocol['grounding_version'] == 'pointer' and protocol['target_limit'] is None
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'protocol.json', dict(previous=str(previous.resolve()), threshold=.5,
        schema_informed_by_exposed_annotations=True, labels_used_for_scores=False,
        witness_only='unknown does not alarm; not proof of correctness',
        strict_score='source contradiction only; disclaimer heuristic removed',
        hybrid_score='explicit witness overrides pointer; unknown retains pointer',
        scope_readout='symbolic assertion span overlap with original tokens; not internal token localization',
        new_model_forwards=0, new_graph_score=False))
    coverage = {}
    for row in manifest['records']:
        directory = output / row['key']
        directory.mkdir()
        with np.load(previous / row['key'] / 'scores.npz') as arrays:
            values = dict(arrays)
        np.testing.assert_array_equal(values['target'], np.arange(len(row['response']['answer_ids'])))
        np.testing.assert_array_equal(values['token_ids'], row['response']['answer_ids'])
        assert values['score'].shape == values['target'].shape and np.isfinite(values['score']).all()
        status, strict, heuristic, claims = witness_token_constraints(source_text(tokenizer, row['source']),
            row['response']['text'], row['response']['offsets'])
        values.update(witness_score=(status == 1).astype(float), strict_score=strict.astype(float),
            hybrid_score=apply_constraints(values['score'], status), witness_status=status,
            heuristic_alarm=heuristic)
        np.savez_compressed(directory / 'scores.npz', **values)
        write_json(directory / 'claims.json', claims)
        coverage[row['key']] = dict(tokens=len(status), recognized=int(np.count_nonzero(status)),
            alarm=int(np.sum(status == 1)), strict=int(np.sum(strict)), heuristic=int(np.sum(heuristic)),
            supported=int(np.sum(status == -1)), unknown=int(np.sum(status == 0)))
    write_json(output / 'scores_frozen.json', dict(status='complete', answers=len(manifest['records']),
        complete_answers=True, labels_read=False, new_model_forwards=0, coverage=coverage))


if __name__ == '__main__':
    main()
