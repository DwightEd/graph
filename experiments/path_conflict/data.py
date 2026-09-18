"""Read existing native samples and compile reviewed claim positions once.

No old response labels, detector scores, or re-created chat templates are used.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAMPLES = '/share/home/tm902089733300000/a903202310/lys/research/reanchor/outputs/samples_20260911_145421_235'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_run(directory):
    directory = Path(directory)
    settings = json.loads((directory / 'settings.json').read_text())
    samples = read_jsonl(directory / 'samples.jsonl')
    prompts = {str(row['source_id']): row['prompt'] for row in read_jsonl(directory / 'prompts.jsonl')}
    return settings, samples, prompts


def inventory(directory, output, cases):
    settings, samples, prompts = read_run(directory)
    rows = []
    for sample in samples:
        path = Path(directory) / sample['trace']
        arrays = {}
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                arrays = dict(array_fields='|'.join(saved.files), prompt_tokens=int(saved['prompt_length']),
                              saved_tokens=len(saved['token_ids']), stored_top_candidates=saved['top_ids'].shape[1])
        rows.append(dict(sample, **arrays, source_id=str(sample['source_id']), trace_available=path.is_file(),
                         abstention=sample['response'].strip() == 'Unable to answer based on given passages.',
                         whole_answer_label='not_annotated'))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'samples.csv', index=False)
    grouped = table.groupby('source_id').agg(answers=('seed', 'size'), tokens=('tokens', 'sum'),
        unique_texts=('response', 'nunique'), abstentions=('abstention', 'sum'),
        traces_available=('trace_available', 'sum')).reset_index()
    grouped.to_csv(output / 'sources.csv', index=False)
    matches = []
    for source, group in table.groupby('source_id'):
        for left, right in combinations(group.to_dict('records'), 2):
            matches.append(dict(source_id=source, seed_a=left['seed'], seed_b=right['seed'],
                identical_response=left['response'] == right['response'], comparison_label='unreviewed'))
    pd.DataFrame(matches).to_csv(output / 'same_question_pairs.csv', index=False)
    validate_text_cases(samples, prompts, cases)
    write_json(output / 'input_settings.json', settings)
    write_json(output / 'reviewed_cases.json', cases)
    print(grouped.to_string(index=False), flush=True)
    return settings, samples, prompts


def validate_text_cases(samples, prompts, cases):
    """Scientific boundaries in one preflight, not branches in native computation."""
    lookup = {(str(row['source_id']), row['seed']): row for row in samples}
    for case in cases:
        for side in ('supported', 'unsupported'):
            annotation = case[side]
            response = lookup[case['source_id'], annotation['seed']]['response']
            assert response.count(annotation['target']) == 1, (case['case_id'], side, 'target not unique')
        for quote in case['evidence'] + case['value_source']:
            assert prompts[case['source_id']].count(quote) == 1, (case['case_id'], 'source quote not unique')


def token_text_offsets(tokenizer, ids):
    """Use the SAVED prompt IDs, with an exact fast-tokenizer round trip."""
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    np.testing.assert_array_equal(encoded['input_ids'], ids)
    return text, np.asarray(encoded['offset_mapping'])


def quote_tokens(text, offsets, quote):
    start = text.index(quote)
    stop = start + len(quote)
    return np.flatnonzero((offsets[:, 0] < stop) & (offsets[:, 1] > start))


def source_groups(tokenizer, prompt_ids, query_count, case):
    rendered, offsets = token_text_offsets(tokenizer, prompt_ids)
    evidence = np.unique(np.concatenate([quote_tokens(rendered, offsets, q) for q in case['evidence']]))
    value_source = np.unique(np.concatenate([quote_tokens(rendered, offsets, q) for q in case['value_source']]))
    assert not np.intersect1d(evidence, value_source).size, 'Evidence/value token masks overlap'
    prompt = np.arange(len(prompt_ids))
    other = np.setdiff1d(prompt, np.r_[evidence, value_source])
    history = np.arange(len(prompt_ids), query_count)
    return dict(evidence=evidence, value_source=value_source, history=history, other_prompt=other)


def compile_side(directory, tokenizer, record, annotation, case):
    """Decision query is immediately BEFORE the reviewed continuation, never after it."""
    with np.load(Path(directory) / record['trace'], allow_pickle=False) as saved:
        ids = saved['token_ids'].tolist()
        prompt_length = int(saved['prompt_length'])
        response_ids = ids[prompt_length:]
        pieces = saved['token_text'][prompt_length:].tolist()
        response = ''.join(pieces)
        ends = np.cumsum([len(piece) for piece in pieces])
        offsets = np.column_stack((np.r_[0, ends[:-1]], ends))
        start = int(quote_tokens(response, offsets, annotation['target'])[0])
        count = prompt_length + start
        prefix_ids = ids[:count]
        prefix_text = tokenizer.decode(prefix_ids, clean_up_tokenization_spaces=False)
        candidates = []
        for candidate in case['candidates']:
            continuation = tokenizer.encode(candidate, add_special_tokens=False)
            decoded = tokenizer.decode(prefix_ids + continuation, clean_up_tokenization_spaces=False)
            assert decoded == prefix_text + candidate, 'Candidate does not append exactly at saved token boundary'
            candidates.append(continuation)
        return dict(prefix_ids=prefix_ids, prompt_length=prompt_length, response_step=start,
            sampled_next=int(response_ids[start]), candidates=candidates,
            groups=source_groups(tokenizer, ids[:prompt_length], count, case),
            top_ids=saved['top_ids'][start], top_logits=saved['top_logits'][start],
            log_normalizer=float(saved['log_normalizer'][start]), trace=record['trace'], seed=record['seed'])


def compile_cases(directory, tokenizer, samples, prompts, cases, output):
    lookup = {(str(row['source_id']), row['seed']): row for row in samples}
    compiled, rows = [], []
    for case in cases:
        sides = {}
        for side in ('supported', 'unsupported'):
            record = lookup[case['source_id'], case[side]['seed']]
            sides[side] = compile_side(directory, tokenizer, record, case[side], case)
            values = sides[side]
            observed = 0 if side == 'supported' else 1
            assert values['candidates'][observed][0] == values['sampled_next'], 'Candidate/observed token mismatch'
            rows.append(dict(case_id=case['case_id'], side=side, seed=record['seed'], trace=record['trace'],
                response_step=values['response_step'], query=len(values['prefix_ids']) - 1,
                prefix_tokens=len(values['prefix_ids']), sampled_next=values['sampled_next'],
                correct_tokens=len(values['candidates'][0]), wrong_tokens=len(values['candidates'][1]),
                prefix_tail=tokenizer.decode(values['prefix_ids'][-40:], clean_up_tokenization_spaces=False),
                supported_candidate=case['candidates'][0], unsupported_candidate=case['candidates'][1],
                evidence_tokens=len(values['groups']['evidence']), value_source_tokens=len(values['groups']['value_source'])))
        a, b = sides.values()
        np.testing.assert_array_equal(a['prefix_ids'][:a['prompt_length']], b['prefix_ids'][:b['prompt_length']])
        same = a['prefix_ids'] == b['prefix_ids']
        compiled.append(dict(case=case, sides=sides, identical_prefix=same))
    pd.DataFrame(rows).to_csv(output / 'decisions.csv', index=False)
    return compiled
