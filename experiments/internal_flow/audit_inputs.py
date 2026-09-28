"""Explicit evidence roles and local claim controls for the missed-case audit."""
import json
from pathlib import Path
import numpy as np

from experiments.entropy_detection.run import read_json
from experiments.probabilistic_detection.cases import KNOWN_CASES
from experiments.native_support.ragtruth_benchmark.data import annotations
from experiments.path_conflict.data import token_text_offsets, quote_tokens
from experiments.path_conflict.paired_inputs import claim_trace
from .inputs import CASES, indices_in_quotes

MANUAL_CLAIMS = {
    '15604': ['private pension income is not taxed'],
    '9022': ['from 9:00 AM to 10:30 PM every day', 'with intimate set to false', 'WiFi is listed as no'],
    '7305': ['from 7:00 AM to 2:30 PM Monday through Saturday, and from 8:00 AM to 2:00 PM on Sunday'],
    '219': ['for more than four days'],
    '12045': ['dump a chimney of lit coals into the grill', 'clean the hot grate with a grill brush and lubricate it'],
    '12219': ['Passage 3 provides further folding instructions'],
}
CONSTRAINTS = {'15604': ['not taxed'], '9022': ['22:30', 'False', "'no'"],
    '7305': ['14:30', '14:0'], '219': ['more than'], '12045': [], '12219': ['Step 3: Fold']}


def roles(tokenizer, prompt, source, quotes, constraints):
    text, offsets = token_text_offsets(tokenizer, prompt)
    evidence = indices_in_quotes(text, offsets, quotes) if quotes else []
    constraint = [index for index in evidence if any(word.lower() in text[offsets[index,0]:offsets[index,1]].lower() for word in constraints)]
    # Multi-token constraints are matched inside the already-selected evidence span.
    for word in constraints:
        start = 0
        while (start := text.find(word, start)) >= 0:
            chosen = np.flatnonzero((offsets[:,0] < start+len(word)) & (offsets[:,1] > start))
            constraint.extend(set(chosen.tolist()) & set(evidence))
            start += len(word)
    return dict(source=source, evidence=evidence, constraint=sorted(set(constraint)),
        other_source=sorted(set(source)-set(evidence)), instruction=sorted(set(range(len(prompt)))-set(source)))


def original_items(packs, fixtures, tokenizer):
    fixture_inputs = {r['key']: r for r in read_json(fixtures/'REGRESSION_INPUTS.json')['records']}
    manual = {r['response_id']: fixture_inputs[r['key']] for r in read_json(fixtures/'REGRESSION_EVALUATION.json')['records'] if r.get('manual_positive')}
    evidence = {identity: [r['evidence'] for r in CASES if r['response_id'] == identity] for identity in KNOWN_CASES}
    result = []
    for task in ('QA', 'Summary', 'Data2txt'):
        for split in ('train', 'test'):
            meta = read_json(packs/f'{task}_{split}.json')
            root = Path(meta['source_cache'])
            selected = [r for r in meta['records'] if r['id'] in KNOWN_CASES]
            truth = annotations(root, read_json(root/'manifest.json'), selected)
            for record in selected:
                identity = record['id']
                response = read_json(root/record['directory']/'response.json')
                source = read_json(root/record['source_file'])
                source_ids = np.flatnonzero(source['source_mask']).tolist()
                spans = truth[identity]['character_spans']
                for number, span in enumerate(spans or [None]):
                    if span is None:
                        targets = []
                    else:
                        offsets = np.array(response['offsets'])
                        target = np.flatnonzero((offsets[:,0] < span['end']) & (offsets[:,1] > span['start']))
                        targets = [[int(target[0]), int(target[-1])+1]]
                    quotes = evidence[identity][number] if evidence[identity] else []
                    group = roles(tokenizer, source['prompt_with_source'], source_ids, quotes, CONSTRAINTS.get(identity, []))
                    base = dict(response_id=identity, task=task, source_id=record['source_id'],
                        prompt_ids=source['prompt_with_source'], groups=group, evidence_quotes=quotes,
                        evidence_status='reviewed local evidence' if quotes else 'no explicit support for invented procedure; evidence mask unavailable' if identity=='12045' else 'normal control, no selected evidence mask')
                    result.append(dict(base, key=f'{identity}_{number}_original', kind='original',
                        answer_ids=response['answer_ids'], answer=response['text'], spans=targets,
                        generator=record['generator'], annotation='official local span' if span else 'official empty labels'))
                    if identity in manual:
                        row = manual[identity]
                        chosen = quote_tokens(row['answer'], np.array(row['offsets']), MANUAL_CLAIMS[identity][number])
                        result.append(dict(base, key=f'{identity}_{number}_manual', kind='manual_supported',
                            answer_ids=row['token_ids'], answer=row['answer'], spans=[[int(chosen[0]),int(chosen[-1])+1]],
                            generator='manual fixture', annotation='source-checked local control; different wording/history'))
    return result


def natural_items(samples, states, tokenizer):
    records = [json.loads(line) for line in (samples/'samples.jsonl').read_text().splitlines()]
    lookup = {(str(r['source_id']),r['seed']):r for r in records}
    result = []
    for case in read_json(Path('experiments/path_conflict/paired_cases.json')):
        for side in ('supported','unsupported'):
            record = lookup[case['source_id'], case[side]['seed']]
            trace, span = claim_trace(samples, record, case[side], include_attention=False)
            prompt = int(trace['prompt_length'])
            ids = trace['token_ids'].tolist()
            with np.load(states/Path(record['trace']).name) as saved:
                source = np.flatnonzero(saved['source_mask']).tolist()
            group = roles(tokenizer, ids[:prompt], source, case['evidence'], ['Only', 'low', 'Remove'])
            result.append(dict(key=case['case_id']+'_'+side, kind='natural_'+side,
                response_id=case['case_id'], source_id=case['source_id'], task='QA',
                prompt_ids=ids[:prompt], answer_ids=ids[prompt:], answer=record['response'], spans=[list(span)],
                groups=group, evidence_quotes=case['evidence'], evidence_status='reviewed local evidence',
                generator='Meta-Llama-3.1-8B-Instruct', annotation='reviewed local claim only; whole answer unknown'))
    return result


def attach_decisions(items, tokenizer):
    """Reviewed semantic choice may occur after the official labelled span onset."""
    manual_choices = {'15604':['not taxed'], '9022':['10:30 PM','false','no.'],
        '7305':['2:30 PM'], '219':['more than'], '12219':['provides further']}
    for item in items:
        if not item['spans']:
            item['decisions'] = []
            continue
        start = item['spans'][0][0]
        identity = item['response_id']
        if item['kind'] in ('original', 'manual_supported'):
            number = int(item['key'].split('_')[1])
            if item['kind'] == 'original' and identity != '12045':
                case = [c for c in CASES if c['response_id']==identity][number]
                marker = case['anchor']+case['wrong']
                begin = item['answer'].index(marker)+len(case['anchor'])
                prefix = item['answer'][:begin]
                left = tokenizer(prefix+case['correct'],add_special_tokens=False)['input_ids']
                right = tokenizer(prefix+case['wrong'],add_special_tokens=False)['input_ids']
                start = next(i for i,(a,b) in enumerate(zip(left,right)) if a!=b)
                np.testing.assert_array_equal(item['answer_ids'][:start+1],right[:start+1])
            elif item['kind'] == 'manual_supported' and identity in manual_choices:
                text,offsets = token_text_offsets(tokenizer,item['answer_ids'])
                start = int(quote_tokens(text,offsets,manual_choices[identity][number])[0])
        item['decisions'] = [start]
