"""Exposed-case mechanism probes; curated alternatives are discovery-only."""
import json


CASES = [
    dict(name='pension', response_id='15604', anchor='private pension income is ',
         wrong='taxed at the normal state tax rate of 5%.', correct='not taxed.', equivalent='untaxed.',
         evidence=['private pension income is not taxed']),
    dict(name='hours', response_id='9022', anchor='from 9:00 AM to ', wrong='2:30 AM.',
         correct='10:30 PM.', equivalent='22:30.', evidence=["'Monday': '9:0-22:30'", "'Sunday': '9:0-22:30'"]),
    dict(name='intimate', response_id='9022', anchor='with a casual and ', wrong='intimate atmosphere',
         correct='classy atmosphere', equivalent='classy ambience',
         evidence=["'intimate': False", "'classy': True"]),
    dict(name='wifi', response_id='9022', anchor='offers outdoor seating ', wrong='and WiFi,',
         correct='but no WiFi,', equivalent='without WiFi,', evidence=["'WiFi': 'no'"]),
    dict(name='duration', response_id='219', anchor='park for ', wrong='four days',
         correct='more than four days', equivalent='over four days', evidence=['For more than four days']),
    dict(name='folding', response_id='12219', anchor='Passage 3 ',
         wrong='does not provide instructions for folding a quilt.',
         correct='provides instructions for folding a quilt.',
         equivalent='gives instructions for folding a quilt.',
         evidence=['Step 3: Fold the lower left corner up toward the center of the quilt making sure the fold is on the bias.']),
    dict(name='weekday_hours', response_id='7305', anchor='business hours are as follows: ',
         wrong='Monday to Sunday from 7:00 AM to 4:00 PM.',
         correct='Monday to Saturday from 7:00 AM to 2:30 PM, and Sunday from 8:00 AM to 2:00 PM.',
         equivalent='Monday through Saturday from 07:00 to 14:30, and Sunday from 08:00 to 14:00.',
         evidence=["'Monday': '7:0-14:30'", "'Sunday': '8:0-14:0'"]),
]


def indices_in_quotes(text, offsets, quotes):
    result = set()
    for quote in quotes:
        if text.count(quote) != 1:
            raise ValueError(f'Quote is not unique: {quote}')
        start = text.index(quote)
        stop = start + len(quote)
        result.update(i for i, (left, right) in enumerate(offsets) if left < stop and right > start)
    return sorted(result)


def prepare(tokenizer, dataset):
    responses = {str(row['id']): row for row in map(json.loads, (dataset/'response.jsonl').read_text().splitlines())}
    sources = {str(row['source_id']): row for row in map(json.loads, (dataset/'source_info.jsonl').read_text().splitlines())}
    probes = []
    for case in CASES:
        response = responses[case['response_id']]
        source = sources[str(response['source_id'])]
        answer = response['response']
        marker = case['anchor'] + case['wrong']
        if answer.count(marker) != 1:
            raise ValueError(f"Target is not unique: {case['name']}")
        start = answer.index(marker) + len(case['anchor'])
        rendered = tokenizer.apply_chat_template([dict(role='user', content=source['prompt'])],
                                                 tokenize=False, add_generation_prompt=True)
        prefix = rendered + answer[:start]
        variants = {side: tokenizer(prefix + case[side], add_special_tokens=False,
                    return_offsets_mapping=True) for side in ('correct', 'wrong', 'equivalent')}
        good, bad = variants['correct']['input_ids'], variants['wrong']['input_ids']
        common = 0
        while common < min(len(good), len(bad)) and good[common] == bad[common]:
            common += 1
        if common == min(len(good), len(bad)):
            raise ValueError('No competing token at the branch point')
        material = source['source_info']
        source_text = material['passages'] if source['task_type'] == 'QA' else str(material)
        offsets = variants['wrong']['offset_mapping'][:common]
        full_text = prefix + case['wrong']
        source_keys = indices_in_quotes(full_text, offsets, [source_text])
        evidence = indices_in_quotes(full_text, offsets, case['evidence'])
        assert set(evidence) <= set(source_keys)
        probes.append(dict(case=case['name'], response_id=case['response_id'], source_id=source['source_id'],
            generator=response['model'], task=source['task_type'], prefix_ids=bad[:common],
            variants={side: encoded['input_ids'] for side, encoded in variants.items()},
            candidate_ids=[good[common], bad[common]],
            candidate_texts={side: case[side] for side in variants},
            groups=dict(source=source_keys, evidence=evidence,
                        other_source=sorted(set(source_keys)-set(evidence))),
            target_character_start=start, intervention_query=common-1,
            answer_start=next(i for i, (_, stop) in enumerate(offsets) if stop > len(rendered)),
            interpretation='same-prefix alternatives; post-branch states are teacher-forced, not independent natural correct/wrong generations'))
    return probes
