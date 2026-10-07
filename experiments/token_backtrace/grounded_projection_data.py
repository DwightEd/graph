"""Strip annotations from natural inputs; deterministic source-disjoint reference."""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from state_audit.tokenization import special_token_ids
from experiments.native_support.ragtruth_benchmark.data import encode_source


DATASET = Path('/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset')
MODEL = '/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct'
CONTROLS = ('token_grounding_20261006_witness_v1',
            'token_grounding_20261006_extension_witness_v1')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def controls():
    cases = []
    for directory in CONTROLS:
        manifest = json.loads((Path('outputs') / directory / 'manifest.json').read_text())
        for row in manifest['records']:
            cases.append(dict(id=row['key'], source_id=row['original']['source_id'],
                task=row['task'], split=row['original']['split'], cohort='regression',
                generator=row['original']['generator'], source=row['source'], response=row['response']))
    return cases


def official_case(row, source, tokenizer, cohort):
    encoded = tokenizer(row['response'], add_special_tokens=False, return_offsets_mapping=True)
    response = dict(answer_ids=encoded['input_ids'], offsets=encoded['offset_mapping'],
        text=row['response'], token_text=[tokenizer.decode([i]) for i in encoded['input_ids']],
        special_ids=special_token_ids(tokenizer))
    return dict(id=str(row['id']), source_id=str(row['source_id']), task=source['task_type'],
        split=row['split'], generator=row['model'], cohort=cohort,
        source=encode_source(source, tokenizer), response=response)


def prepare(output, references=4):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    cases = controls()
    sources = {str(r['source_id']): r for r in map(json.loads, (DATASET / 'source_info.jsonl').open())}
    rows = list(map(json.loads, (DATASET / 'response.jsonl').open()))
    for identity in ('12297', '12471', '17199'):
        row = next(row for row in rows if str(row['id']) == identity)
        cases.append(official_case(row, sources[str(row['source_id'])], tokenizer, 'regression'))
    used = {row['source_id'] for row in cases}
    candidates = [r for r in rows if r['split'] == 'train' and r['model'] == 'llama-2-7b-chat']
    candidates.sort(key=lambda r: hashlib.sha256(('42:' + str(r['id'])).encode()).hexdigest())
    for task in ('QA', 'Summary', 'Data2txt'):
        selected = 0
        for row in candidates:
            source = sources[str(row['source_id'])]
            if source['task_type'] != task or str(row['source_id']) in used:
                continue
            case = official_case(row, source, tokenizer, 'reference')
            size = len(case['source']['prompt_with_source']) + len(case['response']['answer_ids'])
            if size > 1536:
                continue
            cases.append(case)
            used.add(case['source_id'])
            selected += 1
            if selected == references:
                break
        if selected != references:
            raise ValueError(f'{task}: insufficient source-disjoint reference')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'inputs.json', dict(model=MODEL, dataset=str(DATASET), cases=cases,
        reference_rule='sha256(42:id), train llama2-7, unique source, <=1536 total tokens',
        input_annotation_fields_removed=True, regression_previously_label_selected=True))
    return cases
