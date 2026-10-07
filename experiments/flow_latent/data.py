"""Source-disjoint, annotation-free model inputs for the regression pilot."""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from experiments.token_backtrace.grounded_projection_data import (
    DATASET, MODEL, official_case, write_json)


def prepare(output):
    existing = Path('outputs/grounded_projection_20261007_v2_cycle/inputs.json')
    inputs = json.loads(existing.read_text())
    cases = inputs['cases']
    used = {case['source_id'] for case in cases}
    sources = {str(row['source_id']): row for row in
               map(json.loads, (DATASET / 'source_info.jsonl').open())}
    # Whitelist text/identity fields here; annotations never enter persisted inputs.
    rows = []
    for line in (DATASET / 'response.jsonl').open():
        row = json.loads(line)
        if row['split'] == 'train' and row['model'] == 'llama-2-7b-chat':
            rows.append({name: row[name] for name in
                         ('id', 'source_id', 'split', 'model', 'response')})
    rows.sort(key=lambda row: hashlib.sha256(
        ('flow-latent-fit:' + str(row['id'])).encode()).hexdigest())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for task in ('QA', 'Summary', 'Data2txt'):
        count = 0
        for row in rows:
            source = sources[str(row['source_id'])]
            if source['task_type'] != task or str(row['source_id']) in used:
                continue
            case = official_case(row, source, tokenizer, 'fit')
            length = len(case['source']['prompt_with_source']) + len(case['response']['answer_ids'])
            if length > 1536:
                continue
            cases.append(case)
            used.add(case['source_id'])
            count += 1
            if count == 8:
                break
        assert count == 8, task
    assert len(used) == len(cases)
    output.mkdir(parents=True, exist_ok=False)
    inputs.update(cases=cases, fit_rule='sha256(flow-latent-fit:id), 8/source task, <=1536',
                  annotation_fields_used_for_fit=False)
    write_json(output / 'inputs.json', inputs)
    return inputs


def digest_files(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
