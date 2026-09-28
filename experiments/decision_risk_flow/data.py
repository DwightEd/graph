"""Choose sources without labels; retrieve labels only for the requested phase."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.probabilistic_detection.cases import KNOWN_CASES
from experiments.native_support.ragtruth_benchmark.data import annotations

TASKS = ('QA', 'Summary', 'Data2txt')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def order(identity):
    return hashlib.sha256(('risk-response-42/' + str(identity)).encode()).hexdigest()


def prepare(packs, output, fit_sources, dev_sources, test_sources):
    tables = [read_json(packs / f'{task}_{split}.json')
              for task in TASKS for split in ('train', 'test')]
    excluded = {r['source_id'] for table in tables for r in table['records'] if r['id'] in KNOWN_CASES}
    # Original-generation diagnostic sources are never used for fitting either.
    excluded.update(('14304', '14315', '14325', '14375'))
    records = []
    for table in tables:
        for role, count in [('fit', fit_sources), ('dev', dev_sources), ('test', test_sources)]:
            rows = [r for r in table['records'] if r['partition'] == role and r['source_id'] not in excluded]
            sources = sorted({r['source_id'] for r in rows}, key=order)[:count]
            for source in sources:
                row = min((r for r in rows if r['source_id'] == source), key=lambda r: order(r['id']))
                records.append(dict(row, role=role, key=row['id'], root=table['source_cache']))
        for row in table['records']:
            if row['id'] in KNOWN_CASES:
                records.append(dict(row, role='regression', key=row['id'], root=table['source_cache']))
    fit_ids = sorted({r['source_id'] for r in records if r['role'] == 'fit'}, key=order)
    folds = {source: index % 5 for index, source in enumerate(fit_ids)}
    for row in records:
        row['fold'] = folds.get(row['source_id'], -1)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('static', 'responses', 'models', 'scores'):
        (output / name).mkdir()
    write_json(output / 'manifest.json', dict(records=records, folds=5, excluded_sources=sorted(excluded),
        fit_sources_per_task=fit_sources, dev_sources_per_task=dev_sources, test_sources_per_task=test_sources,
        selection='source hash then one answer per source by response hash; no labels',
        observer_not_original_generator=True, original_prompt_unchanged=True, prompt_roles_used=False))
    return records


def inputs(record):
    root = Path(record['root'])
    prompt = read_json(root / record['source_file'])['prompt_with_source']
    response = read_json(root / record['directory'] / 'response.json')
    return prompt, response


def labels(records):
    root = Path(records[0]['root'])
    truth = annotations(root, read_json(root / 'manifest.json'), records)
    return {row['key']: np.asarray(truth[row['id']]['labels'], dtype=np.int64) for row in records}
