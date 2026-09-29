"""Deterministic source-disjoint roster, without loading correctness labels."""
import hashlib
from pathlib import Path

from experiments.decision_risk_flow.data import inputs, read_json, write_json


def order(value):
    return hashlib.sha256(('token-evidence-42/' + str(value)).encode()).hexdigest()


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    records = [row for row in read_json('outputs/anchored_flow_20260929_v1/manifest.json')['records']
               if row['role'] != 'fit']
    previous = read_json('outputs/context_response_20260928_v4/manifest.json')['records']
    for row in previous:
        if row['role'] != 'dev':
            continue
        prompt, response = inputs(row)
        source = read_json(Path(row['root']) / row['source_file'])
        records.append(dict(key=row['key'], dataset='ragtruth', task=row['task'], role='dev',
            pair=row['source_id'], original=row, prompt=prompt, source=source, response=response))
    excluded = {str(row['pair']) for row in records if row['dataset'] == 'ragtruth'}
    excluded.update(str(row['source_id']) for row in previous)
    packs = Path('outputs/probabilistic_detection_20260928_full/packs')
    for task in ('QA', 'Summary', 'Data2txt'):
        metadata = read_json(packs / f'{task}_test.json')
        sources = sorted({row['source_id'] for row in metadata['records']
                          if str(row['source_id']) not in excluded}, key=order)[:12]
        generators = sorted({row['generator'] for row in metadata['records']})
        for index, source_id in enumerate(sources):
            candidates = [row for row in metadata['records'] if row['source_id'] == source_id
                          and row['generator'] == generators[index % len(generators)]]
            selected = min(candidates, key=lambda row: order(row['id']))
            row = dict(selected, root=metadata['source_cache'], kind='observer', key=selected['id'])
            prompt, response = inputs(row)
            source = read_json(Path(row['root']) / row['source_file'])
            records.append(dict(key=row['id'], dataset='ragtruth', task=task, role='heldout',
                pair=source_id, original=row, prompt=prompt, source=source, response=response))
    assert len({row['key'] for row in records}) == len(records)
    partitions = {role: {(row['dataset'], str(row['pair'])) for row in records if row['role'] == role}
                  for role in ('dev', 'case', 'heldout')}
    assert not partitions['dev'] & (partitions['case'] | partitions['heldout'])
    for row in records:
        (output / row['key']).mkdir()
    write_json(output / 'manifest.json', dict(records=records, labels_used=False,
        window=16, batch_size=8, seed=42, primary='reset_cad_tail',
        source_ablation='mask source keys at all layers; preserve tokens and absolute RoPE positions',
        history='full causal history vs last16 tokens independently recomputed from clean prompt KV',
        thresholds='task-wise answer/source-equal unlabeled dev mixture95; GSM step means only for evaluation',
        heldout='36 answer/source pairs, 12/task, generator-balanced; historical official test already exposed'))
    return records
