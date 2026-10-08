"""Compare source compatibility before and after the actual first candidate."""
from collections import defaultdict

import numpy as np
import torch

from .first_choice import first_metrics


def timing_records(programs):
    """Order each source as original/swapped, compatible/incompatible."""
    sources = defaultdict(list)
    for program in programs:
        sources[program['source_id']].append(program)
    records = []
    for pair in sources.values():
        for condition in ('original', 'swapped'):
            candidates = [row for row in pair if row['world'] == condition]
            candidates.sort(key=lambda row: row['proposal_index'] != row['correct_index'])
            correct, wrong = candidates
            position = correct['first_divergence']
            if position != wrong['first_divergence']:
                raise ValueError('Paired first divergence positions differ')
            if correct['answer_ids'][:position] != wrong['answer_ids'][:position]:
                raise ValueError('Prechoice conditions must have the same past prefix')
            if correct['prompt_with_source'] != wrong['prompt_with_source']:
                raise ValueError('Candidate conditions must share the source prompt')
            for label, program in enumerate(candidates):
                records.append(dict(id=program['id'], source_id=program['source_id'],
                    partition=program['partition'], world=condition, label=label,
                    first_divergence=position, candidate_id=program['answer_ids'][position],
                    template_id=program['template_id']))
    return records


def timing_source_groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[record['source_id']].append(record)
    for source, rows in groups.items():
        if len(rows) != 4 or [row['label'] for row in rows] != [0, 1, 0, 1]:
            raise ValueError(f'{source}: need four ordered source/candidate controls')
        if len({row['partition'] for row in rows}) != 1:
            raise ValueError(f'{source}: source overlaps fit/dev')
    return list(groups.values())


def timing_batch(records, arrays, device):
    """Each compact graph scores the candidate actually present in its capture."""
    parts = defaultdict(list)
    rows, candidates = [], []
    offset = 0
    for record in records:
        region = slice(record['compact_start'], record['compact_stop'])
        for name, values in arrays.items():
            if name == 'candidate_vectors':
                continue
            value = values[:, region].copy() if name == 'local_attention' else values[region].copy()
            if name == 'indices':
                value += offset
            parts[name].append(value)
        rows.append(offset + record['receiver_row'])
        candidates.append(arrays['candidate_vectors'][record['record_index']])
        offset += record['compact_stop'] - record['compact_start']
    fields = {}
    for name, values in parts.items():
        axis = 1 if name == 'local_attention' else 0
        fields[name] = torch.from_numpy(np.concatenate(values, axis=axis)).to(device)
    fields['rows'] = torch.tensor(rows, dtype=torch.long, device=device)
    fields['candidate_vectors'] = torch.from_numpy(np.stack(candidates)).to(device)
    return fields


def timing_metrics(records, scores):
    """Reuse paired-prefix and same-word source-swap metric definitions."""
    paired_records, predictions = [], {}
    for source in timing_source_groups(records):
        for correct, wrong in (source[:2], source[2:]):
            paired_records.append(dict(id=correct['id'], source_id=correct['source_id'],
                candidate_ids=[correct['candidate_id'], wrong['candidate_id']], labels=[0, 1]))
            predictions[correct['id']] = np.array([scores[correct['id']], scores[wrong['id']]])
    return first_metrics(paired_records, predictions)
