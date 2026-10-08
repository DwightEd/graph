"""Paired next-token choices evaluated on the same correct source prefix.

The other quote supplies a wrong token candidate, never a teacher-forced wrong
history. Source labels describe exact field copying, not natural factual truth.
"""
from collections import Counter

import numpy as np


def build_choice_records(programs):
    """Use one correct capture per source world and two candidates per row."""
    worlds = {}
    for program in programs:
        worlds.setdefault((program['source_id'], program['world']), []).append(program)

    records = []
    for pair in worlds.values():
        correct = next(row for row in pair if row['proposal_index'] == row['correct_index'])
        other = next(row for row in pair if row['proposal_index'] != row['correct_index'])
        start = correct['first_divergence']
        rows = [position for position in range(start, min(correct['tokens'], other['tokens']))
                if correct['answer_ids'][position] != other['answer_ids'][position]]
        candidate_ids = []
        for position in rows:
            candidate_ids.extend([correct['answer_ids'][position], other['answer_ids'][position]])
        records.append(dict(id=correct['id'], source_id=correct['source_id'],
            partition=correct['partition'], world=correct['world'],
            correct_proposal_index=correct['proposal_index'], tokens=correct['tokens'],
            answer_ids=correct['answer_ids'], other_answer_ids=other['answer_ids'],
            first_divergence=start, rows=np.repeat(rows, 2).tolist(),
            candidate_ids=candidate_ids, labels=[0, 1] * len(rows),
            first=[position == start for position in rows for _ in range(2)],
            template_id=correct['template_id']))
    return records


def source_choice_weights(records):
    """Each source has unit weight, half first-choice and half later choices.

    A source with no later differing candidate gives all its weight to first
    choices. Counts include both source worlds and both candidate directions.
    """
    first_counts = Counter()
    later_counts = Counter()
    for record in records:
        first_counts[record['source_id']] += sum(record['first'])
        later_counts[record['source_id']] += len(record['first']) - sum(record['first'])

    weights = {}
    for record in records:
        source = record['source_id']
        first_share = .5 if later_counts[source] else 1.
        first_weight = first_share / first_counts[source]
        later_weight = .5 / later_counts[source] if later_counts[source] else 0.
        weights[record['id']] = np.asarray(
            [first_weight if first else later_weight for first in record['first']], dtype=np.float32)
    return weights


def choice_coverage(records):
    result = {}
    for partition in ('fit', 'dev'):
        selected = [row for row in records if row['partition'] == partition]
        result[partition] = dict(sources=len({row['source_id'] for row in selected}),
            correct_capture_graphs=len(selected),
            paired_positions=sum(len(row['rows']) // 2 for row in selected),
            choice_examples=sum(len(row['rows']) for row in selected),
            first_choice_examples=sum(sum(row['first']) for row in selected),
            later_choice_examples=sum(len(row['first']) - sum(row['first']) for row in selected),
            sources_without_later=len({row['source_id'] for row in selected} -
                {row['source_id'] for row in selected if not all(row['first'])}))
    return result
