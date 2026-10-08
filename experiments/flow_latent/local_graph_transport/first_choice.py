"""First-divergence source controls with exact one-hop reader inputs.

Labels describe literal source-program compatibility. No natural hallucination
label or wrong teacher-forced history is used by these experiments.
"""
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from experiments.native_support.evaluate import ranking


def first_records(choices):
    """Retain the paired candidates at the first divergent correct prefix."""
    records = []
    for choice in choices:
        positions = np.flatnonzero(choice['first'])
        if len(positions) != 2 or choice['labels'][:2] != [0, 1]:
            raise ValueError('First controls require one compatible/incompatible pair')
        record = {name: choice[name] for name in
                  ('id', 'source_id', 'partition', 'world', 'first_divergence', 'template_id')}
        record['candidate_ids'] = [choice['candidate_ids'][index] for index in positions]
        record['labels'] = [0, 1]
        records.append(record)
    source_groups(records)
    return records


def source_groups(records):
    """Keep both source assignments in the same minibatch."""
    groups = defaultdict(list)
    for record in records:
        groups[record['source_id']].append(record)
    for source, pair in groups.items():
        if len(pair) != 2 or {row['world'] for row in pair} != {'original', 'swapped'}:
            raise ValueError(f'{source}: expected both original and swapped captures')
        if pair[0]['candidate_ids'] != pair[1]['candidate_ids'][::-1]:
            raise ValueError(f'{source}: first candidates must exchange compatibility')
        if pair[0]['partition'] != pair[1]['partition']:
            raise ValueError(f'{source}: source appears across fit/dev')
    return list(groups.values())


def compact_graph(fields, receiver):
    """Keep receiver and true sender nodes; sender graph outputs are unused.

    The reader computes receiver edges from sender *node* embeddings, with no
    recursive graph layer. Thus this one-hop subset preserves selected logits.
    """
    valid = fields['valid'][receiver]
    senders = fields['indices'][receiver, valid]
    selected = np.unique(np.r_[senders, receiver])
    target = int(np.flatnonzero(selected == receiver)[0])
    result = {}
    for name, values in fields.items():
        result[name] = values[:, selected].copy() if name == 'local_attention' else values[selected].copy()
    result['valid'][:] = False
    result['valid'][target] = valid
    result['indices'][:] = 0
    result['indices'][target, valid] = np.searchsorted(selected, senders)
    result['local_attention'] *= result['valid'][None, :, None]
    return result, target


def first_batch(records, arrays, device):
    """Pack disjoint compact graphs, then duplicate only candidate row indices."""
    parts = defaultdict(list)
    selected_rows, candidates = [], []
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
        selected_rows.extend([offset + record['receiver_row']] * 2)
        candidate_region = slice(2 * record['record_index'], 2 * record['record_index'] + 2)
        candidates.append(arrays['candidate_vectors'][candidate_region])
        offset += record['compact_stop'] - record['compact_start']
    fields = {}
    for name, values in parts.items():
        axis = 1 if name == 'local_attention' else 0
        fields[name] = torch.from_numpy(np.concatenate(values, axis=axis)).to(device)
    fields['rows'] = torch.tensor(selected_rows, dtype=torch.long, device=device)
    fields['candidate_vectors'] = torch.from_numpy(np.concatenate(candidates)).to(device)
    return fields


def first_loss(risk, objective):
    """Risk(correct)<risk(wrong); paired softplus cannot use a shared bias."""
    labels = risk.new_tensor([0., 1.]).repeat(len(risk) // 2)
    bce = F.binary_cross_entropy_with_logits(risk, labels)
    pair = F.softplus(risk[::2] - risk[1::2]).mean()
    return bce if objective == 'bce' else bce + pair


def first_metrics(records, scores):
    """Report classification, within-prefix rank and same-word source response."""
    values = np.concatenate([scores[record['id']] for record in records])
    labels = np.tile([0, 1], len(records))
    result = ranking(labels, values)
    result['bce'] = float(np.mean(np.logaddexp(0, values) - labels * values))
    result['accuracy_at_zero'] = float(np.mean((values > 0) == labels))
    result['candidate_ranking_rate'] = float(np.mean(values[1::2] > values[::2]))
    result['compatible95_threshold'] = float(np.quantile(values[::2], .95, method='higher'))
    swaps = defaultdict(dict)
    for record in records:
        for candidate, label, risk in zip(record['candidate_ids'], record['labels'], scores[record['id']]):
            swaps[(record['source_id'], candidate)][label] = float(risk)
    pairs = list(swaps.values())
    differences = np.asarray([pair[1] - pair[0] for pair in pairs])
    result['source_swap'] = dict(pairs=len(pairs),
        direction_rate=float(np.mean(differences > 0)),
        mean_risk_difference=float(differences.mean()),
        both_signs_correct=float(np.mean([pair[0] <= 0 < pair[1] for pair in pairs])))
    return result
