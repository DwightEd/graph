"""Registered loss interventions: change positive weighting, never labels or graphs."""

import zlib
import numpy as np

from .positions import merge_spans

SCHEMES = ('token', 'span_equal', 'onset_half', 'random_onset_half')


def loss_weights(labels, spans, scheme, mean_span_length, seed=0):
    """Positive total is globally preserved; onset controls also preserve each span.

    mean_span_length comes from FIT spans only. Random placement matches the
    exact weight multiset in every span and is fixed across epochs.
    """
    labels = np.asarray(labels, bool)
    weights = np.ones(len(labels), dtype=np.float32)
    union = np.zeros(len(labels), bool)
    rng = np.random.default_rng(seed)
    for start, end in merge_spans(spans):
        length = end - start
        union[start:end] = True
        if scheme == 'span_equal':
            weights[start:end] = mean_span_length / length
        elif scheme in ('onset_half', 'random_onset_half') and length > 1:
            block = np.full(length, length / (2 * (length - 1)), np.float32)
            block[0] = length / 2
            if scheme == 'random_onset_half':
                block = rng.permutation(block)
            weights[start:end] = block
    if scheme not in SCHEMES or not np.array_equal(union, labels):
        raise ValueError('Unknown weighting scheme or span/label disagreement')
    return weights


def prepare_weights(records, prepared, loader, scheme, seed):
    """Read only fit metadata to determine normalization and training weights."""
    samples = []
    total, count = 0, 0
    for record in records:
        _, sample = loader(record, prepared, 'metadata')
        total += int(np.asarray(sample['gold']).sum())
        count += len(merge_spans(sample['spans']))
        samples.append((record, sample))
    if not 0 < count <= total:
        raise ValueError('Fit split has no mapped error spans')
    weights, details = {}, []
    for record, sample in samples:
        identity = str(record['id'])
        local_seed = seed + zlib.crc32(identity.encode())
        values = loss_weights(sample['gold'], sample['spans'], scheme, total / count, local_seed)
        weights[identity] = values
        for start, end in merge_spans(sample['spans']):
            details.append(dict(id=identity, source_id=str(record['source_id']), start=start,
                end=end, tokens=end-start, total_weight=float(values[start:end].sum()),
                onset_weight=float(values[start]), max_weight=float(values[start:end].max())))
    return weights, details
