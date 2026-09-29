"""Label-free span addresses, evidence profiles, and variable-length regimes."""
import re

import numpy as np
from scipy.special import rel_entr


def normalize(values):
    total = values.sum(axis=-1, keepdims=True)
    return np.divide(values, total, out=np.zeros_like(values, dtype=float), where=total > 0)


def js(left, right):
    mixture = (left + right) / 2
    return (rel_entr(left, mixture).sum(-1) + rel_entr(right, mixture).sum(-1)) / 2


def source_spans(prompt, source_mask, tokenizer):
    """Partition known source regions at punctuation, never at annotated evidence."""
    text = [tokenizer.decode([token]) for token in prompt]
    groups, current = [], []
    for key, (part, is_source) in enumerate(zip(text, source_mask)):
        if not is_source:
            if current:
                groups.append(current)
                current = []
            continue
        current.append(key)
        # Colon stays with its value, and decimal points stay within numbers.
        next_text = text[key + 1] if key + 1 < len(text) else ''
        sentence_end = re.search(r'[.!?][\'"}\]]*$', part.rstrip())
        boundary = re.search(r'[,;\n]', part) or (sentence_end and (not next_text or next_text[0].isspace()))
        if boundary:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    merged = []
    for group in groups:
        content = ''.join(text[key] for key in group)
        if not re.search(r'\w', content) and merged and merged[-1][-1] + 1 == group[0]:
            merged[-1].extend(group)
        else:
            merged.append(group)
    return [dict(index=index, keys=keys, text=''.join(text[key] for key in keys))
            for index, keys in enumerate(merged)]


def group_roots(trace, groups):
    """Keep signed embedding-gate and gradient norm measurements distinct."""
    signed = np.stack([trace['root_effect'][:, group['keys']].sum(-1) for group in groups], -1)
    energy = np.stack([np.sqrt((trace['root_norm'][:, group['keys']] ** 2).sum(-1))
                       for group in groups], -1)
    return signed, energy


def joint_profile(root_energy, message_energy):
    """Equal geometric consensus for source ranking, not error probabilities."""
    root = normalize(root_energy)
    message = normalize(message_energy)
    floor = 1e-8 / root.shape[-1]
    return normalize(np.sqrt((root + floor) * (message + floor)))


def adjacent_change(values):
    result = np.zeros(len(values))
    numerator = np.linalg.norm(values[1:] - values[:-1], axis=-1)
    denominator = np.linalg.norm(values[1:], axis=-1) + np.linalg.norm(values[:-1], axis=-1)
    result[1:] = numerator / np.maximum(denominator, np.finfo(float).tiny)
    return result


def upper_tail(values):
    """Unlabelled robust scale; descriptive, not a false-positive guarantee."""
    median = np.median(values[1:])
    mad = np.median(np.abs(values[1:] - median))
    return float(median + 3 * 1.4826 * mad)


def regimes(profile, hidden_change, message_change):
    """Boundaries organize evidence provenance; never broadcast risk scores."""
    source_change = np.zeros(len(profile))
    source_change[1:] = js(profile[1:], profile[:-1])
    changes = np.stack((source_change, hidden_change, message_change), -1)
    thresholds = np.array([upper_tail(changes[:, index]) for index in range(3)])
    unusual = changes > thresholds
    # Require agreement of two modalities, including evidence-address movement.
    boundary = unusual[:, 0] & (unusual[:, 1] | unusual[:, 2])
    boundary[0] = True
    starts = np.flatnonzero(boundary)
    spans = [dict(start=int(start), stop=int(stop))
             for start, stop in zip(starts, np.r_[starts[1:], len(profile)])]
    return spans, changes, thresholds


def odds(logp):
    clipped = np.minimum(logp, -np.finfo(np.float32).eps)
    return clipped - np.log(-np.expm1(clipped))
