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


def relation_edits(text):
    """Generic source-only contrasts; candidates, not a semantic truth parser."""
    from experiments.token_backtrace.logic import DURATION

    boolean = re.compile(r"(?P<prefix>['\"][^'\"]+['\"]\s*:\s*['\"]?)(?P<value>true|false|yes|no)\b", re.I)
    auxiliary = re.compile(r"\b(is|are|was|were|do|does|did|has|have|had|will|can|could|should|would)\s+not\b", re.I)
    opposites = {'true': 'false', 'false': 'true', 'yes': 'no', 'no': 'yes'}
    for match in boolean.finditer(text):
        value = match['value']
        flipped = opposites[value.lower()]
        if value[0].isupper():
            flipped = flipped.capitalize()
        equivalent = value.lower() if value != value.lower() else value.capitalize()
        yield dict(family='boolean', start=match.start('value'), stop=match.end('value'),
                   flip=flipped, equivalent=equivalent, control='ordinary_case')
    for match in auxiliary.finditer(text):
        verb = match[1]
        contraction = {'will': "won't", 'can': "can't"}.get(verb.lower(), verb.lower()+"n't")
        if verb[0].isupper():
            contraction = contraction.capitalize()
        yield dict(family='negation', start=match.start(), stop=match.end(),
                   flip=verb, equivalent=contraction, control='negative_contraction')
    bounds = {'more than': ('less than', 'over'), 'over': ('under', 'more than'),
              'less than': ('more than', 'under'), 'under': ('over', 'less than'),
              'at least': ('at most', 'no less than'), 'at most': ('at least', 'no more than')}
    for match in DURATION.finditer(text):
        bound = (match['bound'] or '').lower()
        if bound in bounds:
            flip, equivalent = bounds[bound]
            yield dict(family='duration_bound', start=match.start('bound'), stop=match.end('bound'),
                       flip=flip, equivalent=equivalent, control='bound_paraphrase')
            yield dict(family='duration_boundary', start=match.start('bound'), stop=match.end('bound'),
                       flip='exactly', equivalent=equivalent, control='bound_paraphrase')


def repeated_source_sets(groups):
    """Same lexical values are grouping hypotheses, not certified shared facts."""
    values = {}
    field = re.compile(r"['\"][^'\"]+['\"]\s*:\s*['\"]([^'\"]+)['\"]")
    for group in groups:
        for match in field.finditer(group['text']):
            value = match[1].strip().lower()
            if re.search(r'\d', value):
                values.setdefault(value, set()).add(group['index'])
    return [dict(value=value, sources=sorted(sources))
            for value, sources in sorted(values.items()) if len(sources) > 1]


def relation_scores(original_logp, flip_logp, equivalent_logp, flip_messages, equivalent_messages):
    """Direction and control residual remain separate from unsigned influence."""
    flip_gain = odds(flip_logp) - odds(original_logp[:, None])
    control_gain = odds(equivalent_logp) - odds(original_logp[:, None])
    gain = flip_gain - np.abs(control_gain)
    # [candidate, layer, head, token] -> [token, candidate], fixed physical heads.
    flip_size = np.sqrt(np.mean(np.square(flip_messages.astype(float)), axis=(1, 2))).T
    control_size = np.sqrt(np.mean(np.square(equivalent_messages.astype(float)), axis=(1, 2))).T
    specificity = np.maximum(flip_size-control_size, 0) / np.maximum(flip_size, 1e-12)
    return gain, gain * specificity, flip_gain, control_gain, specificity
