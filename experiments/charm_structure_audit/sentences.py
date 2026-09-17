"""Deterministic punctuation sentences on ORIGINAL text and character offsets.

This is a documented display unit, not a semantic claim/re-anchor detector.
Decimals, common English abbreviations and dotted initials are not split.
"""

import re

import numpy as np
import pandas as pd


END = re.compile(r'[.!?]+[\"\'”’）)\]]*(?=\s|$)|[。！？]+[\"\'”’）)\]]*|\n[ \t]*\n+')
ABBREVIATIONS = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'st.',
                 'vs.', 'etc.', 'e.g.', 'i.e.', 'fig.', 'no.', 'inc.'}


def sentence_intervals(text):
    """Half-open character intervals, independent of labels and scores."""
    intervals = []
    start = 0
    for ending in END.finditer(text):
        punctuation = ending.group().rstrip('\"\'”’）)]')
        if punctuation.endswith('.'):
            word = re.search(r'(\S+)$', text[:ending.start() + len(punctuation)])
            word = word.group().lower() if word else ''
            initials = re.search(r'(?:\b[a-z]\.){2,}$|\b[a-z]\.$', word)
            if word in ABBREVIATIONS or initials:
                continue
        stop = ending.end()
        left, right = start, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            intervals.append((left, right))
        start = stop
    if text[start:].strip():
        left = start + len(text[start:]) - len(text[start:].lstrip())
        intervals.append((left, len(text.rstrip())))
    return intervals


def locate_sentences(table, samples):
    """Map a token by maximum character overlap; ties go to earlier sentence."""
    result = []
    sentence_rows = []
    for sample in samples:
        group = table[table.id == str(sample['id'])].sort_values('token').copy()
        if set(group.source_id.astype(str)) != {str(sample['source_id'])}:
            raise ValueError('Sentence metadata source ID differs from prediction')
        text = str(sample['response'])
        offsets = np.asarray(sample['offsets'])
        if not np.array_equal(group.token, np.arange(len(offsets))):
            raise ValueError('Sentence mapping needs the complete ordered response token stream')
        if not np.array_equal(group.gold.astype(bool), sample['gold'].astype(bool)):
            raise ValueError('Sentence metadata and prediction labels differ')
        if list(group.text) != [text[a:b] for a, b in offsets]:
            raise ValueError('CSV text does not agree with the original character offsets')
        intervals = sentence_intervals(text)
        ids, crosses = map_offsets(text, offsets, intervals)
        group['char_start'], group['char_end'] = offsets.T
        group['sentence'] = ids
        group['crosses_sentence'] = crosses
        group['sentence_offset'] = -1
        group['sentence_length'] = 0
        for number, (start, end) in enumerate(intervals):
            selected = ids == number
            size = int(selected.sum())
            group.loc[selected, 'sentence_offset'] = np.arange(size)
            group.loc[selected, 'sentence_length'] = size
            sentence_rows.append(dict(id=str(sample['id']), source_id=str(sample['source_id']),
                sentence=number, char_start=start, char_end=end, tokens=size, text=text[start:end],
                error_tokens=int(group.loc[selected, 'gold'].sum()),
                TP=int(((group.gold == 1) & (group.predicted == 1) & selected).sum()),
                FN=int(((group.gold == 1) & (group.predicted == 0) & selected).sum()),
                FP=int(((group.gold == 0) & (group.predicted == 1) & selected).sum()),
                TN=int(((group.gold == 0) & (group.predicted == 0) & selected).sum())))
        group['sentence_third'] = -1
        valid = group.sentence >= 0
        group.loc[valid, 'sentence_third'] = np.minimum(2, (
            (group.loc[valid, 'sentence_offset'] + .5) * 3 /
            group.loc[valid, 'sentence_length']).astype(int))
        result.append(group)
    return pd.concat(result, ignore_index=True), pd.DataFrame(sentence_rows)


def map_offsets(text, offsets, intervals):
    ids = np.full(len(offsets), -1, int)
    crosses = np.zeros(len(offsets), bool)
    if not intervals:
        return ids, crosses
    bounds = np.asarray(intervals)
    for token, (start, end) in enumerate(offsets):
        if end <= start or not text[start:end].strip():
            continue
        overlap = np.maximum(0, np.minimum(bounds[:, 1], end) - np.maximum(bounds[:, 0], start))
        if overlap.max() > 0:
            ids[token] = int(overlap.argmax())
            crosses[token] = np.count_nonzero(overlap) > 1
    return ids, crosses
