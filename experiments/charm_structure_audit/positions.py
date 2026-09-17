"""Locate errors within ANNOTATED spans, not within guessed sentences."""

import numpy as np
import pandas as pd


def merge_spans(intervals):
    merged = []
    for start, end in sorted((int(a), int(b)) for a, b in intervals):
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def annotate(tokens, spans):
    """Keep adjacent annotations distinct. Every error token gets one position."""
    table = tokens.copy().reset_index(drop=True)
    for name in ('span_index', 'span_start', 'span_end', 'span_length', 'offset', 'from_end', 'third'):
        table[name] = -1
    for identity, group in spans.groupby('id'):
        intervals = merge_spans(zip(group.start, group.end))
        for index, (start, end) in enumerate(intervals):
            selected = (table.id == str(identity)) & table.token.between(start, end - 1)
            offset = table.loc[selected, 'token'].to_numpy() - start
            third = np.minimum(2, ((offset + .5) * 3 / (end - start)).astype(int))
            values = np.column_stack((np.full(len(offset), index), np.full(len(offset), start),
                np.full(len(offset), end), np.full(len(offset), end - start), offset, end - start - 1 - offset, third))
            table.loc[selected, ['span_index', 'span_start', 'span_end', 'span_length', 'offset', 'from_end', 'third']] = values
    if not np.array_equal(table.span_index >= 0, table.gold.astype(bool)):
        raise ValueError('Span union does not equal the saved error labels')
    table['outcome'] = np.where(table.gold == 1, np.where(table.predicted == 1, 'TP', 'FN'),
                                np.where(table.predicted == 1, 'FP', 'TN'))
    return table


def position_masks(errors):
    return dict(first=errors.offset == 0, interior=(errors.offset > 0) & (errors.from_end > 0),
                last=(errors.from_end == 0) & (errors.offset > 0),
                early_third=errors.third == 0, middle_third=errors.third == 1, late_third=errors.third == 2)


def counts(group, total_hits):
    hits = int(group.predicted.sum())
    return dict(tokens=len(group), hits=hits, recall=hits / len(group) if len(group) else None,
                share_of_all_hits=hits / total_hits if total_hits else None,
                mean_score=float(group.score.mean()) if len(group) else None,
                spans=len(group[['id', 'span_start']].drop_duplicates()))


def summarize(table):
    errors = table[table.gold == 1]
    total_hits = int(errors.predicted.sum())
    rows = []
    scopes = dict(all=errors, answer_first_span=errors[errors.span_index == 0],
                  later_span=errors[errors.span_index > 0])
    for scope, subset in scopes.items():
        for region, mask in position_masks(subset).items():
            rows.append(dict(scope=scope, region=region, **counts(subset[mask], total_hits)))
    offsets = [dict(offset=int(offset), **counts(group, total_hits))
               for offset, group in errors.groupby('offset')]
    lengths = []
    for lower, upper in ((1, 4), (5, 16), (17, 32), (33, 10**9)):
        subset = errors[errors.span_length.between(lower, upper)]
        for region, mask in position_masks(subset).items():
            lengths.append(dict(length_group=f'{lower}-{upper}', region=region, **counts(subset[mask], total_hits)))
    return pd.DataFrame(rows), pd.DataFrame(offsets), pd.DataFrame(lengths)
