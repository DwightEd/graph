"""Count detected ERROR spans and unflagged NORMAL intervals, then locate tokens.

All denominators are explicit. Gold labels define diagnostic populations only;
no point-adjustment, test threshold fitting, or score-based normal matching.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_predictions, read_tables, read_json, write_json
from .positions import annotate, merge_spans
from .sentences import locate_sentences


INTERVAL_COLUMNS = ['id', 'source_id', 'population', 'start', 'end', 'pair_error_start']


def mask_runs(mask):
    change = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))


def diagnostic_intervals(table, pairs, tier):
    """Keep natural error spans, normal runs and matched windows SEPARATE."""
    rows = []
    for identity, group in table.groupby('id', sort=False):
        base = dict(id=str(identity), source_id=str(group.source_id.iloc[0]), pair_error_start=-1)
        error = group[group.gold == 1][['span_start', 'span_end']].drop_duplicates()
        for start, end in error.itertuples(index=False, name=None):
            rows.append(dict(base, population='gold_error', start=int(start), end=int(end)))
        for start, end in mask_runs(group.sort_values('token').gold == 0):
            rows.append(dict(base, population='normal_run', start=int(start), end=int(end)))
    for pair in pairs:
        if pair['tier'] != tier:
            continue
        for population, key in (('matched_error', 'error_start'), ('matched_normal', 'normal_start')):
            start = int(pair[key])
            rows.append(dict(id=str(pair['id']), source_id=str(pair['source_id']), population=population,
                start=start, end=start + int(pair['length']), pair_error_start=int(pair['error_start'])))
    return pd.DataFrame(rows, columns=INTERVAL_COLUMNS)


def interval_tokens(table, intervals):
    groups = {str(key): value.set_index('token', drop=False) for key, value in table.groupby('id')}
    rows = []
    used = {}
    for interval in intervals.to_dict('records'):
        start, end = interval['start'], interval['end']
        frame = groups[interval['id']].loc[np.arange(start, end)].copy()
        is_error = interval['population'] in ('gold_error', 'matched_error')
        if not np.all(frame.gold == int(is_error)):
            raise ValueError('Diagnostic interval and token labels disagree')
        if str(frame.source_id.iloc[0]) != interval['source_id']:
            raise ValueError('Matched source identity differs from prediction')
        key = interval['id'], interval['population']
        coordinates = set(range(start, end))
        if used.setdefault(key, set()) & coordinates:
            raise ValueError('Overlapping intervals would duplicate tokens in one population')
        used[key].update(coordinates)
        for name, value in interval.items():
            frame[name] = value
        frame['interval_offset'] = np.arange(end - start)
        frame['interval_third'] = np.minimum(2, ((frame.interval_offset + .5) * 3 / (end - start)).astype(int))
        rows.append(frame.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def one_interval(group, text):
    """Any hit, 80% coverage and all-token coverage are different criteria."""
    alarm = group.predicted.astype(bool).to_numpy()
    where = np.flatnonzero(alarm)
    first = group.iloc[0]
    valid = group[group.sentence >= 0]
    detected = group.iloc[where[0]] if len(where) else None
    return dict(id=str(first.id), source_id=str(first.source_id), population=first.population,
        start=int(first.start), end=int(first.end), pair_error_start=int(first.pair_error_start),
        tokens=len(group), alarm_tokens=int(alarm.sum()), alarm_fraction=float(alarm.mean()),
        mean_score=float(group.score.mean()), any_alarm=bool(alarm.any()), all_clear=not bool(alarm.any()),
        alarm80=bool(alarm.mean() >= .8), all_alarm=bool(alarm.all()), onset_alarm=bool(alarm[0]),
        first_alarm_offset=int(where[0]) if len(where) else None,
        first_alarm_fraction=(float(where[0]) + .5) / len(group) if len(where) else None,
        start_sentence_offset=int(valid.sentence_offset.iloc[0]) if len(valid) else None,
        end_sentence_offset=int(valid.sentence_offset.iloc[-1]) if len(valid) else None,
        first_alarm_sentence=int(detected.sentence) if detected is not None else None,
        first_alarm_sentence_offset=int(detected.sentence_offset) if detected is not None else None,
        sentence_start=int(valid.sentence.min()) if len(valid) else None,
        sentence_end=int(valid.sentence.max()) if len(valid) else None,
        crosses_sentences=bool(valid.sentence.nunique() > 1),
        text=text[int(group.char_start.min()):int(group.char_end.max())])


def summarize_intervals(spans):
    rows = []
    for population, group in spans.groupby('population', sort=False):
        positive = population in ('gold_error', 'matched_error')
        alarms = int(group.alarm_tokens.sum())
        tokens = int(group.tokens.sum())
        rows.append(dict(population=population, spans=len(group), tokens=tokens, alarm_tokens=alarms,
            any_alarm=int(group.any_alarm.sum()), no_alarm=int(group.all_clear.sum()),
            alarm80=int(group.alarm80.sum()), all_alarm=int(group.all_alarm.sum()),
            onset_alarm=int(group.onset_alarm.sum()),
            token_recall=alarms / tokens if positive else None,
            token_fpr=alarms / tokens if not positive else None,
            success_definition='any_hit_is_partial_detection' if positive else 'no_false_alarm_in_entire_interval',
            success_count=int(group.any_alarm.sum() if positive else group.all_clear.sum())))
    return pd.DataFrame(rows)



def paired_detection_counts(spans):
    """Count BOTH sides correctly handled without turning token ranks into alarms."""
    columns = ['id', 'source_id', 'pair_error_start']
    error = spans[spans.population == 'matched_error']
    normal = spans[spans.population == 'matched_normal']
    pairs = error.merge(normal, on=columns, suffixes=('_error', '_normal'), validate='one_to_one')
    return dict(pairs=len(pairs), error_any_hit=int(pairs.any_alarm_error.sum()),
        error_80_coverage=int(pairs.alarm80_error.sum()), error_full_coverage=int(pairs.all_alarm_error.sum()),
        normal_completely_clear=int(pairs.all_clear_normal.sum()),
        both_error_any_and_normal_clear=int((pairs.any_alarm_error & pairs.all_clear_normal).sum()),
        both_error_80_and_normal_clear=int((pairs.alarm80_error & pairs.all_clear_normal).sum()),
        both_error_all_and_normal_clear=int((pairs.all_alarm_error & pairs.all_clear_normal).sum()),
        error_higher_mean=int((pairs.mean_score_error > pairs.mean_score_normal).sum()),
        tied_means=int((pairs.mean_score_error == pairs.mean_score_normal).sum()))


def position_counts(tokens):
    rows = []
    for population, frame in tokens.groupby('population'):
        for axis in ('interval_third', 'sentence_third', 'interval_offset'):
            for position, group in frame.groupby(axis):
                row = dict(population=population, axis=axis, position=int(position), tokens=len(group),
                    mean_score=float(group.score.mean()), intervals=group[['id', 'start']].drop_duplicates().shape[0])
                row.update({name: int((group.outcome == name).sum()) for name in ('TP', 'FN', 'FP', 'TN')})
                row['recall'] = row['TP'] / (row['TP'] + row['FN']) if row['TP'] + row['FN'] else None
                row['fpr'] = row['FP'] / (row['FP'] + row['TN']) if row['FP'] + row['TN'] else None
                rows.append(row)
    return pd.DataFrame(rows)


def sentence_span_grid(tokens):
    valid = tokens[tokens.sentence_third >= 0]
    result = valid.groupby(['population', 'interval_third', 'sentence_third']).agg(
        tokens=('token', 'size'), alarms=('predicted', 'sum')).reset_index()
    result['alarm_rate'] = result.alarms / result.tokens
    return result


def first_alarm_positions(spans):
    rows = []
    for population, group in spans.groupby('population'):
        rows.append(dict(population=population, category='no_alarm', spans=int(group.all_clear.sum())))
        for third in range(3):
            positions = ((group.first_alarm_offset + .5) * 3 / group.tokens)
            rows.append(dict(population=population, category=f'third_{third}',
                spans=int(((positions >= third) & (positions < third + 1)).sum())))
    return pd.DataFrame(rows)


def save_localization(table, annotations, samples, threshold, pairs, tier, output):
    from .visualize import localization_plots, token_gallery

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not np.array_equal(table.predicted, (table.score > threshold).astype(int)):
        raise ValueError('Alarms do not agree with the saved model threshold')
    table = annotate(table, annotations)
    table, sentences = locate_sentences(table, samples)
    intervals = diagnostic_intervals(table, pairs, tier)
    tokens = interval_tokens(table, intervals)
    keys = ['population', 'id', 'start']
    texts = {str(s['id']): str(s['response']) for s in samples}
    spans = pd.DataFrame([one_interval(g, texts[str(g.id.iloc[0])]) for _, g in tokens.groupby(keys, sort=False)])
    counts = summarize_intervals(spans)
    write_json(output / 'paired_detection_counts.json', paired_detection_counts(spans))
    positions = position_counts(tokens)
    grid = sentence_span_grid(tokens)
    tokens['length_group'] = pd.cut(tokens.end - tokens.start, [0, 4, 16, 32, np.inf], labels=['1-4', '5-16', '17-32', '33+'])
    lengths = []
    for name, group in tokens.groupby('length_group', observed=True):
        measured = position_counts(group)
        measured['length_group'] = str(name)
        lengths.extend(measured.to_dict('records'))
    for name, frame in (('span_counts', counts), ('spans', spans), ('sentences', sentences),
                        ('position_counts', positions), ('position_counts_by_length', pd.DataFrame(lengths)), ('span_sentence_grid', grid),
                        ('first_alarm_positions', first_alarm_positions(spans))):
        frame.to_csv(output / (name + '.csv'), index=False)
    table.to_csv(output / 'tokens.csv.gz', index=False)
    tokens.to_csv(output / 'interval_tokens.csv.gz', index=False)
    write_json(output / 'protocol.json', dict(threshold=threshold, pair_tier=tier,
        sentence_unit='punctuation_v1_original_offsets_not_semantic_claims',
        sentence_unmapped_tokens=int((table.sentence < 0).sum()),
        cross_sentence_tokens=int(table.crosses_sentence.sum()),
        matched_pairs=int((spans.population == 'matched_error').sum()),
        notes=['Populations overlap with each other; never add their denominators.',
               'Normal runs are not semantically annotated correct claims.',
               'Any hit does not turn the remainder of a gold span into true positives.',
               'Sentence -1 means nontext/whitespace or no sentence overlap; kept in all-token counts.']))
    localization_plots(counts, positions, grid, output / 'figures')
    token_gallery(samples, table, spans, sentences, output / 'gallery.html')
    print(counts.to_string(index=False), flush=True)


def run_localization(args, output, pairs):
    """Only original scores, text and offsets; embeddings are never loaded."""
    completed = 0
    for name in args.models:
        directory = Path(args.root) / name
        if not (directory / 'test' / 'predictions.json').exists():
            print('未发现完整预测：', name, flush=True)
            continue
        table, spans = read_tables(directory / 'test')
        samples = load_predictions(directory / 'test')
        if set(table.id) != {str(s['id']) for s in samples}:
            raise ValueError('Text records do not cover exactly the CSV response set')
        threshold = read_json(directory / 'threshold.json')['value']
        save_localization(table, spans, samples, threshold, pairs, args.pair_tier, output / name)
        completed += 1
    if not completed:
        raise ValueError('No complete predictions available for localization')
