"""Reanalyse every annotated onset with frozen scores and original thresholds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annotation_population import read_jsonl
from .unit_data import read_tokens


def window_scores(answer, start, length, window):
    before = answer.loc[max(0, start-window):start-1, 'score'].to_numpy() if start else np.array([])
    inside = answer.loc[start:start+length-1, 'score'].to_numpy()
    return before, inside


def describe(before, inside, threshold, window):
    alarm = np.flatnonzero(inside > threshold)
    complete = len(before) == window and np.isfinite(before).all()
    return dict(before_available=complete,
                before_alarm=bool(np.any(before > threshold)) if complete else None,
                before_mean=float(before.mean()) if complete else np.nan,
                onset_score=float(inside[0]), onset_alarm=bool(inside[0] > threshold),
                first4_available=len(inside) >= 4,
                first4_alarm=bool(np.any(inside[:4] > threshold)) if len(inside) >= 4 else None,
                first4_sustained=bool(np.sum(inside[:4] > threshold) >= 3) if len(inside) >= 4 else None,
                first_alarm_offset=int(alarm[0]) if len(alarm) else -1,
                span_alarm=bool(len(alarm)))


def measure_onsets(table, threshold, window):
    rows = []
    for identity, answer in table.groupby('id', sort=False):
        answer = answer.set_index('token').sort_index()
        for start, token in answer[answer.offset.eq(0)].iterrows():
            length = int(token.span_length)
            before, inside = window_scores(answer, start, length, window)
            rows.append(dict(id=identity, source_id=token.source_id, onset=start,
                length=length, answer_first=token.span_index == 0,
                run_onset=start == 0 or answer.loc[start-1, 'gold'] == 0,
                clean_before=not bool(answer.loc[max(0, start-window):start-1, 'gold'].any())
                if start else True,
                **describe(before, inside, threshold, window)))
    return pd.DataFrame(rows)


def summarize_events(table):
    rows = []
    for population, mask in dict(all=np.ones(len(table), bool), clean_before=table.clean_before,
                                  answer_first=table.answer_first, run_onsets=table.run_onset).items():
        group = table[mask]
        for phase in ('before', 'onset', 'first4', 'span'):
            valid = group[f'{phase}_alarm'].notna()
            values = group.loc[valid, f'{phase}_alarm'].astype(bool)
            rows.append(dict(population=population, phase=phase, total=len(group),
                measured=int(valid.sum()), alarms=int(values.sum()),
                fraction=float(values.mean()) if len(values) else np.nan))
    return pd.DataFrame(rows)


def measure_pairs(table, pairs, threshold, window):
    answers = {identity: group.set_index('token').sort_index() for identity, group in table.groupby('id')}
    rows = []
    for pair in pairs:
        if pair['tier'] != 'cluster':
            continue
        answer = answers[str(pair['id'])]
        length = pair['length']
        for side, key in (('error', 'error_start'), ('normal', 'normal_start')):
            start = pair[key]
            before, inside = window_scores(answer, start, length, window)
            for phase, values, available in (
                ('before8', before, len(before) == window), ('onset', inside[:1], True),
                ('first4', inside[:4], length >= 4), ('span', inside, True)):
                if available:
                    rows.append(dict(id=str(pair['id']), source_id=str(pair['source_id']),
                        error_start=pair['error_start'], normal_start=pair['normal_start'],
                        side=side, phase=phase, mean=float(values.mean()),
                        alarm=bool(np.any(values > threshold))))
    return pd.DataFrame(rows)


def summarize_pairs(frame):
    keys = ['id', 'source_id', 'error_start', 'normal_start', 'phase']
    paired = frame.pivot(index=keys, columns='side', values=['mean', 'alarm']).dropna()
    paired.columns = ['_'.join(column) for column in paired.columns]
    paired = paired.reset_index()
    paired['win'] = (paired.mean_error > paired.mean_normal).astype(float)
    paired['win'] += .5 * (paired.mean_error == paired.mean_normal)
    rows = []
    for phase, group in paired.groupby('phase'):
        group = group.assign(alarm_difference=group.alarm_error.astype(float)-group.alarm_normal.astype(float))
        sources = group.groupby('source_id')[['win', 'alarm_difference']].mean()
        rng = np.random.default_rng(17)
        draws = rng.integers(0, len(sources), size=(2000, len(sources)))
        low, high = np.quantile(sources.to_numpy()[draws].mean(axis=1), [.025, .975], axis=0)
        rows.append(dict(phase=phase, pairs=len(group), sources=group.source_id.nunique(),
            error_alarms=int(group.alarm_error.sum()), normal_alarms=int(group.alarm_normal.sum()),
            paired_win=group.win.mean(), source_mean_win=sources.win.mean(),
            win_low=low[0], win_high=high[0], source_alarm_difference=sources.alarm_difference.mean(),
            alarm_difference_low=low[1], alarm_difference_high=high[1]))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('charm-tokens', 'lda-tokens', 'annotations', 'pairs', 'charm-threshold', 'lda-thresholds', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--window', type=int, default=8)
    args = parser.parse_args(argv)
    records = {str(row['id']): row for row in read_jsonl(args.annotations)}
    pairs = json.loads(args.pairs.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    settings = dict(window=args.window, purpose='frozen_score_audit_not_attention_reanchor_measurement',
                    thresholds='original independent normal calibration; no test retuning',
                    note='a pre-onset score is not an online warning without prefix-causality verification')
    settings['inputs'] = {key: str(value.resolve()) for key, value in vars(args).items() if isinstance(value, Path) and key != 'output'}
    (args.output / 'protocol.json').write_text(json.dumps(settings, indent=2) + '\n')
    thresholds = dict(charm=json.loads(args.charm_threshold.read_text())['value'],
                      lda=json.loads(args.lda_thresholds.read_text())['full'])
    for model, path, score in (('charm', args.charm_tokens, 'score'), ('lda', args.lda_tokens, 'full')):
        table = read_tokens(path, [score], records).rename(columns={score: 'score'})
        onsets = measure_onsets(table, thresholds[model], args.window)
        onsets.to_csv(args.output / f'{model}_onsets.csv', index=False)
        summarize_events(onsets).to_csv(args.output / f'{model}_incidence.csv', index=False)
        paired = measure_pairs(table, pairs, thresholds[model], args.window)
        paired.to_csv(args.output / f'{model}_pairs.csv', index=False)
        summarize_pairs(paired).to_csv(args.output / f'{model}_pair_summary.csv', index=False)


if __name__ == '__main__':
    main()
