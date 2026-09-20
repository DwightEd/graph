"""Incidence denominators and source-balanced comparisons, not detector metrics."""

import json

import numpy as np
import pandas as pd


def incidence(group):
    eligible = group[group.status.eq('measured')]
    paired = eligible[eligible.normal_event.notna()]
    source_difference = (paired.assign(difference=paired.event.astype(float)
                         - paired.normal_event.astype(float))
                         .groupby('source_id').difference.mean())
    interval = [np.nan, np.nan]
    if len(source_difference) >= 2:
        rng = np.random.default_rng(17)
        draws = rng.choice(source_difference.to_numpy(), (1000, len(source_difference)))
        interval = np.quantile(draws.mean(axis=1), [.025, .975])
    events = int(eligible.event.astype(bool).sum())
    return dict(annotated_onsets=len(group), measured=len(eligible), events=events,
                absent=len(eligible)-events, unavailable=len(group)-len(eligible),
                event_fraction=float(events/len(eligible)) if len(eligible) else np.nan,
                normal_fraction=paired.normal_event.astype(float).mean(),
                paired_onsets=len(paired), paired_sources=len(source_difference),
                source_mean_difference=source_difference.mean(), low=interval[0], high=interval[1])


def summarize_heads(output):
    paths = sorted((output / 'samples').glob('*/heads.csv.gz'))
    keys = ['split', 'task', 'generator', 'phase', 'layer', 'head']
    totals = {}
    for path in paths:
        table = pd.read_csv(path)
        table['measured'] = table.status.eq('measured')
        table['event'] = table.event.eq(True) & table.measured
        table['normal_measured'] = table.normal_event.notna() & table.measured
        table['normal_event'] = table.normal_event.eq(True) & table.normal_measured
        populations = dict(all=np.ones(len(table), bool), clean_before=~table.contaminated_before,
                           answer_first=table.answer_first)
        for population, mask in populations.items():
            summary = table[mask].groupby(keys).agg(
                annotated_onsets=('onset', 'size'), measured=('measured', 'sum'),
                events=('event', 'sum'), normal_measured=('normal_measured', 'sum'),
                normal_events=('normal_event', 'sum'))
            totals[population] = totals[population].add(summary, fill_value=0) if population in totals else summary
    frames = [frame.reset_index().assign(population=population) for population, frame in totals.items()]
    pd.concat(frames).to_csv(output / 'head_incidence.csv', index=False)


def summarize_onsets(output):
    paths = sorted((output / 'samples').glob('*/onsets.csv'))
    table = pd.concat([pd.read_csv(path, dtype={'source_id': str}) for path in paths], ignore_index=True)
    table.to_csv(output / 'onsets.csv', index=False)
    rows = []
    for population, mask in dict(all=np.ones(len(table), bool), clean_before=~table.contaminated_before,
                                  answer_first=table.answer_first, run_onsets=table.run_onset).items():
        keys = ['split', 'task', 'generator', 'phase']
        for identity, group in table[mask].groupby(keys):
            rows.append(dict(zip(keys, identity), population=population, **incidence(group)))
    pd.DataFrame(rows).to_csv(output / 'incidence.csv', index=False)
    population = pd.read_csv(output / 'population_coverage.csv')
    summary = dict(official_answers=len(population), cached_answers=int(population.cached.sum()),
                   missing_answers=int((~population.cached).sum()),
                   answers_with_onsets=len(paths), annotated_onsets=int(len(table)/2),
                   purpose='label_assisted_pre_onset_audit_not_detector_evaluation',
                   statistic='max_across_all_requested_heads_and_observed_steps',
                   special_tokens='excluded_using_observer_tokenizer_all_special_ids',
                   caution='normal reference windows may overlap; percentile is not a p-value')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    summarize_heads(output)
    return summary
