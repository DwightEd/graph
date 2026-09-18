"""Saved-score continuity and loss interventions are different experiments."""

from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from .data import read_json, read_tables, write_json
from .positions import annotate
from .evaluate import metrics, source_interval


def transition_tables(table, threshold, bootstrap):
    """Previous GOLD label is an oracle diagnostic, never a model input."""
    table = table.sort_values(['id', 'token']).copy()
    table['previous_gold'] = table.groupby('id').gold.shift(1)
    valid = table.dropna(subset=['previous_gold'])
    counts = []
    for (previous, current), group in valid.groupby(['previous_gold', 'gold']):
        counts.append(dict(previous_gold=int(previous), current_gold=int(current),
                          tokens=len(group), answers=group.id.nunique(), sources=group.source_id.nunique(),
                          mean_score=float(group.score.mean()), alarms=int((group.score > threshold).sum()),
                          alarm_rate=float((group.score > threshold).mean())))
    conditional, answers = conditional_answers(valid, threshold, bootstrap)
    oracle = metrics(valid.gold, valid.previous_gold, .5)
    oracle['meaning'] = 'previous GOLD label: unavailable to detector; not a deployable baseline'
    return pd.DataFrame(counts), conditional, answers, oracle


def conditional_answers(table, threshold, bootstrap):
    conditional, answers = [], []
    for previous, group in table.groupby('previous_gold'):
        rows = []
        for identity, answer in group.groupby('id'):
            if answer.gold.nunique() == 2:
                rows.append(dict(id=identity, source_id=str(answer.source_id.iloc[0]),
                                 previous_gold=int(previous), auroc=float(roc_auc_score(answer.gold, answer.score)),
                                 error_tokens=int(answer.gold.sum()), normal_tokens=int((answer.gold == 0).sum())))
        columns = ['id', 'source_id', 'previous_gold', 'auroc', 'error_tokens', 'normal_tokens']
        frame = pd.DataFrame(rows, columns=columns)
        interval = source_interval(frame, 'auroc', bootstrap)
        conditional.append(dict(previous_gold=int(previous), mixed_answers=len(frame),
                                **metrics(group.gold, group.score, threshold),
                                **{'within_source_' + key: value for key, value in interval.items()}))
        answers.extend(rows)
    return pd.DataFrame(conditional), pd.DataFrame(answers)


def half_measurements(error, normal, length):
    """Original probability and clipped-logit displays; no score transformation here."""
    indices = np.arange(length)
    result = {}
    for name, mask in (('front', (indices + .5) / length < .5), ('back', (indices + .5) / length >= .5)):
        if not mask.any():
            break
        a, b = error.score.to_numpy()[mask], normal.score.to_numpy()[mask]
        result[name + '_error_score'] = float(a.mean())
        result[name + '_normal_score'] = float(b.mean())
        result[name + '_gap'] = float((logit(np.clip(a, 1e-7, 1 - 1e-7)) -
                                       logit(np.clip(b, 1e-7, 1 - 1e-7))).mean())
        result[name + '_auc'] = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
    if 'front_auc' in result and 'back_auc' in result:
        result['gap_growth'] = result['back_gap'] - result['front_gap']
        result['auc_growth'] = result['back_auc'] - result['front_auc']
    return result


def growth_rows(table, pairs):
    """Same locked windows: late growth on error side minus normal-side growth."""
    groups = {key: value.set_index('token') for key, value in table.groupby('id')}
    rows = []
    for pair in pairs:
        if pair['tier'] != 'cluster':
            continue
        group = groups[str(pair['id'])]
        length = int(pair['length'])
        indices = np.arange(length)
        error = group.loc[pair['error_start'] + indices]
        normal = group.loc[pair['normal_start'] + indices]
        if not error.gold.all() or normal.gold.any() or str(error.source_id.iloc[0]) != str(pair['source_id']):
            raise ValueError('Fixed pair labels/source do not match this model')
        measured = half_measurements(error, normal, length)
        if 'gap_growth' in measured:
            rows.append(dict(id=str(pair['id']), source_id=str(pair['source_id']),
                             error_start=pair['error_start'], length=length, **measured))
    return pd.DataFrame(rows)


def interval_population(table):
    rows = []
    errors = table[table.gold == 1]
    for (identity, start), group in errors.groupby(['id', 'span_start']):
        rows.append(dict(id=identity, source_id=str(group.source_id.iloc[0]), start=int(start),
                         length=len(group), end=int(group.span_end.iloc[0]), mean_score=float(group.score.mean())))
    spans = pd.DataFrame(rows, columns=['id', 'source_id', 'start', 'length', 'end', 'mean_score'])
    positive = len(errors)
    lengths = spans.length.to_numpy()
    onset_fraction = len(spans) / positive if positive else None
    return spans, dict(tokens=len(table), error_tokens=positive, mapped_spans=len(spans),
                       onset_fraction_positive=onset_fraction,
                       continuation_fraction_positive=1 - onset_fraction if positive else None,
                       span_weight_effective_count=float(lengths.sum() ** 2 / np.square(lengths).sum()) if positive else None,
                       note='Effective count describes unequal span weights only; not a gradient ESS or training estimate.')


def save_analysis(table, annotations, threshold, pairs, output, bootstrap):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = annotate(table.sort_values(['id', 'token']).reset_index(drop=True), annotations)
    counts, conditional, answers, oracle = transition_tables(table, threshold, bootstrap)
    growth = growth_rows(table, pairs)
    spans, population = interval_population(table)
    summary = []
    for field in ('front_auc', 'back_auc', 'front_gap', 'back_gap', 'gap_growth', 'auc_growth'):
        if not growth.empty:
            summary.append(dict(measure=field, pairs=len(growth), **source_interval(growth, field, bootstrap)))
    frames = dict(transitions=counts, conditional_ranking=conditional, conditional_answers=answers,
                  paired_growth=growth, growth_summary=pd.DataFrame(summary), span_population=spans)
    for name, frame in frames.items():
        frame.to_csv(output / (name + '.csv'), index=False)
    write_json(output / 'population.json', population)
    write_json(output / 'oracle_previous_label.json', oracle)
    write_json(output / 'overall.json', metrics(table.gold, table.score, threshold))
    return counts, conditional, growth, pd.DataFrame(summary)


def plots(counts, growth, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f'{r.previous_gold} -> {r.current_gold}\nn={r.tokens}' for r in counts.itertuples()]
    ax.bar(labels, counts.mean_score)
    ax.set(ylabel='Mean original score', xlabel='Previous gold label -> current gold label', ylim=(0, 1),
           title='Observed transitions; not a randomized context intervention')
    fig.tight_layout()
    fig.savefig(output / 'transitions.png', dpi=160)
    plt.close(fig)
    if not growth.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for role in ('error', 'normal'):
            values = growth.groupby('source_id')[[f'front_{role}_score', f'back_{role}_score']].mean().mean()
            ax.plot(['Front half', 'Back half'], values, marker='o', label=role)
        ax.set(ylabel='Equal-source mean score', ylim=(0, 1), title='Same matched windows: score growth')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'matched_growth.png', dpi=160)
        plt.close(fig)


def observe_model(directory, table, spans, threshold, pairs, output, bootstrap):
    from .continuity_history import analyze_history, read_calibration

    counts, ranking, growth, _ = save_analysis(table, spans, threshold, pairs, output, bootstrap)
    plots(counts, growth, output)
    calibration = read_calibration(directory, table)
    analyze_history(table, spans, threshold, pairs, output / 'history_controls', bootstrap,
                    calibration=calibration)
    return counts, ranking


def run_observe(args, output, pairs):
    reports = []
    identity = None
    for name in args.models:
        directory = Path(args.root) / name
        if not (directory / 'test/tokens.csv').exists():
            print('No saved token table:', name, flush=True)
            continue
        table, spans = read_tables(directory / 'test')
        current = table.sort_values(['id', 'token'])[['id', 'source_id', 'token', 'gold', 'text']].reset_index(drop=True)
        if identity is not None and not identity.equals(current):
            raise ValueError('Compared models must have identical token coordinates and labels')
        identity = current
        threshold = float(read_json(directory / 'threshold.json')['value'])
        counts, ranking = observe_model(directory, table, spans, threshold, pairs, output / name, args.bootstrap)
        reports.append(dict(model=name, counts=counts.to_dict('records'), conditional=ranking.to_dict('records')))
        print(name, '\n', ranking.to_string(index=False), flush=True)
    if not reports:
        raise ValueError('No completed model tables found')
    write_json(output / 'protocol.json', dict(stage='observational_saved_scores_v2', models=[r['model'] for r in reports],
        labels_enter_predictions=False, refit=False, paired_rule='existing cluster pairs only',
        limitations=['Transitions are observational and composition-confounded, not do(history).',
                     'Source bootstrap is exploratory; repeated test analysis is not independent confirmation.',
                     'Previous gold label is an explicitly labelled oracle, not a model baseline.',
                     'Observed score persistence cannot identify its training or native-LLM cause.']))


def run_continuity(args, output, pairs):
    if args.continuity_stage == 'train':
        from .continuity_train import run_training
        run_training(args, output, pairs)
    else:
        run_observe(args, output, pairs)
    with tarfile.open(output / 'continuity_review.tar.gz', 'w:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.csv', '.json', '.md', '.png'):
                archive.add(path, arcname=path.relative_to(output))
