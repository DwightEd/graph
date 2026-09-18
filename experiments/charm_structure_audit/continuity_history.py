"""Test whether past scores substitute for current evidence, using saved scores."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import read_json, read_tables, write_json
from .evaluate import source_interval
from .history_math import SCORE_NAMES, score_table
from .positions import annotate


def ranking(labels, scores):
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    positive = int(labels.sum())
    negative = len(labels) - positive
    return dict(tokens=len(labels), positives=positive, negatives=negative,
                auroc=float(roc_auc_score(labels, scores)) if positive and negative else None,
                ap=float(average_precision_score(labels, scores)) if positive else None)


def populations(table):
    labels = table.gold.astype(bool)
    onset = labels & table.offset.eq(0)
    first = onset & table.span_index.eq(0)
    return dict(all=np.ones(len(table), bool), first_or_normal=first | ~labels,
                onset_or_normal=onset | ~labels, continuation_or_normal=(labels & ~onset) | ~labels,
                previous_normal=table.previous_gold.eq(0), previous_error=table.previous_gold.eq(1))


def summarize_rankings(table):
    """Every score uses exactly the same finite token set for each population."""
    covered = np.isfinite(table[list(SCORE_NAMES)]).all(axis=1)
    rows = []
    for population, mask in populations(table).items():
        selected = table.loc[covered & mask]
        for name in SCORE_NAMES:
            rows.append(dict(method=name, population=population, eligible=int(np.sum(mask)),
                             sources=selected.source_id.nunique(), **ranking(selected.gold, selected[name])))
    return pd.DataFrame(rows)


def conditional_rankings(table, bootstrap):
    """Gold-history strata are diagnostics, not inputs. Disclose mixed-cell support."""
    covered = np.isfinite(table[list(SCORE_NAMES)]).all(axis=1)
    valid = table.loc[covered].copy()
    valid['past_bin'] = np.minimum(9, (valid.past_mean * 10).astype(int))
    rows, cells = [], []
    for condition, columns in (('answer', ['id']), ('previous_label', ['id', 'previous_gold']),
                               ('past_score', ['id', 'past_bin'])):
        for key, group in valid.groupby(columns, sort=False):
            if group.gold.nunique() < 2:
                continue
            for name in SCORE_NAMES:
                cells.append(dict(condition=condition, cell=str(key), method=name,
                                  id=str(group.id.iloc[0]), source_id=str(group.source_id.iloc[0]),
                                  **ranking(group.gold, group[name])))
    frame = pd.DataFrame(cells)
    if frame.empty:
        return frame, pd.DataFrame(rows)
    for (condition, name), group in frame.groupby(['condition', 'method']):
        group = group.assign(pair_count=group.positives * group.negatives)
        group['numerator'] = group.auroc * group.pair_count
        answers = group.groupby(['id', 'source_id'])[['numerator', 'pair_count']].sum().reset_index()
        answers['auroc'] = answers.numerator / answers.pair_count
        rows.append(dict(condition=condition, method=name, cells=len(group), tokens=int(group.tokens.sum()),
                         **source_interval(answers, 'auroc', bootstrap)))
    return frame, pd.DataFrame(rows)



def conditional_deltas(cells, bootstrap):
    """Paired source differences on identical mixed cells, not independent CIs."""
    if cells.empty:
        return pd.DataFrame()
    counted = cells.assign(pair_count=cells.positives * cells.negatives)
    counted['numerator'] = counted.auroc * counted.pair_count
    keys = ['condition', 'method', 'id', 'source_id']
    answers = counted.groupby(keys)[['numerator', 'pair_count']].sum().reset_index()
    answers['auroc'] = answers.numerator / answers.pair_count
    identities = ['condition', 'id', 'source_id']
    current = answers[answers.method == 'current'][identities + ['auroc']]
    compared = answers.merge(current, on=identities, suffixes=('', '_current'), validate='many_to_one')
    compared['change'] = compared.auroc - compared.auroc_current
    rows = []
    for (condition, method), group in compared.groupby(['condition', 'method']):
        rows.append(dict(condition=condition, method=method, answers=len(group),
                         **source_interval(group, 'change', bootstrap)))
    return pd.DataFrame(rows)


def pair_measurements(table, pairs):
    """Lock both partners; never select them by detection success or score."""
    answers = {str(key): group.set_index('token') for key, group in table.groupby('id')}
    rows = []
    for pair in pairs:
        if pair['tier'] != 'cluster':
            continue
        group = answers[str(pair['id'])]
        offsets = np.arange(pair['length'])
        error = group.loc[pair['error_start'] + offsets]
        normal = group.loc[pair['normal_start'] + offsets]
        if not error.gold.eq(1).all() or not normal.gold.eq(0).all():
            raise ValueError('Locked pair does not match saved labels')
        if not error.source_id.eq(str(pair['source_id'])).all():
            raise ValueError('Locked pair source differs from saved scores')
        common = np.isfinite(error[list(SCORE_NAMES)]).all(axis=1).to_numpy()
        common &= np.isfinite(normal[list(SCORE_NAMES)]).all(axis=1).to_numpy()
        fractions = (offsets + .5) / pair['length']
        regions = dict(all=np.ones(len(offsets), bool), front=fractions < .5, back=fractions >= .5)
        for region, mask in regions.items():
            selected = mask & common
            if not selected.any():
                continue
            rows.extend(measure_pair_scores(error, normal, selected, pair, region))
    return pd.DataFrame(rows)


def measure_pair_scores(error, normal, mask, pair, region):
    rows = []
    for name in SCORE_NAMES:
        a = error[name].to_numpy()[mask]
        b = normal[name].to_numpy()[mask]
        measured = ranking(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
        rows.append(dict(method=name, region=region, id=str(pair['id']), source_id=str(pair['source_id']),
                         error_start=pair['error_start'], normal_start=pair['normal_start'],
                         length=pair['length'], tokens_each=len(a), error_mean=a.mean(),
                         normal_mean=b.mean(), gap=(a - b).mean(), auroc=measured['auroc']))
    return rows


def pair_summaries(frame, bootstrap):
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    keys = ['id', 'source_id', 'error_start', 'normal_start', 'region']
    current = frame[frame.method == 'current'][keys + ['auroc', 'gap']]
    compared = frame.merge(current, on=keys, suffixes=('', '_current'), validate='many_to_one')
    compared['auc_change'] = compared.auroc - compared.auroc_current
    compared['gap_change'] = compared.gap - compared.gap_current
    rows = []
    for (method, region), group in compared.groupby(['method', 'region']):
        for field in ('auroc', 'gap', 'auc_change', 'gap_change'):
            rows.append(dict(method=method, region=region, measure=field, pairs=len(group),
                             **source_interval(group, field, bootstrap)))
    return pd.DataFrame(rows), compared, growth_summary(frame, bootstrap)


def growth_summary(frame, bootstrap):
    keys = ['method', 'id', 'source_id', 'error_start', 'normal_start']
    front = frame[frame.region == 'front']
    back = frame[frame.region == 'back']
    both = front.merge(back, on=keys, suffixes=('_front', '_back'), validate='one_to_one')
    rows = []
    for method, group in both.groupby('method'):
        group = group.copy()
        for field in ('error_mean', 'normal_mean', 'gap', 'auroc'):
            group['change'] = group[field + '_back'] - group[field + '_front']
            rows.append(dict(method=method, measure=field + '_growth', pairs=len(group),
                             **source_interval(group, 'change', bootstrap)))
    return pd.DataFrame(rows)


def calibrated_thresholds(calibration, original_threshold, window, beta, seed, fpr):
    thresholds = dict.fromkeys(SCORE_NAMES)
    thresholds['current'] = original_threshold
    if calibration is None:
        return thresholds
    control = score_table(calibration, window, beta, seed)
    normal = control.gold.eq(0) & control.text.str.len().gt(0)
    for name in SCORE_NAMES:
        if name == 'current':
            continue
        values = control.loc[normal & np.isfinite(control[name]), name]
        if len(values):
            thresholds[name] = float(np.quantile(values, 1 - fpr, method='higher'))
    return thresholds


def transition_scores(table, thresholds):
    covered = np.isfinite(table[list(SCORE_NAMES)]).all(axis=1)
    rows = []
    for (previous, current), group in table.loc[covered].groupby(['previous_gold', 'gold']):
        for name in SCORE_NAMES:
            threshold = thresholds[name]
            rate = float((group[name] > threshold).mean()) if threshold is not None else None
            rows.append(dict(method=name, previous_gold=int(previous), current_gold=int(current),
                             tokens=len(group), sources=group.source_id.nunique(),
                             mean_score=float(group[name].mean()), threshold=threshold, alarm_rate=rate))
    return pd.DataFrame(rows)


def exit_scores(table, thresholds, window):
    """Observe true error-run exits; the predictor itself is never reset there."""
    rows = []
    for identity, group in table.groupby('id', sort=False):
        labels = group.gold.to_numpy().astype(bool)
        exits = np.flatnonzero(labels[:-1] & ~labels[1:]) + 1
        for end in exits:
            for offset in range(window):
                token = end + offset
                if token >= len(group) or labels[token]:
                    break
                record = group.iloc[token]
                for name in SCORE_NAMES:
                    value, threshold = record[name], thresholds[name]
                    alarm = float(value > threshold) if threshold is not None else np.nan
                    rows.append(dict(id=str(identity), source_id=str(record.source_id), exit_token=int(end),
                                     offset=offset, method=name, score=value, alarm=alarm))
    return pd.DataFrame(rows)


def summarize_exits(frame, bootstrap):
    rows = []
    if frame.empty:
        return pd.DataFrame(rows)
    for (name, offset), group in frame.groupby(['method', 'offset']):
        for field in ('score', 'alarm'):
            valid = group.dropna(subset=[field])
            rows.append(dict(method=name, offset=int(offset), measure=field, tokens=len(valid),
                             **source_interval(valid, field, bootstrap)))
    return pd.DataFrame(rows)


def analyze_history(table, spans, threshold, pairs, output, bootstrap=200,
                    window=10, beta=.5, seed=17, calibration=None, fpr=.05):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values(['id', 'token']).reset_index(drop=True)
    ordered = annotate(ordered, spans)
    controls = score_table(ordered, window, beta, seed)
    controls['previous_gold'] = controls.groupby('id', sort=False).gold.shift(1)
    thresholds = calibrated_thresholds(calibration, threshold, window, beta, seed, fpr)
    pair_rows = pair_measurements(controls, pairs)
    pair_summary, changes, growth = pair_summaries(pair_rows, bootstrap)
    cells, conditional = conditional_rankings(controls, bootstrap)
    exits = exit_scores(controls, thresholds, window)
    outputs = dict(metrics=summarize_rankings(controls), conditional_cells=cells,
                   conditional_ranking=conditional, conditional_deltas=conditional_deltas(cells, bootstrap),
                   matched_pairs=pair_rows, matched_summary=pair_summary,
                   paired_changes=changes, growth_summary=growth, exits=exits,
                   exit_summary=summarize_exits(exits, bootstrap), transitions=transition_scores(controls, thresholds))
    for name, frame in outputs.items():
        frame.to_csv(output / (name + '.csv'), index=False)
    controls.to_csv(output / 'tokens.csv.gz', index=False)
    write_json(output / 'protocol.json', dict(window=window, beta=beta, seed=seed, probability_domain=True,
        labels_enter_scores=False, common_coverage='all methods; no history at first token', thresholds=thresholds,
        calibration_available=calibration is not None, refit=False, llm_forward=False,
        limitations=['Past detector scores still contain supervised representation information.',
                     'Conditional ranks and fixed-pair growth are observational, not native-LLM causality.',
                     'Missing calibration means no transformed-score alarm claim.',
                     'Source intervals are exploratory; AUROC differences are not causal percentages.']))
    return outputs


def read_calibration(directory, test):
    path = directory / 'calibration' / 'tokens.csv'
    if not path.is_file():
        return None
    table = pd.read_csv(path, dtype={'id': str, 'source_id': str}, keep_default_na=False)
    recipe = read_json(directory / 'training.json')
    expected = set(map(str, recipe['partitions']['calibration']))
    if set(table.id) != expected or set(table.source_id) & set(test.source_id):
        raise ValueError('Calibration scores do not match the original held-out source partition')
    return table


def run_saved_history(root, models, output, pairs, bootstrap=200, window=10, beta=.5):
    done = []
    for name in models:
        directory = Path(root) / name
        if not (directory / 'test/tokens.csv').is_file():
            print('No completed saved scores:', name, flush=True)
            continue
        table, spans = read_tables(directory / 'test')
        threshold = float(read_json(directory / 'threshold.json')['value'])
        calibration = read_calibration(directory, table)
        reports = analyze_history(table, spans, threshold, pairs, Path(output) / name,
                                  bootstrap, window, beta, calibration=calibration)
        print(name, '\n', reports['metrics'].query('population == "all"').to_string(index=False), flush=True)
        done.append(name)
    if not done:
        raise ValueError('No completed models found under the requested root')
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/charm_structure_audit_qa/QA/seed_0'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--models', nargs='+', default=['node_only', 'charm_in'])
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args(argv)
    if args.window < 1 or not 0 <= args.beta < 1 or args.bootstrap < 0:
        raise ValueError('window must be positive, beta in [0,1), bootstrap nonnegative')
    output = args.output or args.root / 'audit_history_v2'
    pair_path = args.pairs or args.root / 'charm_in/test/cluster_audit/pairs.json'
    pairs = read_json(pair_path) if pair_path.is_file() else []
    from .data import prepare_output
    prepare_output(output, dict(root=str(args.root.resolve()), models=args.models,
                               window=args.window, beta=args.beta, pairs=str(pair_path.resolve()), version=2))
    run_saved_history(args.root, args.models, output, pairs, args.bootstrap, args.window, args.beta)


if __name__ == '__main__':
    main()
