"""Separate what the node model already detects from full-CHARM differences."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_json, read_tables, write_json
from .evaluate import analyze, compare_pairs
from .positions import annotate


IDENTITY = ['id', 'source_id', 'token', 'gold', 'text']


def aligned_models(root):
    """Use original model-specific calibration, never test-tune a common cutoff."""
    tables, spans, thresholds = {}, {}, {}
    for name in ('node_only', 'charm_in'):
        directory = Path(root) / name
        table, annotation = read_tables(directory / 'test')
        table = table.sort_values(['id', 'token']).reset_index(drop=True)
        threshold = read_json(directory / 'threshold.json')
        if not np.array_equal(table.predicted, (table.score > threshold['value']).astype(int)):
            raise ValueError(name + ': saved alarms differ from its threshold')
        tables[name] = annotate(table, annotation)
        spans[name], thresholds[name] = annotation, threshold
    node, full = tables['node_only'], tables['charm_in']
    if not node[IDENTITY].equals(full[IDENTITY]):
        raise ValueError('Node and full models have different token identities')
    if not node[['span_start', 'span_end']].equals(full[['span_start', 'span_end']]):
        raise ValueError('Node and full models have different annotation intervals')
    return tables, spans, thresholds


def comparison_tokens(tables):
    """Drop unrelated structural columns: this report is about model decisions."""
    node, full = tables['node_only'], tables['charm_in']
    columns = IDENTITY + ['span_index', 'span_start', 'span_end', 'span_length', 'offset', 'third']
    result = node[columns].copy()
    for short, frame in (('node', node), ('full', full)):
        result[short + '_score'] = frame.score
        result[short + '_alarm'] = frame.predicted.astype(bool)
        result[short + '_correct'] = frame.predicted.to_numpy() == frame.gold.to_numpy()
    correct_node, correct_full = result.node_correct, result.full_correct
    result['decision_group'] = np.select(
        [correct_node & correct_full, correct_node & ~correct_full, ~correct_node & correct_full],
        ['both_correct', 'node_only_correct', 'full_only_correct'], default='both_wrong')
    return result


def relative_regions(offset, length):
    """Overlapping diagnostic views; half and third counts must not be added."""
    offset, length = np.asarray(offset), np.asarray(length)
    center = (offset + .5) / length
    return {'all': np.ones(len(offset), bool), 'first': offset == 0,
            'front_half': center < .5, 'back_half': center >= .5,
            'early_third': center < 1/3,
            'middle_third': (center >= 1/3) & (center < 2/3),
            'late_third': center >= 2/3}


def decision_counts(table):
    rows = []
    for label, population in ((1, 'error'), (0, 'normal')):
        subset = table[table.gold == label]
        regions = {'all': np.ones(len(subset), bool)}
        if label:
            regions = relative_regions(subset.offset, subset.span_length)
        for region, mask in regions.items():
            chosen = subset[mask]
            for name in ('both_correct', 'node_only_correct', 'full_only_correct', 'both_wrong'):
                count = int((chosen.decision_group == name).sum())
                rows.append(dict(population=population, region=region, group=name,
                    tokens=count, denominator=len(chosen), fraction=count/len(chosen) if len(chosen) else None))
    return pd.DataFrame(rows)


def matched_tokens(table, pairs, tier):
    """A fixed relative-offset pairing; keep each model's decisions separate."""
    indexed = table.set_index(['id', 'token'])
    blocks, used = [], set()
    for pair in pairs:
        if pair['tier'] != tier:
            continue
        identity, length = str(pair['id']), int(pair['length'])
        error_positions = np.arange(pair['error_start'], pair['error_start'] + length)
        normal_positions = np.arange(pair['normal_start'], pair['normal_start'] + length)
        error = indexed.loc[[(identity, int(t)) for t in error_positions]].reset_index()
        normal = indexed.loc[[(identity, int(t)) for t in normal_positions]].reset_index()
        if not error.gold.all() or normal.gold.any():
            raise ValueError('Fixed pair labels disagree with original predictions')
        if set(error.source_id) != {str(pair['source_id'])}:
            raise ValueError('Fixed pair source does not match original predictions')
        coordinates = {(identity, int(t)) for t in np.r_[error_positions, normal_positions]}
        if used & coordinates:
            raise ValueError('Fixed pairs reuse a token in the requested tier')
        used.update(coordinates)
        error['pair_error_start'] = pair['error_start']
        error['pair_length'] = length
        error['pair_offset'] = np.arange(length)
        error['normal_token'] = normal.token.to_numpy()
        error['normal_text'] = normal.text.to_numpy()
        for name in ('node_score', 'full_score', 'node_alarm', 'full_alarm'):
            error['normal_' + name] = normal[name].to_numpy()
        blocks.append(error)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def pair_success(paired, thresholds):
    rows = []
    if paired.empty:
        return pd.DataFrame()
    for model in ('node', 'full'):
        outcomes = []
        for _, group in paired.groupby(['id', 'pair_error_start'], sort=False):
            error = group[model + '_alarm'].to_numpy()
            normal = group['normal_' + model + '_alarm'].to_numpy()
            outcomes.append((error.any(), error.mean() >= .8, error.all(), not normal.any()))
        flags = np.asarray(outcomes)
        rows.append(dict(model=model, pairs=len(flags), error_any=int(flags[:, 0].sum()),
            error80=int(flags[:, 1].sum()), error_all=int(flags[:, 2].sum()), normal_clear=int(flags[:, 3].sum()),
            both_any_and_clear=int((flags[:, 0] & flags[:, 3]).sum()),
            both80_and_clear=int((flags[:, 1] & flags[:, 3]).sum()),
            threshold=thresholds['node_only' if model == 'node' else 'charm_in']['value']))
    return pd.DataFrame(rows)


def summarize_positions(tables):
    """Recall and FPR keep separate denominators. Matched normal positions are elsewhere."""
    rows = []
    for model, table in tables.items():
        errors = table[table.gold == 1]
        for region, mask in relative_regions(errors.offset, errors.span_length).items():
            chosen = errors[mask]
            rows.append(dict(model=model, region=region, tokens=len(chosen), hits=int(chosen.predicted.sum()),
                recall=float(chosen.predicted.mean()) if len(chosen) else None))
    return pd.DataFrame(rows)


def matched_positions(paired):
    rows = []
    if paired.empty:
        return pd.DataFrame()
    masks = relative_regions(paired.pair_offset, paired.pair_length)
    for model in ('node', 'full'):
        for region, mask in masks.items():
            group = paired[mask]
            rows.append(dict(model=model, region=region, tokens=len(group),
                error_score=group[model+'_score'].mean(), normal_score=group['normal_'+model+'_score'].mean(),
                recall=group[model+'_alarm'].mean(), fpr=group['normal_'+model+'_alarm'].mean()))
    return pd.DataFrame(rows)


def run_compare(args, output, pairs):
    from .node_plots import comparison_plots

    tables, spans, thresholds = aligned_models(args.root)
    table = comparison_tokens(tables)
    paired = matched_tokens(table, pairs, args.pair_tier)
    metrics, pair_results = [], {}
    selected = [p for p in pairs if p['tier'] == args.pair_tier]
    for name, frame in tables.items():
        scores, pair_results[name] = analyze(frame, spans[name], thresholds[name]['value'],
                                            selected, output/name, args.bootstrap)
        metrics.append(dict(model=name, **scores))
    compare_pairs(pair_results, output, args.bootstrap, reference='node_only')
    counts = decision_counts(table)
    positions = summarize_positions(tables)
    matched = matched_positions(paired)
    for name, frame in (('models', pd.DataFrame(metrics)), ('decision_counts', counts),
                        ('positions', positions), ('matched_positions', matched),
                        ('pair_success', pair_success(paired, thresholds))):
        frame.to_csv(output/(name+'.csv'), index=False)
    table.to_csv(output/'comparison_tokens.csv.gz', index=False)
    paired.to_csv(output/'paired_tokens.csv.gz', index=False)
    write_json(output/'protocol.json', dict(thresholds=thresholds, pair_tier=args.pair_tier,
        interpretation='Two separately trained models; decision changes are NOT causal message contributions.',
        calibration='Own original model thresholds; equal target calibration FPR does not ensure equal test FPR.'))
    comparison_plots(positions, counts, matched, output/'figures')
    print(pd.DataFrame(metrics)[['model', 'auroc', 'ap', 'recall', 'fpr']].to_string(index=False))
    return table, paired
