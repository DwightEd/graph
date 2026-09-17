"""Same token coordinates, fixed thresholds and fixed matched pairs."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .data import write_json
from .positions import annotate, summarize


def metrics(labels, scores, threshold):
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    alarm = scores > threshold
    positive, negative = int(labels.sum()), int((~labels).sum())
    tp, fp = int((labels & alarm).sum()), int((~labels & alarm).sum())
    return dict(tokens=len(labels), positives=positive, negatives=negative,
        auroc=float(roc_auc_score(labels, scores)) if positive and negative else None,
        ap=float(average_precision_score(labels, scores)) if positive else None,
        tp=tp, fp=fp, fn=positive-tp, tn=negative-fp,
        recall=tp/positive if positive else None, fpr=fp/negative if negative else None)


def answer_metrics(table, threshold):
    rows = []
    for identity, group in table.groupby('id', sort=False):
        rows.append(dict(id=identity, source_id=str(group.source_id.iloc[0]),
                         **metrics(group.gold, group.score, threshold)))
    frame = pd.DataFrame(rows)
    valid = frame[frame.auroc.notna()]
    pairs = valid.positives * valid.negatives
    return frame, dict(mixed_answers=len(valid),
        macro_auroc=float(valid.auroc.mean()) if len(valid) else None,
        pair_weighted_auroc=float((valid.auroc*pairs).sum()/pairs.sum()) if pairs.sum() else None)


def error_roles(table, threshold):
    labels = table.gold.astype(bool)
    first = labels & (table.span_index == 0) & (table.offset == 0)
    onset = labels & (table.offset == 0)
    masks = dict(answer_first=first, later_onset=onset & ~first, continuation=labels & ~onset)
    rows = []
    for name, mask in masks.items():
        selected = mask | ~labels
        rows.append(dict(role=name, **metrics(mask[selected], table.score[selected], threshold)))
    return rows


def pair_regions(length):
    offset = np.arange(length)
    third = np.minimum(2, ((offset + .5) * 3 / length).astype(int))
    return dict(all=np.ones(length, bool), first=offset == 0, interior=(offset > 0) & (offset < length - 1),
                last=(offset == length-1) & (offset > 0), early_third=third == 0,
                middle_third=third == 1, late_third=third == 2)


def paired_scores(table, pairs, threshold):
    """Evaluate each region against the SAME region of its normal partner."""
    groups = {identity: group.set_index('token') for identity, group in table.groupby('id')}
    rows = []
    for pair in pairs:
        group = groups[str(pair['id'])]
        length, start, normal = pair['length'], pair['error_start'], pair['normal_start']
        error = group.loc[np.arange(start, start + length)]
        clean = group.loc[np.arange(normal, normal + length)]
        if not error.gold.all() or clean.gold.any():
            raise ValueError('Locked pair does not agree with current token labels')
        for region, selected in pair_regions(length).items():
            if not selected.any():
                continue
            a, b = error.score.to_numpy()[selected], clean.score.to_numpy()[selected]
            if not np.isfinite(np.r_[a, b]).all():
                continue
            measured = metrics(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b], threshold)
            rows.append(dict(id=str(pair['id']), source_id=str(pair['source_id']), tier=pair['tier'],
                error_start=start, normal_start=normal, length=length, region=region,
                first_span=int(error.span_index.iloc[0]) == 0,
                auroc=measured['auroc'], error_score=float(a.mean()), normal_score=float(b.mean()),
                margin=float(a.mean()-b.mean()), error_tokens=len(a), normal_tokens=len(b),
                hits=measured['tp'], false_alarms=measured['fp'],
                offset_win=float(((a>b) + .5*(a==b)).mean())))
    return pd.DataFrame(rows)


def source_interval(frame, column, bootstrap=200):
    values = frame.groupby('source_id')[column].mean().to_numpy()
    if not len(values):
        return dict(mean=None, low=None, high=None, sources=0)
    result = dict(mean=float(values.mean()), low=None, high=None, sources=len(values))
    if bootstrap and len(values) > 1:
        rng = np.random.default_rng(42)
        means = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(bootstrap)]
        result['low'], result['high'] = map(float, np.quantile(means, [.025, .975]))
    return result


def paired_summary(frame, bootstrap):
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for (tier, region), group in frame.groupby(['tier', 'region']):
        error_tokens, normal_tokens = int(group.error_tokens.sum()), int(group.normal_tokens.sum())
        interval = source_interval(group, 'auroc', bootstrap)
        rows.append(dict(tier=tier, region=region, pairs=len(group),
            auroc_macro=float(group.auroc.mean()), score_margin=float(group.margin.mean()),
            error_tokens=error_tokens, normal_tokens=normal_tokens,
            hits=int(group.hits.sum()), false_alarms=int(group.false_alarms.sum()),
            recall=float(group.hits.sum()/error_tokens), fpr=float(group.false_alarms.sum()/normal_tokens),
            **{'source_auc_'+key: value for key, value in interval.items()}))
    return pd.DataFrame(rows)


def analyze(table, spans, threshold, pairs, output, bootstrap=200):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not np.array_equal(table.predicted.to_numpy(), (table.score > threshold).astype(int).to_numpy()):
        raise ValueError('Saved token alarms disagree with the supplied threshold')
    table = annotate(table, spans)
    overall = metrics(table.gold, table.score, threshold)
    answers, within = answer_metrics(table, threshold)
    positions, offsets, lengths = summarize(table)
    matched = paired_scores(table, pairs, threshold)
    roles = error_roles(table, threshold)
    write_json(output/'summary.json', dict(overall=overall, within_answer=within,
        error_roles=roles, threshold=threshold, answers=table.id.nunique(),
        annotated_token_spans=len(table[table.gold==1][['id', 'span_start']].drop_duplicates())))
    for name, frame in (('positions', positions), ('offsets', offsets), ('positions_by_length', lengths),
                        ('answers', answers), ('pair_scores', matched),
                        ('matched_positions', paired_summary(matched, bootstrap))):
        frame.to_csv(output/(name+'.csv'), index=False)
    table.to_csv(output/'tokens.csv.gz', index=False)
    return overall, matched


def compare_pairs(all_pairs, output, bootstrap=200, reference='charm_in'):
    """Paired differences, not a comparison of unrelated aggregate AUCs."""
    if reference not in all_pairs or all_pairs[reference].empty:
        return
    keys = ['id', 'source_id', 'tier', 'error_start', 'normal_start', 'region']
    base = all_pairs[reference]
    rows = []
    for name, frame in all_pairs.items():
        if frame.empty:
            continue
        common = base.merge(frame, on=keys, suffixes=('_base', '_control'), validate='one_to_one')
        common['auc_delta'] = common.auroc_control - common.auroc_base
        common['margin_delta'] = common.margin_control - common.margin_base
        for (tier, region), group in common.groupby(['tier', 'region']):
            rows.append(dict(model=name, reference=reference, tier=tier, region=region, pairs=len(group),
                auc_base=float(group.auroc_base.mean()), auc_control=float(group.auroc_control.mean()),
                auc_delta=float(group.auc_delta.mean()), margin_delta=float(group.margin_delta.mean()),
                **{'source_delta_'+k:v for k,v in source_interval(group, 'auc_delta', bootstrap).items()}))
    pd.DataFrame(rows).to_csv(Path(output)/'paired_comparison.csv', index=False)
