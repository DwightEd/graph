"""Token/span diagnostics. Gold boundaries never extend model predictions."""

import csv
import html
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import write_json


def divide(a, b):
    return float(a / b) if b else None


def runs(mask):
    change = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def threshold_at_fpr(samples, fpr=.05):
    """Calibrate on normal text tokens in a held-out calibration partition."""
    negative = np.concatenate([s['score'][(~s['gold']) & (s['offsets'][:, 1] > s['offsets'][:, 0])]
                               for s in samples])
    if not len(negative):
        raise ValueError('calibration has no normal tokens')
    threshold = float(np.quantile(negative, 1 - fpr, method='higher'))
    return dict(value=threshold, rule='score > threshold', target_fpr=fpr,
                achieved_fpr=float((negative > threshold).mean()), negatives=len(negative),
                origin='source_disjoint_calibration_normal_text_tokens')


def classification(labels, scores, threshold, weights=None):
    y, score = np.asarray(labels, bool), np.asarray(scores, float)
    good = np.isfinite(score)
    y, score = y[good], score[good]
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)[good]
    pred = score > threshold
    tp, fn, fp, tn = (float(w[mask].sum()) for mask in
                      (y & pred, y & ~pred, ~y & pred, ~y & ~pred))
    both = tp + fn > 0 and fp + tn > 0
    return dict(tokens=len(y), positives=int(y.sum()), coverage=divide(good.sum(), len(good)),
                prevalence=divide(tp + fn, w.sum()),
                auroc=float(roc_auc_score(y, score, sample_weight=w)) if both else None,
                ap=float(average_precision_score(y, score, sample_weight=w)) if tp + fn else None,
                accuracy=divide(tp + tn, w.sum()), precision=divide(tp, tp + fp),
                recall=divide(tp, tp + fn), fpr=divide(fp, fp + tn),
                f1=divide(2 * tp, 2 * tp + fn + fp), tp=tp, fn=fn, fp=fp, tn=tn)


def scopes(sample):
    y, onset = sample['gold'].astype(bool), sample['onset'].astype(bool)
    all_tokens = np.ones(len(y), bool)
    first, after, until = np.zeros(len(y), bool), np.zeros(len(y), bool), all_tokens.copy()
    if y.any():
        t = np.flatnonzero(y)[0]
        first[t], after[t+1:], until[t+1:] = True, True, False
    return dict(all_error=(y, all_tokens), text_only=(y, sample['offsets'][:, 1] > sample['offsets'][:, 0]),
                first_error_vs_normal=(first, ~y | first),
                first_error_until_first=(first, until), span_onset_vs_normal=(onset, ~y | onset),
                continuation_vs_normal=(y & ~onset, ~onset), strict_post_first=(y, after))


def span_rows(sample, threshold):
    y, score = sample['gold'].astype(bool), sample['score']
    pred = score > threshold
    first = int(np.flatnonzero(y)[0]) if y.any() else -1
    result = []
    for index, (start, end) in enumerate(sample['spans']):
        start, end = int(start), int(end)
        hit = np.flatnonzero(pred[start:end])
        cont = float(pred[start+1:end].mean()) if end > start + 1 else None
        result.append(dict(id=sample['id'], source_id=sample['source_id'], task=sample['task'],
                           annotation_index=index, start=start, end=end, length=end-start,
                           first_in_answer=start == first, onset_score=float(score[start]),
                           onset_hit=bool(pred[start]), coverage=float(pred[start:end].mean()),
                           continuation_coverage=cont, any_hit=bool(len(hit)),
                           delay_if_detected=int(hit[0]) if len(hit) else None,
                           onset_missed_later_80=bool(not pred[start] and cont is not None and cont >= .8),
                           onset_missed_later_all=bool(not pred[start] and cont == 1.)))
    return result


def boundary_summary(samples, threshold):
    spans = [r for s in samples for r in span_rows(s, threshold)]
    first = [r for r in spans if r['first_in_answer']]
    # Distinct annotated spans can overlap in token space; answer-first counted once.
    first = list({r['id']: r for r in first}.values())
    def summarize(rows):
        eligible = [r for r in rows if r['length'] > 1]
        delays = [r['delay_if_detected'] for r in rows if r['any_hit']]
        return dict(count=len(rows), onset_recall=divide(sum(r['onset_hit'] for r in rows), len(rows)),
                    mean_token_coverage=divide(sum(r['coverage'] for r in rows), len(rows)),
                    fully_covered=divide(sum(r['coverage'] == 1 for r in rows), len(rows)),
                    undetected=sum(not r['any_hit'] for r in rows),
                    mean_delay_detected_only=float(np.mean(delays)) if delays else None,
                    missed_onset_count=sum(not r['onset_hit'] for r in rows),
                    missed_onset_later_all_count=sum(r['onset_missed_later_all'] for r in eligible),
                    missed_onset_later_all_given_missed=divide(sum(r['onset_missed_later_all'] for r in eligible),
                                                              sum(not r['onset_hit'] for r in eligible)),
                    missed_onset_later80_count=sum(r['onset_missed_later_80'] for r in eligible),
                    missed_onset_later80_denominator=len(eligible),
                    missed_onset_later80_rate=divide(sum(r['onset_missed_later_80'] for r in eligible), len(eligible)))
    post_y, post_s = [], []
    predicted_segments = gold_segments = matched = 0
    clean_answers, clean_false = 0, 0
    for s in samples:
        y, pred = s['gold'].astype(bool), s['score'] > threshold
        if not y.any():
            clean_answers += 1; clean_false += int(pred.any())
        post = np.zeros(len(y), bool)
        for _, end in runs(y):
            post[end:min(len(y), end + 5)] = True
        post &= ~y & (s['offsets'][:, 1] > s['offsets'][:, 0])
        post_y.extend(y[post]); post_s.extend(s['score'][post])
        truth, proposed = runs(y), runs(pred)
        gold_segments += len(truth); predicted_segments += len(proposed)
        overlaps = []
        for i, (a, b) in enumerate(truth):
            for j, (c, d) in enumerate(proposed):
                intersection = max(0, min(b, d) - max(a, c))
                union = max(b, d) - min(a, c)
                overlaps.append((intersection / union, i, j))
        used_i, used_j = set(), set()
        for iou, i, j in sorted(overlaps, reverse=True):
            if iou >= .5 and i not in used_i and j not in used_j:
                used_i.add(i); used_j.add(j); matched += 1
    lengths = {name: summarize([r for r in spans if lo <= r['length'] <= hi])
               for name, lo, hi in [('1', 1, 1), ('2-4', 2, 4), ('5-16', 5, 16), ('17+', 17, 10**9)]}
    return dict(annotation_spans=summarize(spans), first_error_spans=summarize(first),
                by_span_length=lengths, post_end_5_normal=classification(post_y, post_s, threshold),
                clean_answer_false_alarm=divide(clean_false, clean_answers), clean_answers=clean_answers,
                predicted_contiguous_runs=predicted_segments, gold_contiguous_runs=gold_segments,
                matched_iou50_greedy=matched,
                span_precision_iou50=divide(matched, predicted_segments),
                span_recall_iou50=divide(matched, gold_segments),
                span_f1_iou50=divide(2 * matched, predicted_segments + gold_segments)), spans


def group_report(samples, threshold):
    report = {}
    for name in scopes(samples[0]):
        labels, scores = [], []
        for s in samples:
            y, mask = scopes(s)[name]
            labels.extend(y[mask]); scores.extend(s['score'][mask])
        report[name] = classification(labels, scores, threshold)
    first, continuation = report['span_onset_vs_normal'], report['continuation_vs_normal']
    total = first['positives'] + continuation['positives']
    reconstructed = None
    if total and first['auroc'] is not None and continuation['auroc'] is not None:
        reconstructed = (first['positives'] * first['auroc'] + continuation['positives'] * continuation['auroc']) / total
    boundaries, span_table = boundary_summary(samples, threshold)
    return dict(responses=len(samples), sources=len({s['source_id'] for s in samples}),
                token_scopes=report, boundaries=boundaries,
                auc_decomposition=dict(onset_positive_share=divide(first['positives'], total),
                                       continuation_positive_share=divide(continuation['positives'], total),
                                       reconstructed_all_auroc=reconstructed,
                                       note='same scores and same normal negatives; not an AP decomposition')), span_table


def paired_delta(samples, key, bootstrap=200):
    """Source-bootstrap ranking change on common observed tokens; inference only."""
    y = np.concatenate([s['gold'] for s in samples])
    a = np.concatenate([s['score'] for s in samples])
    b = np.concatenate([s[key] for s in samples])
    sid = np.concatenate([np.repeat(s['source_id'], len(s['gold'])) for s in samples])
    valid = np.isfinite(a) & np.isfinite(b)
    y, a, b, sid = y[valid], a[valid], b[valid], sid[valid]
    ra, rb = classification(y, a, .5), classification(y, b, .5)
    delta = {k: rb[k] - ra[k] if ra[k] is not None and rb[k] is not None else None for k in ('auroc', 'ap')}
    unique, inverse = np.unique(sid, return_inverse=True)
    rng, draws = np.random.default_rng(42), []
    for _ in range(bootstrap if len(unique) > 1 else 0):
        weights = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[inverse]
        aa, bb = classification(y, a, .5, weights), classification(y, b, .5, weights)
        if aa['auroc'] is not None and bb['auroc'] is not None:
            draws.append([bb[k] - aa[k] for k in ('auroc', 'ap')])
    return dict(tokens=len(y), mean_absolute_score_change=float(np.mean(abs(a-b))) if len(y) else None,
                max_absolute_score_change=float(np.max(abs(a-b))) if len(y) else None,
                delta_control_minus_real=delta, bootstrap_order=['auroc', 'ap'],
                source_bootstrap_ci95=np.quantile(draws, [.025, .975], axis=0).tolist() if draws else None)


def write_table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def report(samples, output, threshold, protocol, bootstrap=200, examples=40):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    groups = {'ALL': samples}
    for s in samples:
        groups.setdefault(s['task'] + '|' + s['generator'], []).append(s)
    result = dict(protocol=protocol, threshold=threshold, groups={}, interventions={})
    span_table = []
    for group, rows in groups.items():
        value, spans = group_report(rows, threshold['value'])
        result['groups'][group] = value
        if group == 'ALL':
            span_table = spans
    for key in samples[0]:
        if key.startswith('score_'):
            available = [s for s in samples if key in s]
            result['interventions'][key] = paired_delta(available, key, bootstrap)
            changed = [dict(s, score=s[key]) for s in available]
            result['interventions'][key]['token_scopes'] = group_report(changed, threshold['value'])[0]['token_scopes']
    result['representation_by_response'] = {
        s['id']: s.get('diagnostic', {}).get('representation', {}) for s in samples
    }
    result['graph_changes_by_response'] = {
        s['id']: {k: v for k, v in s.get('diagnostic', {}).items() if k != 'representation'} for s in samples
    }
    result['structure_correlations'] = {}
    base_score = np.concatenate([s['score'] for s in samples])
    for name in ('in_rp', 'in_rr', 'out_degree', 'local_rr_fraction', 'self_attention_mean'):
        values = np.concatenate([s['structure_' + name] for s in samples])
        result['structure_correlations'][name] = float(spearmanr(values, base_score).statistic) if np.ptp(values) and np.ptp(base_score) else None
    result['warnings'] = [
        'Gold spans are used only to inspect predictions; no point-adjustment or gold-span expansion.',
        'Frozen perturbations are sensitivity/OOD tests, not retrained accuracy attribution.',
        'Correlations and embedding similarity do not prove factual dependency.',
        'Post-token detection is not before-emission forecasting; original out-degree also uses future graph structure.',
    ]
    write_json(output / 'report.json', result)
    write_table(output / 'spans.csv', span_table)
    token_rows = []
    for s in samples:
        first = int(np.flatnonzero(s['gold'])[0]) if s['gold'].any() else -1
        for t, (a, b) in enumerate(s['offsets']):
            row = dict(id=s['id'], source_id=s['source_id'], task=s['task'], token=t,
                       text=s['response'][a:b], gold=int(s['gold'][t]), score=float(s['score'][t]),
                       predicted=int(s['score'][t] > threshold['value']),
                       span_onset=int(s['onset'][t]), answer_first=int(t == first))
            row.update({k: float(v[t]) for k, v in s.items() if k.startswith('structure_')})
            token_rows.append(row)
    write_table(output / 'tokens.csv', token_rows)
    # Select examples only AFTER metrics; evaluation itself uses all samples.
    missed = {r['id'] for r in span_table if r['onset_missed_later_80']}
    chosen = sorted(samples, key=lambda s: (s['id'] not in missed, s['id']))[:examples]
    page = ['<!doctype html><meta charset="utf-8"><title>CHARM token audit</title>',
            '<style>body{font:16px sans-serif;max-width:1200px;margin:35px auto;line-height:2} .tp{background:#b4e3bb}.fn{background:#f9b6b6}.fp{background:#ffdf93} span{white-space:pre-wrap;padding:2px}.first{outline:2px solid black}</style>',
            '<h1>CHARM token audit</h1><p>Green=TP; red=FN; amber=FP; black outline=first error. '
            'Examples prioritize missed-onset/high-continuation cases; they are not a random sample.</p>']
    for s in chosen:
        page.append('<h2>' + html.escape(s['id'] + ' · ' + s['task']) + '</h2><p>')
        first = int(np.flatnonzero(s['gold'])[0]) if s['gold'].any() else -1
        for t, (a, b) in enumerate(s['offsets']):
            predicted, gold = s['score'][t] > threshold['value'], bool(s['gold'][t])
            style = ('tp' if predicted and gold else 'fn' if gold else 'fp' if predicted else '') + (' first' if t == first else '')
            page.append(f'<span class="{style}" title="token={t}, score={s["score"][t]:.5f}, gold={int(gold)}">{html.escape(s["response"][a:b])}</span>')
        page.append('</p>')
    (output / 'gallery.html').write_text('\n'.join(page), encoding='utf-8')
    return result
