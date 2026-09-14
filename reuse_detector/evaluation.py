"""Label-using evaluation only. No scoring/refitting or oracle seeds in this module.

Source weights are fixed from COMPLETE answers before gold-defined truncation.
The calibration threshold uses unlabeled source maxima, never normal labels.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from binding_detector.evaluation import label_views
from .run import write_json, digest


class Ranking:
    """Pre-sort once; fast exact weighted AUROC/AP for source bootstrap and ties."""
    def __init__(self, y, score):
        self.y = np.asarray(y, bool); s = np.asarray(score, float)
        if s.shape != self.y.shape or not np.isfinite(s).all():
            raise ValueError('evaluation requires complete finite score coverage')
        self.order = np.argsort(s, kind='stable')
        self.starts = np.r_[0, 1 + np.flatnonzero(np.diff(s[self.order]) != 0)] if len(s) else np.array([], int)

    def measure(self, weights=None):
        w = np.ones(len(self.y)) if weights is None else np.asarray(weights, float)
        if w.shape != self.y.shape or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError('invalid evaluation weights')
        total = float(w.sum()); positive = float(w @ self.y); negative = total - positive
        out = dict(prevalence=positive/total if total else None, auroc=None, ap=None)
        if min(positive, negative) <= 0:
            return out
        yw, nw = (w * self.y)[self.order], (w * ~self.y)[self.order]
        p, n = np.add.reduceat(yw, self.starts), np.add.reduceat(nw, self.starts)
        out['auroc'] = float((p * (np.cumsum(n) - .5*n)).sum() / (positive*negative))
        cp, cn = np.cumsum(p[::-1]), np.cumsum(n[::-1])
        precision = np.divide(cp, cp+cn, out=np.zeros_like(cp), where=(cp+cn) > 0)
        out['ap'] = float((p[::-1] * precision).sum() / positive)
        return out


def scoped_metrics(y, scores, source, full_weights, primary, bootstrap):
    metrics = {k: Ranking(y, v) for k, v in scores.items()}
    result = dict(tokens=len(y), positives=int(y.sum()),
                  pooled={k: m.measure() for k, m in metrics.items()},
                  source_fixed_full_answer={k: m.measure(full_weights) for k, m in metrics.items()})
    controls = [k for k in ('seed_only', 'single_hop', 'mass_matched_uniform', 'lag_group_permuted', 'raw_entropy') if k != primary]
    _, inv = np.unique(source, return_inverse=True)
    ns = len(np.unique(source)); rng = np.random.default_rng(20260914)
    values = {k: [] for k in controls}
    for _ in range(bootstrap if ns > 1 else 0):
        multiplicity = np.bincount(rng.integers(ns, size=ns), minlength=ns)[inv]
        main = metrics[primary].measure(multiplicity)
        if main['auroc'] is None:
            continue
        for name in controls:
            other = metrics[name].measure(multiplicity)
            values[name].append([main['auroc']-other['auroc'], main['ap']-other['ap']])
    result['paired_source_bootstrap'] = dict(primary=primary, unit='source', metric='pooled AUROC/AP differences',
        controls={k: dict(valid=len(v), delta_ci95=np.quantile(v, [.025, .975], axis=0).tolist() if v else None)
                  for k, v in values.items()})
    return result


def alarm_metrics(records, targets, scores, thresholds, method):
    tp = fp = positives = negatives = 0; normal = []; before = []; first_hit = []
    span_hits, delays, spillover = [], [], []
    for r in records:
        rid = r['id']; y, onset = targets[rid]['all_error'][0], targets[rid]['span_onset_full_stream'][0]
        key = r['task']+'|'+r['generator']
        if key not in thresholds:
            raise ValueError('no unlabeled alarm calibration for '+key)
        alarm = scores[rid][method] > thresholds[key][method]
        label = onset if method == 'onset_rank' else y
        tp += int((alarm & label).sum()); fp += int((alarm & ~label).sum())
        positives += int(label.sum()); negatives += int((~label).sum())
        if not y.any():
            normal.append(bool(alarm.any()))
        else:
            t = np.flatnonzero(y)[0]
            before.append(bool(alarm[:t].any())); first_hit.append(bool(alarm[t]))
        # Contiguous error runs: not a claim that the model internally uses these spans.
        bounds = np.flatnonzero(np.diff(np.r_[False, y, False])).reshape(-1, 2)
        for a, b in bounds:
            found = np.flatnonzero(alarm[a:b]); span_hits.append(bool(len(found)))
            if len(found):
                delays.append(int(found[0]))
            stop = min(len(y), b+8)
            mask = ~y[b:stop]
            if mask.any():
                spillover.append(float(alarm[b:stop][mask].mean()))
    def avg(x): return float(np.mean(x)) if len(x) else None
    return dict(token_precision=tp/(tp+fp) if tp+fp else None, token_recall=tp/positives if positives else None,
                normal_token_fpr=fp/negatives if negatives else None,
                normal_responses=len(normal), normal_response_any_alarm=avg(normal),
                erroneous_response_early_alarm_rate=avg(before), first_error_exact_recall=avg(first_hit),
                contiguous_error_run_recall=avg(span_hits), mean_detected_run_delay_tokens=avg(delays),
                mean_post_run_normal_alarm_rate_next8=avg(spillover),
                threshold_origin='unlabeled calibration source maxima; no normal-FPR guarantee')


def evaluate(rows, out, annotation_path, bootstrap=200):
    out = Path(out); scoredir = out/'scores'
    freeze = json.loads((scoredir/'prediction_freeze.json').read_text())
    if not freeze.get('complete') or freeze['labels_read'] or freeze['settings']['roster_identity'] != digest(rows):
        raise ValueError('label-free prediction freeze mismatch')
    records = [r for r in rows if r['official_split']=='test']
    if not records:
        raise ValueError('no official test responses selected')
    ids = {r['id'] for r in records}; gold = {}
    # First access to annotations in the entire pipeline occurs here.
    with Path(annotation_path).open(encoding='utf-8') as stream:
        for line in stream:
            g = json.loads(line)
            rid = str(g['id'])
            if rid in ids:
                if rid in gold: raise ValueError('duplicate annotation ID')
                gold[rid] = g
    if set(gold) != ids: raise ValueError('missing evaluation labels')
    targets, scores = {}, {}
    for r in records:
        g = gold[r['id']]
        if (str(g['source_id'])!=r['source_id'] or g['split']!=r['official_split']
                or hashlib.sha256(g['response'].encode()).hexdigest()!=r['response_sha256']):
            raise ValueError('annotation identity mismatch')
        targets[r['id']] = label_views(np.asarray(r['offsets']), g['labels'], len(g['response']))
        with np.load(scoredir/(r['id']+'.npz'), allow_pickle=False) as a:
            if str(a['input_identity']) != digest(r): raise ValueError('score/input mismatch')
            scores[r['id']] = {k[7:]: a[k].copy() for k in a.files if k.startswith('score__')}
    thresholds = json.loads((scoredir/'thresholds.json').read_text())['values']
    groups = {'ALL': records}
    for r in records:
        groups.setdefault(r['task']+'|'+r['generator'], []).append(r)
    report = dict(method='unsupervised_local_reuse_v1', training_labels_used=False,
                  semantics='scores are anomalies, not truth probabilities; attention reuse is not signed causal support',
                  observer=True, historically_used_test=True, groups={})
    for group, selected in groups.items():
        counts = {}
        for r in selected: counts[r['source_id']] = counts.get(r['source_id'],0)+len(r['offsets'])
        result = dict(responses=len(selected), sources=len(counts), views={}, alarms={})
        for name in next(iter(targets.values())):
            ys, ss, ws = [], [], []; values = {k: [] for k in next(iter(scores.values()))}
            for r in selected:
                y, mask = targets[r['id']][name]
                ys.append(y[mask]); ss.append(np.repeat(r['source_id'], int(mask.sum())))
                ws.append(np.full(int(mask.sum()),1/counts[r['source_id']]))
                for k in values: values[k].append(scores[r['id']][k][mask])
            y, source, w = np.concatenate(ys), np.concatenate(ss), np.concatenate(ws)
            sc = {k: np.concatenate(v) for k, v in values.items()}
            primary = 'onset_rank' if name.startswith(('first_error','span_onset')) else 'reuse'
            result['views'][name] = scoped_metrics(y, sc, source, w, primary, bootstrap)
            main = result['views'][name]['pooled'][primary]
            print(json.dumps(dict(group=group, view=name, primary=primary, tokens=len(y), positives=int(y.sum()), **main)), flush=True)
        for name in ('onset_rank','reuse','mass_matched_uniform','lag_group_permuted'):
            result['alarms'][name] = alarm_metrics(selected, targets, scores, thresholds, name)
        report['groups'][group] = result
    # Post-hoc evaluation is replayable. Do not overwrite previous inference artifacts.
    write_json(out/'evaluation.json', report)
    write_json(out/'complete.json', dict(complete=True, inference_labels_used=False, evaluation_only_labels=True))
    return report
