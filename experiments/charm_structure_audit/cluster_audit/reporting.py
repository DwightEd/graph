"""Evaluate fixed matched pairs; never tune scores, directions or thresholds."""

import csv
import gzip
import html
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import CALIPERS, PROFILE_NAMES, STRUCTURE
from .matching import TIERS


PAIR_METRICS = ("within_pair_token_auroc", "mean_score_margin", "span_mean_win",
                "offset_win", "error_token_recall", "normal_cluster_token_fpr",
                "normal_cluster_any_alarm", "error_start_hit", "normal_start_alarm", "start_win")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def win(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (left > right).astype(float) + .5 * (left == right)


def ranking(error, normal):
    labels = np.r_[np.ones(len(error)), np.zeros(len(normal))]
    scores = np.r_[error, normal]
    return dict(auroc=float(roc_auc_score(labels, scores)),
                ap=float(average_precision_score(labels, scores)),
                error_tokens=len(error), normal_tokens=len(normal),
                prevalence=len(error) / len(labels))


def evaluate_pair(score, pair, threshold):
    """All token pairs stay WITHIN their matched error/normal interval pair."""
    start, normal, length = pair["error_start"], pair["normal_start"], pair["length"]
    error_scores, normal_scores = score[start:start + length], score[normal:normal + length]
    if not np.isfinite(np.r_[error_scores, normal_scores]).all():
        return None
    measured = ranking(error_scores, normal_scores)
    return dict(within_pair_token_auroc=measured["auroc"],
                mean_score_margin=float(error_scores.mean() - normal_scores.mean()),
                span_mean_win=float(win(error_scores.mean(), normal_scores.mean())),
                offset_win=float(win(error_scores, normal_scores).mean()),
                error_token_recall=float((error_scores > threshold).mean()),
                normal_cluster_token_fpr=float((normal_scores > threshold).mean()),
                normal_cluster_any_alarm=float((normal_scores > threshold).any()),
                error_start_hit=float(error_scores[0] > threshold),
                normal_start_alarm=float(normal_scores[0] > threshold),
                start_win=float(win(error_scores[0], normal_scores[0])))


def source_mean(rows, field, bootstrap=200, seed=42):
    by_source = {}
    for row in rows:
        if row[field] is not None:
            by_source.setdefault(row["source_id"], []).append(row[field])
    values = np.asarray([np.mean(v, axis=0) for v in by_source.values()])
    if not len(values):
        return dict(sources=0, mean=None, ci95=None)
    rng = np.random.default_rng(seed)
    draws = [values[rng.integers(len(values), size=len(values))].mean(axis=0)
             for _ in range(bootstrap if len(values) > 1 else 0)]
    return dict(sources=len(values), mean=np.asarray(values.mean(axis=0)).tolist(),
                ci95=np.quantile(draws, [.025, .975], axis=0).tolist() if draws else None)


def metric_summary(rows, bootstrap):
    return dict(pairs=len(rows), sources=len({r["source_id"] for r in rows}),
                pair_macro={key: float(np.mean([r[key] for r in rows])) if rows else None
                            for key in PAIR_METRICS},
                source_balanced={key: source_mean(rows, key, bootstrap) for key in PAIR_METRICS})


def summarize_pairs(pairs, scores, threshold, bootstrap=200, thresholds=None):
    """Every control is compared with baseline on IDENTICAL complete pairs."""
    rows, pooled_error, pooled_normal = [], [], []
    control_rows, start_rows = {}, {}
    for pair in pairs:
        identity = {k: pair[k] for k in ("id", "source_id", "error_start", "normal_start", "length", "tier")}
        saved = scores[pair["id"]]
        base = evaluate_pair(saved["score"], pair, threshold)
        rows.append(dict(identity, **base))
        start, normal, length = pair["error_start"], pair["normal_start"], pair["length"]
        pooled_error.extend(saved["score"][start:start + length])
        pooled_normal.extend(saved["score"][normal:normal + length])
        for name, score in saved.items():
            if name == "score":
                continue
            altered = evaluate_pair(score, pair, (thresholds or {}).get(name, threshold))
            if altered is not None:
                values = {key: altered[key] - base[key] for key in PAIR_METRICS}
                control_rows.setdefault(name, []).append(dict(identity, **values))
            if np.isfinite(score[[start, normal]]).all():
                start_rows.setdefault(name, []).append(dict(identity,
                    start_win_delta=float(win(score[start], score[normal]) -
                                          win(saved["score"][start], saved["score"][normal]))))
    controls = {}
    for name in sorted({key for row in scores.values() for key in row if key != "score"}):
        changes = control_rows.get(name, [])
        beginnings = start_rows.get(name, [])
        complete_keys = {(r["id"], r["error_start"]) for r in changes}
        common_base = [r for r in rows if (r["id"], r["error_start"]) in complete_keys]
        controls[name] = dict(complete_pairs=len(changes), total_pairs=len(pairs),
                             baseline_on_same_pairs=metric_summary(common_base, bootstrap),
                             delta_control_minus_base=metric_summary(changes, bootstrap),
                             start_only_pairs=len(beginnings),
                             start_only_win_delta=source_mean(beginnings, "start_win_delta", bootstrap))
    result = metric_summary(rows, bootstrap)
    result["pooled_matched_token"] = ranking(np.asarray(pooled_error), np.asarray(pooled_normal)) if pairs else None
    result["controls"] = controls
    return result, rows


def balance_report(pairs):
    rows = []
    for index, name in enumerate(STRUCTURE):
        error = np.asarray([r["error_structure"][index] for r in pairs])
        normal = np.asarray([r["normal_structure"][index] for r in pairs])
        differences = error - normal
        spread = np.sqrt((np.var(error) + np.var(normal)) / 2) if pairs else 0.
        rows.append(dict(feature=name, pairs=len(pairs), caliper=float(CALIPERS[index]),
                         error_mean=float(error.mean()) if pairs else None,
                         normal_mean=float(normal.mean()) if pairs else None,
                         mean_absolute_gap=float(abs(differences).mean()) if pairs else None,
                         max_absolute_gap=float(abs(differences).max()) if pairs else None,
                         standardized_mean_difference=float(differences.mean() / spread) if spread else None))
    return rows


def token_rows(samples, pairs, threshold):
    by_id = {s["id"]: s for s in samples}
    rows = []
    for pair in pairs:
        sample = by_id[pair["id"]]
        for role, start in (("error", pair["error_start"]), ("normal", pair["normal_start"])):
            for offset, token in enumerate(range(start, start + pair["length"])):
                left, right = sample["offsets"][token]
                score = float(sample["score"][token])
                alarm = score > threshold
                outcome = ("TP" if alarm else "FN") if role == "error" else ("FP" if alarm else "TN")
                rows.append(dict(id=sample["id"], source_id=sample["source_id"], tier=pair["tier"],
                    pair_error_start=pair["error_start"], role=role, token=token, relative_offset=offset,
                    text=sample["response"][left:right], score=score, threshold=threshold,
                    alarm=int(alarm), outcome=outcome))
    return rows


def head_rows(profiles, heads, bootstrap):
    rows = []
    for tier in TIERS:
        selected = [r for r in profiles if r["tier"] == tier]
        measured = source_mean(selected, "difference", bootstrap)
        if not selected:
            continue
        mean = np.asarray(measured["mean"])
        bounds = measured["ci95"]
        for feature, name in enumerate(PROFILE_NAMES):
            for channel in range(mean.shape[1]):
                rows.append(dict(tier=tier, feature=name, layer=channel // heads, head=channel % heads,
                    pairs=len(selected), sources=measured["sources"], error_minus_normal=mean[feature, channel],
                    low=bounds[0][feature][channel] if bounds else None,
                    high=bounds[1][feature][channel] if bounds else None,
                    interpretation="association_only_exploratory_uncorrected"))
    return rows


def gallery(samples, pairs, path):
    by_id = {s["id"]: s for s in samples}
    page = ['<!doctype html><meta charset="utf-8"><title>Matched clusters</title>',
            '<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto;line-height:1.6}table{width:100%}td{padding:12px;vertical-align:top}span{white-space:pre-wrap}details{margin:1em 0}</style>',
            '<h1>结构相近的错误 / 正常片段</h1><p>全部配对按身份排序展示，不按成绩选例。括号为原分数和固定阈值结果；“正常”指无幻觉标注。</p>']
    for pair in sorted(pairs, key=lambda r: (r["tier"], r["id"], r["error_start"])):
        sample = by_id[pair["id"]]
        page.append('<details><summary>' + html.escape(f'{pair["tier"]} | {pair["id"]} | error {pair["error_start"]} / normal {pair["normal_start"]} | L={pair["length"]}') + '</summary><table><tr>')
        for role, start in (("错误", pair["error_start"]), ("正常", pair["normal_start"])):
            fragments = []
            for token in range(start, start + pair["length"]):
                a, b = sample["offsets"][token]
                text = html.escape(sample["response"][a:b])
                fragments.append(f'<span title="token {token}">{text} ({sample["score"][token]:.3f})</span>')
            page.append('<td><strong>' + role + '</strong><p>' + ' '.join(fragments) + '</p></td>')
        page.append('</tr></table><p>结构匹配距离：' + str(round(pair["structure_distance"], 4)) + '</p></details>')
    Path(path).write_text('\n'.join(page), encoding="utf-8")
