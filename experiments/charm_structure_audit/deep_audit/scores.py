"""Population, exact AUC accounting, and within-answer localization."""

import html
from pathlib import Path

import numpy as np

from .common import (finite_mean, fraction, intervals, membership, merged_spans,
                     metric, roles, win_credit, write_csv, write_json)


POSITIVE = ("first_error", "later_onset", "continuation")
NEGATIVE = ("normal_clean", "normal_before", "normal_after")


def population(samples):
    gold = np.concatenate([s["gold"] for s in samples]).astype(bool)
    masks = {name: np.concatenate([roles(s)[name] for s in samples]) for name in POSITIVE + NEGATIVE}
    lengths = [end - start for s in samples for start, end in merged_spans(s)]
    text = np.concatenate([s["offsets"][:, 1] > s["offsets"][:, 0] for s in samples])
    result = dict(answers=len(samples), sources=len({s["source_id"] for s in samples}),
                  tokens=len(gold), error_tokens=int(gold.sum()), normal_tokens=int((~gold).sum()),
                  error_fraction=fraction(gold.sum(), len(gold)), text_tokens=int(text.sum()),
                  nontext_tokens=int((~text).sum()), error_text_tokens=int((gold & text).sum()),
                  clean_answers=sum(not np.any(s["gold"]) for s in samples),
                  all_error_answers=sum(bool(np.all(s["gold"])) for s in samples),
                  raw_annotation_spans=sum(len(s["spans"]) for s in samples),
                  overlap_merged_spans=len(lengths),
                  contiguous_error_runs=sum(len(intervals(s["gold"])) for s in samples))
    result["roles"] = {name: dict(tokens=int(mask.sum()), fraction_all=fraction(mask.sum(), len(gold)),
                                 fraction_error=fraction(mask.sum(), gold.sum()) if name in POSITIVE else None)
                       for name, mask in masks.items()}
    result["span_lengths"] = {name: sum(lo <= length <= hi for length in lengths)
                              for name, lo, hi in [("1", 1, 1), ("2_4", 2, 4), ("5_16", 5, 16), ("17_plus", 17, 10**9)]}
    result["scope"] = "Only the saved evaluated answers, not an assumed full RAGTruth census."
    return result


def auc_accounting(samples, key="score"):
    """Disjoint role/context/pair cells sum exactly to the pooled AUROC."""
    available = [s for s in samples if key in s]
    masks = [{name: mask & np.isfinite(s[key]) for name, mask in roles(s).items()} for s in available]
    positives = sum(int(s["gold"][np.isfinite(s[key])].sum()) for s in available)
    negatives = sum(int((~s["gold"].astype(bool)[np.isfinite(s[key])]).sum()) for s in available)
    denominator = positives * negatives
    cells = []
    for positive in POSITIVE:
        pos = np.concatenate([s[key][mask[positive]] for s, mask in zip(available, masks)])
        for negative in NEGATIVE:
            neg = np.concatenate([s[key][mask[negative]] for s, mask in zip(available, masks)])
            pairs = len(pos) * len(neg)
            wins = float(win_credit(pos, neg).sum() * len(neg)) if len(neg) else 0.
            same_pairs, same_wins = 0, 0.
            for sample, mask in zip(available, masks):
                local_pos, local_neg = sample[key][mask[positive]], sample[key][mask[negative]]
                same_pairs += len(local_pos) * len(local_neg)
                if len(local_neg):
                    same_wins += float(win_credit(local_pos, local_neg).sum() * len(local_neg))
            for scope, count, credit in [("within_answer", same_pairs, same_wins),
                                         ("cross_answer", pairs - same_pairs, wins - same_wins)]:
                cells.append(dict(positive_role=positive, negative_context=negative, scope=scope,
                                  pairs=count, wins=credit, auroc=fraction(credit, count),
                                  pair_fraction=fraction(count, denominator),
                                  contribution_above_chance=fraction(credit - .5 * count, denominator)))
    return dict(pairs=denominator, reconstructed_auroc=fraction(sum(c["wins"] for c in cells), denominator),
                cells=cells, note="Pair accounting, NOT causal contribution; AP is not additive.")


def within_answers(samples, threshold, key="score"):
    rows = []
    for sample in samples:
        if key not in sample:
            continue
        masks = roles(sample)
        for role in ("all_error", *POSITIVE):
            positive = sample["gold"].astype(bool) if role == "all_error" else masks[role]
            for context in ("all_normal", "normal_before", "normal_after"):
                negative = ~sample["gold"].astype(bool) if context == "all_normal" else masks[context]
                selected = positive | negative
                values = metric(positive[selected], sample[key][selected], threshold)
                rows.append(dict(id=sample["id"], source_id=sample["source_id"], role=role,
                                 negative_context=context, **values))
    summary = []
    for role in ("all_error", *POSITIVE):
        for context in ("all_normal", "normal_before", "normal_after"):
            group = [r for r in rows if r["role"] == role and r["negative_context"] == context and r["auroc"] is not None]
            pairs = sum(r["positive_tokens"] * r["negative_tokens"] for r in group)
            wins = sum(r["auroc"] * r["positive_tokens"] * r["negative_tokens"] for r in group)
            summary.append(dict(role=role, negative_context=context, eligible_answers=len(group),
                                macro_auroc=finite_mean([r["auroc"] for r in group]),
                                macro_ap=finite_mean([r["ap"] for r in group]),
                                pairs=pairs, pair_weighted_auroc=fraction(wins, pairs)))
    return summary, rows


def span_comparisons(sample, threshold, window):
    """All-positive spans get coverage, NOT a fictitious within-span AUROC."""
    gold = sample["gold"].astype(bool)
    score = sample["score"]
    spans = merged_spans(sample)
    rows, gaps = [], []
    for index, (start, end) in enumerate(spans):
        left = spans[index - 1][1] if index else 0
        right = spans[index + 1][0] if index + 1 < len(spans) else len(gold)
        neighbors = np.r_[np.arange(max(left, start - window), start), np.arange(end, min(right, end + window))]
        normal = np.flatnonzero(~gold)
        positive = np.arange(start, end)
        hit = np.flatnonzero(score[start:end] > threshold)
        row = dict(id=sample["id"], source_id=sample["source_id"], span=index, start=start, end=end,
                   length=end - start, coverage=float(np.mean(score[start:end] > threshold)),
                   onset_score=float(score[start]), onset_hit=bool(score[start] > threshold),
                   score_std=float(np.std(score[start:end])), score_range=float(np.ptp(score[start:end])),
                   continuation_mean=finite_mean(score[start + 1:end]),
                   delay=int(hit[0]) if len(hit) else None, within_span_auroc=None)
        for name, negatives in [("same_answer", normal), ("nearby", neighbors)]:
            credits = win_credit(score[positive], score[negatives])
            row[name + "_normal_tokens"] = len(negatives)
            row[name + "_auroc"] = finite_mean(credits)
            row[name + "_onset_credit"] = finite_mean(credits[:1])
        rows.append(row)
        if index and start > left:
            gap_score = score[left:start]
            gaps.append(dict(id=sample["id"], source_id=sample["source_id"], left_span=index - 1,
                             right_span=index, start=left, end=start, tokens=start - left,
                             false_positives=int((gap_score > threshold).sum()),
                             fpr=float(np.mean(gap_score > threshold)), mean_score=float(gap_score.mean())))
    return rows, gaps


def token_ledger(samples, threshold):
    negatives = np.concatenate([s["score"][~s["gold"].astype(bool)] for s in samples])
    positive_count = sum(int(s["gold"].sum()) for s in samples)
    controls = sorted({key for s in samples for key in s if key.startswith("score_")})
    for sample in samples:
        masks = roles(sample)
        member = membership(sample)
        local_normal = sample["score"][~sample["gold"].astype(bool)]
        global_credit = win_credit(sample["score"], negatives)
        local_credit = win_credit(sample["score"], local_normal)
        for token, (start, end) in enumerate(sample["offsets"]):
            gold = bool(sample["gold"][token])
            alarm = bool(sample["score"][token] > threshold)
            outcome = ("TP" if alarm else "FN") if gold else ("FP" if alarm else "TN")
            role = next(name for name, mask in masks.items() if mask[token])
            row = dict(id=sample["id"], source_id=sample["source_id"], task=sample["task"],
                       generator=sample["generator"], token=token, char_start=int(start), char_end=int(end),
                       text=sample["response"][start:end], gold=int(gold), score=float(sample["score"][token]),
                       threshold=threshold, outcome=outcome, role=role, span=int(member[token]),
                       auroc_credit_vs_all_normal=finite_mean([global_credit[token]]) if gold else None,
                       auroc_credit_vs_same_answer_normal=finite_mean([local_credit[token]]) if gold else None,
                       contribution_above_chance=fraction(global_credit[token] - .5, positive_count)
                       if gold and len(negatives) else None)
            row.update({key: finite_mean([sample[key][token]]) if key in sample else None for key in controls})
            yield row


def control_comparison(samples, key, threshold, bootstrap=0, seed=42):
    """Same finite coordinates, same source bootstrap draws, unchanged threshold."""
    available = [s for s in samples if key in s]
    gold = np.concatenate([s["gold"] for s in available]).astype(bool)
    base = np.concatenate([s["score"] for s in available])
    other = np.concatenate([s[key] for s in available])
    source = np.concatenate([np.repeat(s["source_id"], len(s["gold"])) for s in available])
    common = np.isfinite(base) & np.isfinite(other)
    gold, base, other, source = gold[common], base[common], other[common], source[common]
    baseline, control = metric(gold, base, threshold), metric(gold, other, threshold)
    unique, inverse = np.unique(source, return_inverse=True)
    rng, draws = np.random.default_rng(seed), []
    for _ in range(bootstrap if len(unique) > 1 else 0):
        weights = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[inverse]
        first, second = metric(gold, base, threshold, weights), metric(gold, other, threshold, weights)
        if first["auroc"] is not None and second["auroc"] is not None:
            draws.append([second[name] - first[name] for name in ("auroc", "ap")])
    changes = {}
    for role in POSITIVE + NEGATIVE:
        mask = np.concatenate([roles(s)[role] for s in available])[common]
        changes[role] = dict(tokens=int(mask.sum()), mean_delta=finite_mean((other - base)[mask]),
                             mean_absolute_delta=finite_mean(abs(other - base)[mask]),
                             lost_alarms=int(((base > threshold) & (other <= threshold) & mask).sum()),
                             gained_alarms=int(((base <= threshold) & (other > threshold) & mask).sum()))
    role_rankings = {}
    for role in POSITIVE:
        positive = np.concatenate([roles(s)[role] for s in available])[common]
        selected = positive | ~gold
        role_rankings[role] = dict(baseline=metric(positive[selected], base[selected], threshold),
                                   control=metric(positive[selected], other[selected], threshold))
    paired = [dict(s, score=np.where(np.isfinite(s[key]), s["score"], np.nan)) for s in available]
    baseline_within = within_answers(paired, threshold)[0]
    return dict(common_tokens=len(gold), total_tokens=sum(len(s["gold"]) for s in samples),
                baseline=baseline, control=control, changes=changes, role_rankings=role_rankings,
                baseline_within=baseline_within, control_within=within_answers(available, threshold, key)[0],
                delta_ci95=np.quantile(draws, [.025, .975], axis=0).tolist() if draws else None,
                bootstrap_valid=len(draws), bootstrap_order=["auroc", "ap"])


def gallery(samples, threshold, output):
    """All answers, exact original text spans; no selected success-only gallery."""
    chunks = ["<!doctype html><meta charset='utf-8'><title>Token audit</title>",
              "<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto;line-height:2}mark{white-space:pre-wrap}details{margin:15px 0}</style>",
              "<h1>逐 token 审计</h1><p>TP=命中错误；FN=漏报；FP=正常词误报；TN=正常词正确放过。所有回答均可展开；重复offset按token显示。</p>"]
    for sample in samples:
        chunks.append("<details><summary>" + html.escape(sample["id"] + " · " + sample["task"]) + "</summary>")
        member = membership(sample)
        for index, (start, end) in enumerate(sample["offsets"]):
            gold = bool(sample["gold"][index])
            alarm = sample["score"][index] > threshold
            outcome = ("TP" if alarm else "FN") if gold else ("FP" if alarm else "TN")
            title = f"token={index} span={member[index]} {outcome} score={sample['score'][index]:.6f}"
            chunks.append("<mark title='" + html.escape(title) + "'>[" + outcome + "] " + html.escape(sample["response"][start:end]) + "</mark> ")
        chunks.append("</details>")
    (Path(output) / "tokens.html").write_text("\n".join(chunks), encoding="utf-8")


def grouped_scores(samples, threshold):
    groups = {}
    for sample in samples:
        key = "|".join((sample["task"], sample["generator"], sample.get("split", "unknown")))
        groups.setdefault(key, []).append(sample)
    result = {}
    for key, group in groups.items():
        gold = np.concatenate([sample["gold"] for sample in group])
        score = np.concatenate([sample["score"] for sample in group])
        result[key] = dict(population=population(group), all_tokens=metric(gold, score, threshold),
                           within_answers=within_answers(group, threshold)[0])
    return result


def analyze_scores(samples, settings, output, window=10, bootstrap=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    threshold = float(settings["threshold"]["value"])
    gold = np.concatenate([s["gold"] for s in samples])
    score = np.concatenate([s["score"] for s in samples])
    text = np.concatenate([s["offsets"][:, 1] > s["offsets"][:, 0] for s in samples])
    within, answer_rows = within_answers(samples, threshold)
    spans, gaps = [], []
    for sample in samples:
        span_rows, gap_rows = span_comparisons(sample, threshold, window)
        spans.extend(span_rows)
        gaps.extend(gap_rows)
    result = dict(population=population(samples), threshold=settings["threshold"],
                  all_tokens=metric(gold, score, threshold), text_only=metric(gold[text], score[text], threshold),
                  auc_accounting=auc_accounting(samples), within_answers=within,
                  span_macro_nearby_auroc=finite_mean([r["nearby_auroc"] for r in spans if r["nearby_auroc"] is not None]),
                  gap_fpr=fraction(sum(r["false_positives"] for r in gaps), sum(r["tokens"] for r in gaps)),
                  groups=grouped_scores(samples, threshold),
                  controls={key: control_comparison(samples, key, threshold, bootstrap)
                            for key in sorted({key for s in samples for key in s if key.startswith("score_")})})
    write_json(output / "scores.json", result)
    write_csv(output / "tokens.csv", token_ledger(samples, threshold))
    write_csv(output / "answer_rankings.csv", answer_rows)
    write_csv(output / "spans.csv", spans)
    write_csv(output / "normal_gaps.csv", gaps)
    gallery(samples, threshold, output)
    return result
