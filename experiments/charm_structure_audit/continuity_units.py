"""Sentence ranking/calibration and fixed-pair delay audits from frozen scores."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .annotation_population import read_jsonl, token_population
from .data import read_json, write_json
from .evaluate import source_interval
from .unit_data import read_tokens


def ranking(labels, scores, weights=None):
    labels = np.asarray(labels, bool)
    return dict(positives=int(labels.sum()), negatives=int((~labels).sum()),
        auroc=float(roc_auc_score(labels, scores, sample_weight=weights)) if labels.any() and (~labels).any() else None,
        ap=float(average_precision_score(labels, scores, sample_weight=weights)) if labels.any() else None)


def token_rankings(table, methods):
    common = np.isfinite(table[methods]).all(axis=1)
    error = table.gold.eq(1)
    first = error & table.span_index.eq(0) & table.offset.eq(0)
    onset = error & table.offset.eq(0)
    populations = dict(all=np.ones(len(table), bool), first_or_normal=first | ~error,
        onset_or_normal=onset | ~error, continuation_or_normal=(error & ~onset) | ~error)
    weights = pd.Series(1., index=table.index)
    weights.loc[error] = 1 / table.loc[error, "span_length"]
    if (error & common).any():
        weights.loc[error & common] *= (error & common).sum() / weights.loc[error & common].sum()
    rows = []
    for population, mask in populations.items():
        group = table[common & mask]
        for method in methods:
            rows.append(dict(population=population, weighting="token", method=method,
                tokens=len(group), sources=group.source_id.nunique(), **ranking(group.gold, group[method])))
    for method in methods:
        group = table[common]
        rows.append(dict(population="all", weighting="positive_span_equal", method=method,
            tokens=len(group), sources=group.source_id.nunique(),
            **ranking(group.gold, group[method], weights.loc[group.index])))
    return pd.DataFrame(rows)


def pool(values):
    """Inputs are only frozen scores; pooling has no access to labels/boundaries."""
    values = np.asarray(values, float)
    return dict(mean=float(values.mean()), maximum=float(values.max()),
                top4_mean=float(np.sort(values)[-4:].mean()))


def sentence_scores(table, methods):
    rows = []
    for (identity, number), group in table[table.sentence.ge(0)].groupby(["id", "sentence"]):
        finite = np.isfinite(group[methods]).all(axis=1)
        if not finite.any():
            continue
        for method in methods:
            for aggregation, score in pool(group.loc[finite, method]).items():
                rows.append(dict(id=identity, source_id=str(group.source_id.iloc[0]), sentence=number,
                    method=method, aggregation=aggregation, score=score, gold=int(group.gold.any()),
                    tokens=len(group), covered_tokens=int(finite.sum()), error_tokens=int(group.gold.sum()),
                    first_error_offset=int(np.flatnonzero(group.gold)[0]) if group.gold.any() else -1,
                    completion_delay_tokens=int(group.token.max() - group.loc[group.gold.eq(1), "token"].min())
                    if group.gold.any() else -1))
    return pd.DataFrame(rows)


def sentence_metrics(sentences, calibration, target_fpr):
    rows = []
    for (method, aggregation), group in sentences.groupby(["method", "aggregation"]):
        normal = calibration[(calibration.method == method) & (calibration.aggregation == aggregation)
                             & calibration.gold.eq(0)] if calibration is not None else None
        threshold = float(normal.score.quantile(1 - target_fpr, interpolation="higher")) if normal is not None and len(normal) else None
        bins = dict(all=np.ones(len(group), bool), short_1_8=group.tokens.le(8),
                    medium_9_32=group.tokens.between(9, 32), long_33_plus=group.tokens.gt(32))
        for length_bin, mask in bins.items():
            selected = group[mask]
            if selected.empty:
                continue
            row = dict(method=method, aggregation=aggregation, length_bin=length_bin,
                sentences=len(selected), sources=selected.source_id.nunique(), threshold=threshold,
                calibration_normal_sentences=len(normal) if normal is not None else 0,
                **ranking(selected.gold, selected.score))
            row.update(alarm_metrics(selected, threshold))
            rows.append(row)
    return pd.DataFrame(rows)


def alarm_metrics(group, threshold):
    if threshold is None:
        return dict(recall=None, fpr=None)
    positive = group.gold.eq(1)
    alarm = group.score > threshold
    return dict(recall=float(alarm[positive].mean()) if positive.any() else None,
                fpr=float(alarm[~positive].mean()) if (~positive).any() else None)


def sentence_length_controls(sentences, bootstrap):
    """Same answer and token-length bin; only mixed cells identify within-cell AUC."""
    cells = []
    sentences = sentences.assign(length_bin=np.floor(np.log2(sentences.tokens)).astype(int))
    keys = ["method", "aggregation", "id", "source_id", "length_bin"]
    for identity, group in sentences.groupby(keys):
        if group.gold.nunique() == 2:
            cells.append(dict(zip(keys, identity), auroc=roc_auc_score(group.gold, group.score),
                              sentences=len(group)))
    frame = pd.DataFrame(cells)
    rows = []
    if not frame.empty:
        for (method, aggregation), group in frame.groupby(["method", "aggregation"]):
            rows.append(dict(method=method, aggregation=aggregation, cells=len(group),
                             sentences=int(group.sentences.sum()), **source_interval(group, "auroc", bootstrap)))
    return frame, pd.DataFrame(rows)


def pair_delays(table, pairs, methods, horizons):
    answers = {str(key): group.set_index("token") for key, group in table.groupby("id")}
    rows = []
    for pair in pairs:
        if pair["tier"] != "cluster":
            continue
        group = answers[str(pair["id"])]
        error = group.loc[np.arange(pair["length"]) + pair["error_start"]]
        normal = group.loc[np.arange(pair["length"]) + pair["normal_start"]]
        if not error.gold.eq(1).all() or not normal.gold.eq(0).all():
            raise ValueError("Locked pair label mismatch")
        for count in horizons:
            if len(error) < count:
                continue
            left, right = error.iloc[:count], normal.iloc[:count]
            if not np.isfinite(left[methods]).all().all() or not np.isfinite(right[methods]).all().all():
                continue
            for method in methods:
                a, b = left[method].to_numpy(), right[method].to_numpy()
                for aggregation, score_a, score_b in (("current", a[-1], b[-1]), ("prefix_mean", a.mean(), b.mean())):
                    rows.append(dict(id=str(pair["id"]), source_id=str(pair["source_id"]),
                        error_start=pair["error_start"], normal_start=pair["normal_start"], length=pair["length"],
                        observed_tokens=count, method=method, aggregation=aggregation,
                        error_score=score_a, normal_score=score_b, gap=score_a - score_b,
                        paired_win=float(score_a > score_b) + .5 * float(score_a == score_b)))
    return pd.DataFrame(rows)


def summarize_delays(frame, horizons, bootstrap):
    if frame.empty:
        return pd.DataFrame()
    keys = ["id", "source_id", "error_start", "normal_start", "method", "aggregation"]
    frame = frame.copy()
    frame["complete_horizons"] = frame.groupby(keys).observed_tokens.transform("nunique") == len(horizons)
    rows = []
    for cohort, data in (("eligible_at_k", frame), ("same_pairs_all_k", frame[frame.complete_horizons])):
        for identity, group in data.groupby(["method", "aggregation", "observed_tokens"]):
            for metric in ("paired_win", "gap"):
                rows.append(dict(method=identity[0], aggregation=identity[1], observed_tokens=identity[2],
                    cohort=cohort, metric=metric, pairs=len(group), **source_interval(group, metric, bootstrap)))
    return pd.DataFrame(rows)


def run(args):
    records = {str(row["id"]): row for row in read_jsonl(args.responses)}
    table = read_tokens(args.tokens, args.methods, records)
    calibration = read_tokens(args.calibration, args.methods, records) if args.calibration else None
    if calibration is not None and set(calibration.source_id) & set(table.source_id):
        raise ValueError("Calibration and test sources overlap")
    args.output.mkdir(parents=True, exist_ok=True)
    token_rankings(table, args.methods).to_csv(args.output / "token_rankings.csv", index=False)
    write_json(args.output / "token_population.json", token_population(table))
    sentences = sentence_scores(table, args.methods)
    controls = sentence_scores(calibration, args.methods) if calibration is not None else None
    sentences.to_csv(args.output / "sentence_scores.csv", index=False)
    measured = sentence_metrics(sentences, controls, args.target_fpr)
    measured.to_csv(args.output / "sentence_metrics.csv", index=False)
    cells, conditional = sentence_length_controls(sentences, args.bootstrap)
    cells.to_csv(args.output / "sentence_length_cells.csv", index=False)
    conditional.to_csv(args.output / "sentence_length_controls.csv", index=False)
    pairs = read_json(args.pairs) if args.pairs else []
    delays = pair_delays(table, pairs, args.methods, args.horizons)
    delays.to_csv(args.output / "paired_delays.csv", index=False)
    summarize_delays(delays, args.horizons, args.bootstrap).to_csv(args.output / "delay_summary.csv", index=False)
    write_json(args.output / "protocol.json", dict(methods=args.methods, refit=False,
        calibration_available=calibration is not None, target_sentence_fpr=args.target_fpr,
        sentences="independent punctuation boundaries; any labeled token makes sentence positive",
        score_coverage="same finite token positions per sentence across methods; label uses full sentence",
        sentence_timing="completed sentence; saved-score causality inherited, not established here",
        paired_delays="gold-window diagnostic only; observed_tokens=1 is the first error token",
        token_reweighting="evaluation only; not a training-loss intervention",
        horizons=args.horizons, population_tokens=len(table),
        sentence_unmapped_tokens=int(table.sentence.lt(0).sum()),
        unmapped_error_tokens=int((table.sentence.lt(0) & table.gold.eq(1)).sum())))
    print(measured[measured.length_bin.eq("all")].to_string(index=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--methods", nargs="+", default=["score"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--target-fpr", type=float, default=.05)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(count < 1 for count in args.horizons) or not 0 < args.target_fpr < 1 or args.bootstrap < 1:
        raise ValueError("Positive horizons/bootstrap and target FPR in (0,1) required")
    run(args)


if __name__ == "__main__":
    main()
