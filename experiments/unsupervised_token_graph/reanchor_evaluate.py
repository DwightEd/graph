"""Evaluate saved source-flow measurements; never rerun a scorer on labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import Ranking, label_views, scoped_metrics


SCORES = ("source_mismatch_bits", "permuted_source_mismatch_bits", "event_strength", "prompt_deficit")


def aggregate_channels(values, quantile=.9):
    """Aggregate scalar measurements AFTER per-head computation; report support."""
    valid = np.isfinite(values)
    result = np.full(values.shape[1], np.nan)
    count = valid.sum(axis=0)
    for t in np.flatnonzero(count):
        result[t] = np.quantile(values[:, t][valid[:, t]], quantile, method="higher")
    return result, count


def evaluate(prediction_dir, annotations_path, output=None, split="test", quantile=.9, bootstrap=200):
    root = Path(prediction_dir)
    complete = json.loads((root / "complete.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if not complete["complete"] or summary["labels_read"] or complete["responses"] != len(summary["responses"]):
        raise ValueError("complete label-free predictions are required before evaluation")
    records = [r for r in summary["responses"] if r["split"] == split]
    if not records:
        raise ValueError("no identity-bound records in requested split; use native NPZ identities or the existing population/records index")
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation needs unique response IDs")
    gold = {}
    with Path(annotations_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row["id"]) in ids:
                gold[str(row["id"])] = row
    blocks = []
    for record in records:
        annotation = gold[record["id"]]
        digest = hashlib.sha256(annotation["response"].encode()).hexdigest()
        if (not record["response_sha256"] or digest != record["response_sha256"]
                or str(annotation["source_id"]) != record["source_id"] or annotation["split"] != record["split"]):
            raise ValueError("response/source/split identity mismatch: " + record["id"])
        with np.load(root / record["file"], allow_pickle=False) as arrays:
            if json.loads(str(arrays["record_json"])) != record or "offsets" not in arrays:
                raise ValueError("saved identity and response offsets are required: " + record["id"])
            offsets = arrays["offsets"]
            if str(arrays["cache_format"]) == "canonical_csr" and len(offsets) + int(arrays["prompt_length"]) != int(arrays["total_tokens"]):
                raise ValueError("canonical token count and response offsets disagree")
            if (offsets.ndim != 2 or offsets.shape[1] != 2 or np.any(offsets < 0)
                    or np.any(offsets[:, 1] < offsets[:, 0]) or np.any(offsets[:, 1] > len(annotation["response"]))):
                raise ValueError("invalid response-relative token offsets")
            positions = arrays["prediction_positions"] - int(arrays["prompt_length"])
            usable = (positions >= 0) & (positions < len(offsets))
            scores, channels = {}, {}
            for name in SCORES:
                if name == "prompt_deficit":
                    known = 1 - arrays["unknown_mass"]
                    values = 1 - np.divide(arrays["prompt_reach"], known,
                                           out=np.full_like(known, np.nan), where=known > 1e-12)
                elif name in arrays:
                    values = arrays[name]
                else:
                    continue
                score, support = aggregate_channels(values, quantile)
                scores[name] = np.full(len(offsets), np.nan)
                channels[name] = np.zeros(len(offsets), int)
                scores[name][positions[usable]] = score[usable]
                channels[name][positions[usable]] = support[usable]
            blocks.append(dict(record=record, views=label_views(offsets, annotation["labels"]),
                               scores=scores, channels=channels, tokens=len(offsets)))
    groups = {"ALL": blocks}
    for block in blocks:
        record = block["record"]
        groups.setdefault(record["task"] + "|" + record["generator"], []).append(block)
    report = dict(version=summary["version"], alignment="predict_next: score(q) -> token(q+1)",
                  split=split, channel_quantile=quantile, groups={},
                  interpretation="fixed high-score hypotheses, not truth probabilities; mismatch has conditional coverage")
    for group, selected in groups.items():
        counts = {}
        for b in selected:
            sid = b["record"]["source_id"]
            counts[sid] = counts.get(sid, 0) + b["tokens"]
        views = {}
        for view in selected[0]["views"]:
            metrics = {}
            for name in SCORES:
                available = [b for b in selected if name in b["scores"]]
                if not available:
                    continue
                labels, scores, sources, weights = [], [], [], []
                for b in available:
                    y, mask = b["views"][view]
                    labels.extend(y[mask]); scores.extend(b["scores"][name][mask])
                    sid = b["record"]["source_id"]
                    sources.extend([sid] * int(mask.sum())); weights.extend([1 / counts[sid]] * int(mask.sum()))
                metrics[name] = scoped_metrics(labels, scores, sources, weights, bootstrap)
            views[view] = metrics
        # Real vs rewired must also be compared on their COMMON coverage.
        paired = {}
        for view in selected[0]["views"]:
            y, a, b, sid = [], [], [], []
            for block in selected:
                if "permuted_source_mismatch_bits" not in block["scores"]:
                    continue
                label, mask = block["views"][view]
                real = block["scores"]["source_mismatch_bits"]
                null = block["scores"]["permuted_source_mismatch_bits"]
                common = mask & np.isfinite(real) & np.isfinite(null)
                y.extend(label[common]); a.extend(real[common]); b.extend(null[common])
                sid.extend([block["record"]["source_id"]] * int(common.sum()))
            real, null = Ranking(y, a), Ranking(y, b)
            point_a, point_b = real.measure(), null.measure()
            delta = {k: point_a[k] - point_b[k] if point_a[k] is not None and point_b[k] is not None else None
                     for k in ("auroc", "ap")}
            unique, inv = np.unique(sid, return_inverse=True)
            draws = []; rng = np.random.default_rng(20260915)
            for _ in range(bootstrap if len(unique) > 1 else 0):
                weight = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[inv]
                r, n = real.measure(weight), null.measure(weight)
                if r["auroc"] is not None:
                    draws.append([r["auroc"] - n["auroc"], r["ap"] - n["ap"]])
            paired[view] = dict(tokens=len(y), real=point_a, history_permuted=point_b, delta=delta,
                source_bootstrap_ci95=np.quantile(draws, [.025, .975], axis=0).tolist() if draws else None)
        report["groups"][group] = dict(responses=len(selected), sources=len(counts), views=views,
                                       common_coverage_real_vs_permuted=paired)
    if output is not None:
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", help="RAGTruth response.jsonl; defaults to saved population settings")
    parser.add_argument("--if-available", action="store_true", help="skip only when no annotation path is configured")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--channel-quantile", type=float, default=.9)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args(argv)
    if not args.annotations:
        settings = json.loads((Path(args.predictions) / "settings.json").read_text())
        args.annotations = settings.get("annotations")
    if not args.annotations:
        if args.if_available:
            print("Evaluation skipped: no ANNOTATIONS or population dataset path; graph results are saved.")
            return
        parser.error("set --annotations, or use an existing population with settings.json during analysis")
    result = evaluate(args.predictions, args.annotations, args.output, args.split, args.channel_quantile, args.bootstrap)
    for group, values in result["groups"].items():
        for name, metric in values["views"]["all_error"].items():
            print(json.dumps(dict(group=group, score=name, coverage=metric["coverage"], **metric["pooled"])))


if __name__ == "__main__":
    main()
