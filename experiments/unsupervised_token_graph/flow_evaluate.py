"""Evaluation-only label join with exact response-token alignment and tie handling."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _metrics(y, s, w=None):
    result = {"tokens": len(y), "positives": int(np.sum(y)), "auroc": None, "ap": None}
    if len(np.unique(y)) == 2:
        result.update(auroc=float(roc_auc_score(y, s, sample_weight=w)),
                      ap=float(average_precision_score(y, s, sample_weight=w)))
    return result


def paired_source_difference(graph, uniform):
    """Bootstrap paired source differences, preserving both methods' dependence."""
    sources = sorted(set(graph) & set(uniform))
    difference = np.array([graph[s] - uniform[s] for s in sources])
    result = {"eligible_sources": len(sources), "graph_minus_uniform": float(difference.mean()) if len(sources) else None,
              "bootstrap95": None, "resamples": 1000, "seed": 20260914}
    if len(sources) >= 2:
        rng = np.random.default_rng(20260914)
        draws = rng.choice(difference, size=(1000, len(sources)), replace=True).mean(axis=1)
        result["bootstrap95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    return result


def evaluate_frozen(output, annotations):
    from .flow_run import _hash, _write, verify_freeze
    verify_freeze(output)
    output = Path(output)
    if (output / "evaluation.json").exists():
        raise FileExistsError("evaluation already exists; preserve original results")
    predictions = json.loads((output / "scores.json").read_text())
    ids = {r["response_id"] for r in predictions["graph"]}
    labels, identities = {}, {}
    with open(annotations, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rid = str(row["id"])
            if rid not in ids:
                continue
            if rid in labels:
                raise ValueError("duplicate annotation ID")
            offsets = np.asarray(row["offsets"])
            if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.dtype.kind not in "iu" or np.any(offsets < 0) or np.any(offsets[:, 1] < offsets[:, 0]):
                raise ValueError("invalid response-relative character offsets")
            target = np.where(offsets[:, 1] > offsets[:, 0], 0, -1)
            for span in row["labels"]:
                if not 0 <= span["start"] < span["end"]:
                    raise ValueError("invalid annotation span")
                target[(offsets[:, 1] > offsets[:, 0]) & (offsets[:, 0] < span["end"]) & (offsets[:, 1] > span["start"])] = 1
            labels[rid] = target
            identities[rid] = row.get("token_ids")
    if set(labels) != ids:
        raise ValueError("missing annotations; no silent coverage filtering")
    report = {"annotations_sha256": _hash(annotations), "freeze_sha256": _hash(output / "freeze.json"),
              "coverage": float(sum(np.sum(y >= 0) for y in labels.values()) / sum(len(y) for y in labels.values())),
              "unavailable_tokens": {rid: int(np.sum(y < 0)) for rid, y in labels.items()},
              "coverage_scope": "zero-length offsets unavailable for lexical labels; no unknown token is labeled normal",
              "methods": {}}
    for mode, rows in predictions.items():
        if {r["response_id"] for r in rows} != ids or len(rows) != len(ids):
            raise ValueError("method rosters differ")
        counts = {}
        for row in rows:
            counts[row["source_id"]] = counts.get(row["source_id"], 0) + 1
        streams = {name: [] for name in ("all", "through_first_error", "strict_post_first")}
        within, any_alarm, normal_alarm, onset = {}, {}, {}, {}
        for row in rows:
            y, s = labels[row["response_id"]], np.asarray(row["score"])
            if len(y) != len(s) or not len(y) or len(row["token_indices"]) != len(s) or not np.isfinite(s).all():
                raise ValueError("token alignment/score mismatch; never infer alignment by tail slicing")
            token_ids = row.get("token_ids")
            if token_ids is None or identities[row["response_id"]] != token_ids:
                raise ValueError("evaluation requires exact frozen observer response token_ids")
            index = np.arange(len(y)); positives = np.flatnonzero(y == 1)
            valid = y >= 0
            first = int(positives[0]) if len(positives) else len(y)
            masks = {"all": valid, "through_first_error": valid & (index <= first),
                     "strict_post_first": (valid & (index > first)) if len(positives) else np.zeros(len(y), bool)}
            for name, mask in masks.items():
                streams[name].append((y[mask], s[mask], row["source_id"]))
            if len(positives):
                before = s[valid & (index <= first)]
                midrank = 1 + np.sum(before > s[first]) + 0.5 * (np.sum(before == s[first]) - 1)
                onset.setdefault(row["source_id"], []).append(dict(
                    midrank_percentile=float(midrank / len(before)), reciprocal_midrank=float(1 / midrank),
                    alarm_before_first=bool(np.any(np.asarray(row["alarm"])[valid & (index < first)]))))
            local = _metrics(y[valid], s[valid])
            if local["auroc"] is not None:
                within.setdefault(row["source_id"], []).append(local["auroc"])
            alarm = bool(np.any(row["alarm"]))
            any_alarm[row["source_id"]] = any_alarm.get(row["source_id"], False) or alarm
            normal_alarm.setdefault(row["source_id"], []).append((bool(valid.all()) and not bool((y == 1).any()), alarm))
        summaries = {}
        for name, parts in streams.items():
            eligible = {}
            for y_part, _, source in parts:
                if len(y_part):
                    eligible[source] = eligible.get(source, 0) + 1
            y, s = (np.concatenate([p[i] for p in parts]) for i in range(2))
            w = np.concatenate([np.full(len(yy), 1. / (len(eligible) * eligible[source] * len(yy)))
                                if len(yy) else np.array([]) for yy, _, source in parts])
            summaries[name] = {"pooled": _metrics(y, s), "source_answer_token_weighted": _metrics(y, s, w)}
        normals = [any(a for _, a in pairs) for pairs in normal_alarm.values() if all(n for n, _ in pairs)]
        summaries.update(within_answer_source_macro_auroc=float(np.mean([np.mean(v) for v in within.values()])) if within else None,
                         within_answer_eligible_sources=len(within), source_any_alarm=float(np.mean(list(any_alarm.values()))),
                         normal_source_any_alarm=float(np.mean(normals)) if normals else None,
                         normal_sources=len(normals))
        summaries["first_error_localization"] = {
            key: float(np.mean([np.mean([v[key] for v in values]) for values in onset.values()])) if onset else None
            for key in ("midrank_percentile", "reciprocal_midrank", "alarm_before_first")}
        summaries["first_error_localization"].update(eligible_sources=len(onset),
                                                    interpretation="post-observation; candidates from response start through first error")
        source_auc = np.array([np.mean(v) for v in within.values()])
        if len(source_auc) >= 2:
            rng = np.random.default_rng(20260914)
            draws = rng.choice(source_auc, size=(1000, len(source_auc)), replace=True).mean(axis=1)
            summaries["within_answer_source_macro_auroc_bootstrap95"] = np.quantile(draws, [0.025, 0.975]).tolist()
        else:
            summaries["within_answer_source_macro_auroc_bootstrap95"] = None
        summaries["per_source"] = {
            "within_answer_auroc": {source: float(np.mean(v)) for source, v in within.items()},
            "first_error_midrank_percentile": {source: float(np.mean([v["midrank_percentile"] for v in values]))
                                               for source, values in onset.items()}}
        report["methods"][mode] = summaries
    graph = report["methods"]["graph"]["per_source"]
    uniform = report["methods"]["mass_matched_uniform"]["per_source"]
    report["paired_graph_vs_uniform"] = {
        metric: paired_source_difference(graph[metric], uniform[metric])
        for metric in ("first_error_midrank_percentile", "within_answer_auroc")}
    report["paired_graph_vs_uniform"]["direction"] = "negative first-error percentile; positive within-answer AUROC"
    _write(output / "evaluation.json", report)
    return report
