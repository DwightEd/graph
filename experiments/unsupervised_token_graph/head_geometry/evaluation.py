"""Only this module reads natural hallucination labels, after score freeze."""

from collections import defaultdict
import csv
import json

import numpy as np

from ..evaluate import Ranking, label_views, scoped_metrics
from ..evaluation_data import EvaluationBinding, read_sources
from ..fixed_graph.evaluation import interval_report, metric_arrays, token_spans
from ..offline_span.data import write_json
from .pipeline import read_json


def read_blocks(args):
    root = args.output / "predictions"
    freeze = read_json(root / "freeze.json")
    if not freeze["complete"] or freeze["labels_used"]:
        raise ValueError("Freeze complete label-free scores before reading annotations")
    with (args.dataset / "response.jsonl").open(encoding="utf-8") as stream:
        annotations = {str(row["id"]): row for row in map(json.loads, stream)}
    sources, _ = read_sources(args.dataset / "response.jsonl", args.source_info)
    binder = EvaluationBinding(dict(cache="/"), args.tokenizer)
    window = read_json(args.output / "settings.json")["window"]
    blocks = []
    for row in freeze["records"]:
        annotation = annotations[row["id"]]
        with np.load(root / row["file"], allow_pickle=False) as saved:
            identity, offsets = binder.bind(row, annotation, saved, sources)
            views = label_views(offsets, annotation["labels"])
            error = views["all_error"][0]
            previous = np.r_[False, error[:-1]]
            views["previous_normal"] = (error, ~previous)
            views["previous_error"] = (error, previous)
            views["full_window"] = (error, saved["window_count"] == window)
            views["warmup"] = (error, saved["window_count"] < window)
            names = [*freeze["methods"], "position"]
            blocks.append(dict(record=identity, views=views, tokens=len(offsets),
                scores={name: saved[name].copy() for name in names},
                alarms={name: saved[name + "__alarm"].copy() for name in freeze["methods"]},
                spans={name: saved[name + "__spans"].copy() for name in freeze["methods"]},
                gold=token_spans(offsets, annotation["labels"])))
    if not blocks:
        raise ValueError("No frozen TEST predictions")
    return blocks, freeze


def paired_difference(blocks, view, left_name, right_name, draws):
    labels, left, sources, _ = metric_arrays(blocks, view, left_name)
    _, right, _, _ = metric_arrays(blocks, view, right_name)
    covered = np.isfinite(left) & np.isfinite(right)
    first, second = Ranking(labels[covered], left[covered]), Ranking(labels[covered], right[covered])
    a, b = first.measure(), second.measure()
    delta = {key: a[key] - b[key] if a[key] is not None else None for key in ("auroc", "ap")}
    unique, inverse = np.unique(sources[covered], return_inverse=True)
    random, samples = np.random.default_rng(17), []
    for _ in range(draws if len(unique) > 1 else 0):
        counts = np.bincount(random.integers(len(unique), size=len(unique)), minlength=len(unique))
        a, b = first.measure(counts[inverse]), second.measure(counts[inverse])
        if a["auroc"] is not None:
            samples.append([a["auroc"] - b["auroc"], a["ap"] - b["ap"]])
    return dict(tokens=int(covered.sum()), delta=delta, order=["auroc", "ap"],
                bootstrap_valid=len(samples),
                ci95=np.quantile(samples, [.025, .975], axis=0).tolist() if samples else None)


def alarm_metrics(blocks, view, method):
    true_positive = false_positive = positive = negative = 0
    for block in blocks:
        labels, eligible = block["views"][view]
        valid = eligible & np.isfinite(block["scores"][method])
        alarm = block["alarms"][method]
        true_positive += int((alarm & labels & valid).sum())
        false_positive += int((alarm & ~labels & valid).sum())
        positive += int((labels & valid).sum())
        negative += int((~labels & valid).sum())
    return dict(recall=true_positive / positive if positive else None,
                fpr=false_positive / negative if negative else None)


def evaluate_group(blocks, methods, draws):
    views = {}
    for view in blocks[0]["views"]:
        views[view] = {}
        for name in [*methods, "position"]:
            arrays = metric_arrays(blocks, view, name)
            metric = scoped_metrics(*arrays, bootstrap=0)
            if name != "position":
                metric.update(alarm_metrics(blocks, view, name))
            views[view][name] = metric
    differences = {}
    for view in ("all_error", "first_error_until_first", "continuation_vs_normal", "previous_error"):
        differences[view] = {}
        for control in ("raw", "contrast", "moment", "log_diagonal"):
            differences[view][control] = paired_difference(
                blocks, view, "all__log_moment", "all__" + control, draws)
    return dict(answers=len(blocks), views=views, primary_minus_control=differences,
                spans={name: interval_report(blocks, name) for name in methods})


def save_table(path, groups):
    fields = ("group", "scope", "method", "view", "tokens", "positives", "coverage",
              "auroc", "ap", "source_weighted_auroc", "recall", "fpr")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group, result in groups.items():
            for view, methods in result["views"].items():
                for name, metric in methods.items():
                    scope, method = name.split("__") if "__" in name else ("all", name)
                    writer.writerow(dict(group=group, scope=scope, method=method, view=view,
                        tokens=metric["evaluated_tokens"], positives=metric["evaluated_positives"],
                        coverage=metric["coverage"], auroc=metric["pooled"]["auroc"],
                        ap=metric["pooled"]["ap"],
                        source_weighted_auroc=metric["source_fixed_full_answer"]["auroc"],
                        recall=metric.get("recall"), fpr=metric.get("fpr")))


def evaluate(args):
    blocks, freeze = read_blocks(args)
    groups = defaultdict(list)
    groups["ALL"] = blocks
    for block in blocks:
        row = block["record"]
        groups[row["task"] + "|" + row["generator"]].append(block)
    report = dict(primary=freeze["primary"], labels_used_for_fit=False,
                  score_direction="higher = rarer head configuration in mixed TRAIN",
                  threshold="unlabelled mixed calibration quantile, not controlled normal FPR",
                  evaluation_status="exploratory; representation motivated by earlier labelled audits",
                  groups={})
    for group, selected in groups.items():
        result = evaluate_group(selected, freeze["methods"], args.bootstrap)
        report["groups"][group] = result
        for name in freeze["methods"]:
            metric = result["views"]["all_error"][name]
            print(json.dumps(dict(group=group, method=name, tokens=metric["evaluated_tokens"],
                  coverage=metric["coverage"], **metric["pooled"])), flush=True)
    write_json(args.output / "predictions/evaluation.json", report)
    save_table(args.output / "predictions/metrics.csv", report["groups"])
    return report
