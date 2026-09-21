"""Only this module reads natural hallucination labels, after score freeze."""

import json

import numpy as np

from ..evaluate import Ranking, label_views, scoped_metrics
from ..evaluation_data import EvaluationBinding, read_sources
from ..fixed_graph.evaluation import interval_report, metric_arrays, token_spans
from .pipeline import read_json
from .reporting import evaluation_groups, group_identity, save_reports


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
                window_count=saved["window_count"].copy(), window=window,
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


def availability_report(blocks, method):
    counts = np.concatenate([block["window_count"][np.isfinite(block["scores"][method])]
                             for block in blocks])
    values, frequencies = np.unique(counts, return_counts=True)
    return dict(tokens=sum(block["tokens"] for block in blocks), scored_tokens=len(counts),
                configured_window=blocks[0]["window"],
                full_window_tokens=int((counts == blocks[0]["window"]).sum()),
                scored_window_lengths={str(value): int(count) for value, count in zip(values, frequencies)})


def add_span_halves(blocks):
    for block in blocks:
        error = block["views"]["all_error"][0]
        front, back = np.zeros(len(error), bool), np.zeros(len(error), bool)
        for start, end in block["gold"]:
            middle = start + (end - start + 1) // 2
            front[start:middle] = True
            back[middle:end] = True
        block["views"]["front_half_vs_normal"] = (front, front | ~error)
        block["views"]["back_half_vs_normal"] = (back, back | ~error)


def evaluate_group(blocks, methods, draws, primary):
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
    controls = [name for name in methods if name.startswith("all__") and name != primary]
    for view in ("all_error", "first_error_until_first", "span_onset_vs_normal",
                 "continuation_vs_normal", "previous_error", "front_half_vs_normal", "back_half_vs_normal"):
        differences[view] = {}
        for control in controls:
            differences[view][control] = paired_difference(
                blocks, view, primary, control, draws)
    return dict(answers=len(blocks), views=views, primary_minus_control=differences,
                availability=availability_report(blocks, primary),
                spans={name: interval_report(blocks, name) for name in methods})


def evaluate(args):
    blocks, freeze = read_blocks(args)
    add_span_halves(blocks)
    groups = evaluation_groups(blocks)
    identities = {name: group_identity(selected) for name, selected in groups.items()}
    channels = np.asarray(read_json(args.output / "observations/manifest.json")["channels"])
    report = dict(primary=freeze["primary"], labels_used_for_fit=False,
                  score_direction="higher = departure from mixed TRAIN; energy scores use standardized residuals",
                  threshold="unlabelled mixed calibration quantile, not controlled normal FPR",
                  evaluation_status="exploratory; representation motivated by earlier labelled audits",
                  dataset="RAGTruth",
                  head_selection=dict(layers=np.unique(channels[:, 0]).tolist(),
                                      heads=np.unique(channels[:, 1]).tolist()), groups={})
    computed = {}
    for group, selected in groups.items():
        # A one-task/one-generator run has identical ALL, task and generator groups.
        signature = tuple(id(block) for block in selected)
        if signature not in computed:
            computed[signature] = evaluate_group(selected, freeze["methods"], args.bootstrap, freeze["primary"])
        result = computed[signature]
        report["groups"][group] = result
        if "|" not in group:
            continue
        for name in freeze["methods"]:
            metric = result["views"]["all_error"][name]
            print(json.dumps(dict(identities[group], group=group, method=name, tokens=metric["evaluated_tokens"],
                  coverage=metric["coverage"], **metric["pooled"])), flush=True)
    save_reports(args.output / "predictions", report, identities)
    return report
