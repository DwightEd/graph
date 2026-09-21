"""Natural-label evaluation only after priors and all scores are frozen."""

from collections import defaultdict
import json

from ..evaluate import scoped_metrics
from ..fixed_graph.evaluation import interval_report, metric_arrays
from ..head_geometry.evaluation import (
    add_span_halves, alarm_metrics, availability_report, paired_difference, read_blocks, save_table,
)
from ..offline_span.data import write_json
from .pipeline import PRIMARY


def evaluate_group(blocks, methods, draws):
    views = {}
    for view in blocks[0]["views"]:
        views[view] = {}
        for name in [*methods, "position"]:
            metric = scoped_metrics(*metric_arrays(blocks, view, name), bootstrap=0)
            if name != "position":
                metric.update(alarm_metrics(blocks, view, name))
            views[view][name] = metric
    differences = {}
    comparisons = [name for name in methods if name.startswith("contrast__") and name != PRIMARY]
    comparisons.append("raw__all")
    for view in ("all_error", "first_error_until_first", "continuation_vs_normal", "previous_error",
                 "front_half_vs_normal", "back_half_vs_normal"):
        differences[view] = {control: paired_difference(blocks, view, PRIMARY, control, draws)
                             for control in comparisons}
    return dict(answers=len(blocks), views=views, primary_minus_control=differences,
                availability=availability_report(blocks, PRIMARY),
                symbolic_minus_random=symbolic_comparisons(blocks, methods, draws),
                spans={name: interval_report(blocks, name) for name in methods})


def symbolic_comparisons(blocks, methods, draws):
    controls = [name for name in methods if name.startswith("contrast__symbolic_random_")]
    return {view: {name: paired_difference(blocks, view, "contrast__drop_symbolic", name, draws)
                   for name in controls}
            for view in ("all_error", "span_onset_vs_normal", "continuation_vs_normal", "previous_error")}


def evaluate(args):
    blocks, freeze = read_blocks(args)
    add_span_halves(blocks)
    groups = defaultdict(list)
    groups["ALL"] = blocks
    for block in blocks:
        row = block["record"]
        groups[row["task"] + "|" + row["generator"]].append(block)
    report = dict(primary=PRIMARY, labels_used_for_fit=False, groups={},
        score_direction="higher = rarer configuration; this hypothesis is unchanged",
        threshold="mixed unlabelled calibration quantile; not normal-only FPR",
        interpretation="head prior ablation, not proven noise removal or factual binding detection")
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
