"""Audit continuity on frozen predictions; no model, reference refit or new thresholds."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ..evaluate import Ranking, scoped_metrics
from ..fixed_graph.evaluation import metric_arrays
from .continuity_decomposition import decompose_difference
from .continuity_matching import match_spans
from .continuity_metrics import (
    annotation_summary,
    answer_rankings,
    merge_token_spans,
    offset_profile,
    paired_span_rows,
    span_rows,
    summarize_spans,
    within_answer_difference,
    within_answer_summary,
)
from .continuity_order import order_audit
from .continuity_reporting import save_audit
from .evaluation import add_span_halves, alarm_metrics, paired_difference, read_blocks
from .pipeline import read_json
from .reporting import evaluation_groups, group_identity

VIEWS = ("all_error", "first_error_until_first", "span_onset_vs_normal",
         "continuation_vs_normal", "front_half_vs_normal", "back_half_vs_normal",
         "full_window", "warmup", "matched_spans")
PAIRS = (("pair_state_smooth", "pair_state"), ("pair_full", "pair_diagonal"),
         ("pair_full", "pair_persistence"), ("pair_diagonal", "pair_state"),
         ("pair_full", "pair_full_w1"), ("moment", "moment_diagonal"), ("moment", "raw"))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/head_cross_terms_v1"))
    parser.add_argument("--dataset", type=Path, help="defaults to the saved run's dataset")
    parser.add_argument("--source-info", type=Path)
    parser.add_argument("--tokenizer", help="only needed to recover exact missing offsets")
    parser.add_argument("--methods", nargs="+", help="frozen method names; default: all__ methods")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--neighborhood", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if min(args.bootstrap, args.permutations, args.neighborhood) < 0 or args.threads < 1:
        parser.error("bootstrap/permutations/neighborhood must be nonnegative; threads must be positive")
    return args


def load_frozen(args):
    settings = read_json(args.output / "settings.json")
    inputs = SimpleNamespace(
        output=args.output, dataset=args.dataset or Path(settings["dataset"]),
        source_info=args.source_info or settings["source_info"],
        tokenizer=args.tokenizer or settings["tokenizer"],
    )
    blocks, freeze = read_blocks(inputs, include_context=True)
    methods = args.methods or [name for name in freeze["methods"] if name.startswith("all__")]
    unknown = set(methods) - set(freeze["methods"])
    if unknown:
        raise ValueError("Methods were not frozen: " + ", ".join(sorted(unknown)))
    return blocks, freeze, methods, settings


def normalize_annotations(block):
    result = dict(block, gold=merge_token_spans(block["gold"]), views=dict(block["views"]))
    error = result["views"]["all_error"][0]
    onset = np.zeros(len(error), bool)
    for start, _ in result["gold"]:
        onset[start] = True
    result["views"]["span_onset_full_stream"] = (onset, np.ones(len(error), bool))
    result["views"]["span_onset_vs_normal"] = (onset, onset | ~error)
    result["views"]["continuation_vs_normal"] = (error & ~onset, ~onset)
    return result


def prepare_controls(blocks, methods, neighborhood):
    prepared, matching = [], []
    for original in tqdm(blocks, desc="match normal spans", unit="answer"):
        block = normalize_annotations(original)
        valid = block["offsets"][:, 1] > block["offsets"][:, 0]
        finite = np.logical_and.reduce([np.isfinite(block["scores"][name]) for name in methods])
        block["common_finite"] = finite & valid
        block["scores"] = {name: np.where(finite & valid, block["scores"][name], np.nan)
                           for name in methods}
        pairs, unmatched, diagnostics = match_spans(block, finite & valid, neighborhood=neighborhood)
        block["pairs"] = pairs
        eligible = np.zeros(block["tokens"], bool)
        for pair in pairs:
            eligible[pair["error_start"]:pair["error_end"]] = True
            eligible[pair["normal_start"]:pair["normal_end"]] = True
        block["views"]["matched_spans"] = (block["views"]["all_error"][0], eligible)
        prepared.append(block)
        matching.append({"record": block["record"], "pairs": pairs, "unmatched": unmatched,
                             "diagnostics": diagnostics})
    add_span_halves(prepared)
    return prepared, matching


def rank_view(blocks, method, view):
    metric = scoped_metrics(*metric_arrays(blocks, view, method), bootstrap=0)
    metric.update(alarm_metrics(blocks, view, method))
    rows = answer_rankings(blocks, method, view)
    within = within_answer_summary(rows)
    positives = metric["evaluated_positives"]
    total_pairs = positives * (metric["evaluated_tokens"] - positives)
    cross_pairs = total_pairs - within["within_answer_pairs"]
    cross_auc = None
    if cross_pairs and metric["pooled"]["auroc"] is not None:
        within_correct = sum(row["pairs"] * row["auroc"] for row in rows if row["pairs"])
        cross_auc = (metric["pooled"]["auroc"] * total_pairs - within_correct) / cross_pairs
    metric.update(within=within, cross_answer_auroc=cross_auc, cross_answer_pairs=cross_pairs)
    return metric, rows


def span_balanced_auc(blocks, method):
    """Each observed gold span gets positive weight one; negative tokens keep weight one."""
    labels, scores, weights = [], [], []
    for block in blocks:
        value = block["scores"][method]
        finite = np.isfinite(value)
        weight = np.ones(len(value))
        for start, end in block["gold"]:
            count = int(finite[start:end].sum())
            if count:
                weight[start:end] = 1. / count
        labels.extend(block["views"]["all_error"][0][finite])
        scores.extend(value[finite])
        weights.extend(weight[finite])
    return Ranking(labels, scores).measure(weights)["auroc"]


def measure_methods(blocks, methods):
    result, answer_rows, all_spans, matched, profiles = {}, [], [], [], []
    for method in tqdm(methods, desc="continuity metrics", leave=False):
        views = {}
        for view in VIEWS:
            views[view], rows = rank_view(blocks, method, view)
            answer_rows.extend(rows)
        spans = span_rows(blocks, method)
        pairs = [row for block in blocks for row in paired_span_rows(block, method, block["pairs"])]
        result[method] = {"views": views, "spans": summarize_spans(spans),
                             "span_balanced_auroc": span_balanced_auc(blocks, method),
                             "matched": {side: summarize_spans([row for row in pairs if row["side"] == side])
                                      for side in ("error", "normal")}}
        all_spans.extend(spans)
        matched.extend(pairs)
        profiles.extend(comparison_profiles(blocks, method))
    return result, {"answers": answer_rows, "spans": all_spans, "matched": matched, "profiles": profiles}


def comparison_profiles(blocks, method):
    selections = {"all_error": blocks}
    for side in ("error", "normal"):
        selections["matched_" + side] = [
            dict(block, gold=[(pair[side + "_start"], pair[side + "_end"])
                              for pair in block["pairs"]]) for block in blocks
        ]
    rows = []
    for name, selected in selections.items():
        curves = offset_profile(selected, method)
        rows.extend(dict(row, selection=name) for row in curves["offsets"] + curves["phases"])
    return rows


def compare_methods(blocks, methods, draws, seed):
    result, unavailable = {}, []
    for first, second in tqdm(PAIRS, desc="source paired comparisons", leave=False):
        left, right = "all__" + first, "all__" + second
        key = first + "_minus_" + second
        if left not in methods or right not in methods:
            unavailable.append({"left": left, "right": right, "reason": "not_in_selected_frozen_methods"})
            continue
        pooled, within = {}, {}
        for view in ("all_error", "span_onset_vs_normal", "continuation_vs_normal", "matched_spans"):
            pooled[view] = paired_difference(blocks, view, left, right, draws)
            within[view] = within_answer_difference(blocks, view, left, right, draws)
        result[key] = {"left": left, "right": right, "pooled": pooled, "within": within,
                          "decomposition": decompose_difference(blocks, left, right, draws, seed)}
    return result, unavailable


def audit_group(blocks, original, methods, args):
    metrics, tables = measure_methods(blocks, methods)
    comparisons, unavailable = compare_methods(blocks, methods, args.bootstrap, args.seed)
    base = "all__pair_state" if "all__pair_state" in methods else "all__raw"
    order = order_audit(blocks, base, blocks[0]["window"], args.permutations, args.seed) if base in methods else None
    coverage = {name: sum(int(np.isfinite(block["scores"][name]).sum()) for block in original)
                for name in methods}
    result = {"identity": group_identity(blocks), "annotation": annotation_summary(blocks, methods[0]),
                  "official_annotation_spans": sum(block["annotation_spans"] for block in blocks),
                  "original_scored_tokens": coverage, "methods": metrics, "comparisons": comparisons,
                  "unavailable_comparisons": unavailable, "order_null": order}
    return result, tables


def audit(args):
    original, freeze, methods, settings = load_frozen(args)
    blocks, matching = prepare_controls(original, methods, args.neighborhood)
    groups, tables = {}, {}
    original_groups = evaluation_groups(original)
    selected_groups = {key: value for key, value in evaluation_groups(blocks).items() if "|" in key}
    for group, selected in tqdm(selected_groups.items(), desc="audit task/generator", unit="group"):
        groups[group], tables[group] = audit_group(selected, original_groups[group], methods, args)
        print(json.dumps(dict(group=group, **groups[group]["annotation"])), flush=True)
    report = {
        "purpose": "label_assisted_continuity_audit_not_new_detector",
        "natural_labels_used_for_fit": False, "input_primary": freeze["primary"], "methods": methods,
        "threshold": "existing mixed unlabelled calibration; not a controlled normal FPR",
        "statistical_status": "exploratory_source_bootstrap_uncorrected_for_multiple_comparisons",
        "coverage": "intersection of selected methods and positive-width response offsets",
        "gold_unit": "overlapping token annotations merged; adjacent annotations remain distinct",
        "bootstrap": args.bootstrap, "permutations": args.permutations, "seed": args.seed,
        "normal_neighborhood": args.neighborhood, "groups": groups,
    }
    archive = save_audit(args.output, report, tables, matching, blocks, settings)
    print(json.dumps({"archive": str(archive), "report": str(args.output / "continuity/summary.md")}), flush=True)
    return report


def main(argv=None):
    args = arguments(argv)
    with threadpool_limits(limits=args.threads):
        return audit(args)


if __name__ == "__main__":
    main()
