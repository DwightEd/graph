"""Are equally clustered, unlabelled-normal spans distinguishable from errors?"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .features import head_profile, prepare_features, representation_geometry
from .io import (load_capture, load_graph, population, read_comparisons,
                 read_json, read_predictions)
from .matching import TIERS, match_answer
from .reporting import (balance_report, gallery, head_rows, ranking, source_mean,
                        summarize_pairs, token_rows, write_csv, write_json)


DEFAULTS = dict(position_caliper=.25, repeat_caliper=.15, caliper_multiplier=1.,
                head_rms_caliper=.05, head_max_caliper=.25)


def add_identity(sample, rows):
    identity = {key: sample[key] for key in ("id", "source_id", "task", "generator")}
    first = int(np.flatnonzero(sample["gold"])[0]) if sample["gold"].any() else -1
    return [dict(row, **identity, first_in_answer=row["error_start"] == first) for row in rows]


def answer_measurements(sample, data, pairs, states):
    graph = data["graph"]
    profiles, geometry = [], []
    stage_values = {"input": graph["x"][int(graph["prompt_length"]):], **states}
    with np.load(sample["prediction_file"], allow_pickle=False) as saved:
        if "embedding" in saved.files:
            stage_values["final_embedding"] = saved["embedding"]
    for pair in pairs:
        start, normal, length = pair["error_start"], pair["normal_start"], pair["length"]
        identity = {key: pair[key] for key in ("id", "source_id", "tier", "error_start")}
        difference = head_profile(data, start, length) - head_profile(data, normal, length)
        profiles.append(dict(identity, difference=difference))
        for stage, values in stage_values.items():
            geometry.append(dict(identity, stage=stage, **representation_geometry(values, start, normal, length)))
    return profiles, geometry


def process_answers(samples, prepared, captures, config, comparisons):
    all_pairs, status, skipped, profiles, geometry, capture_info = [], [], [], [], [], []
    scores = {}
    head_count = None
    for sample in tqdm(samples, desc="match normal/error clusters", unit="answer"):
        graph, path = load_graph(sample, prepared)
        current_heads = (int(graph["layers"]), int(graph["heads"]))
        if head_count is not None and current_heads != head_count:
            raise ValueError("Head profile aggregation requires a single observer geometry")
        head_count = current_heads
        data = prepare_features(graph)
        pairs, answer_status, answer_skipped = match_answer(sample, data, config)
        pairs = add_identity(sample, pairs)
        extra, states, info = load_capture(sample, path, captures)
        score = {k: v for k, v in sample.items() if k == "score" or k.startswith("score_")}
        for name, values in comparisons.items():
            if sample["id"] in values:
                score[name] = values[sample["id"]]
        score.update(extra)
        scores[sample["id"]] = score
        answer_profiles, answer_geometry = answer_measurements(sample, data, pairs, states)
        all_pairs.extend(pairs)
        status.extend(add_identity(sample, answer_status))
        skipped.extend(add_identity(sample, answer_skipped))
        profiles.extend(answer_profiles)
        geometry.extend(answer_geometry)
        capture_info.append(dict(id=sample["id"], **info))
        del data, graph, states
    return all_pairs, status, skipped, profiles, geometry, scores, capture_info, head_count[1]


def group_results(pairs, scores, threshold, bootstrap, thresholds):
    report, metric_rows, balance = {}, [], []
    for tier in TIERS:
        selected = [r for r in pairs if r["tier"] == tier]
        measured, rows = summarize_pairs(selected, scores, threshold, bootstrap, thresholds)
        measured["by_error_role"] = {}
        for name, first in (("answer_first_span", True), ("later_span", False)):
            subset = [r for r in selected if r["first_in_answer"] == first]
            measured["by_error_role"][name] = summarize_pairs(subset, scores, threshold, 0, thresholds)[0]
        measured["by_task_generator"] = {}
        for task, generator in sorted({(r["task"], r["generator"]) for r in selected}):
            subset = [r for r in selected if (r["task"], r["generator"]) == (task, generator)]
            measured["by_task_generator"][task + "|" + generator] = summarize_pairs(subset, scores, threshold, 0, thresholds)[0]
        report[tier] = measured
        metric_rows.extend(rows)
        balance.extend(dict(row, tier=tier) for row in balance_report(selected))
    return report, metric_rows, balance


def common_cohort(pairs, scores, threshold, bootstrap, thresholds):
    keys = [{(r["id"], r["error_start"]) for r in pairs if r["tier"] == tier} for tier in TIERS]
    common = set.intersection(*keys)
    return dict(error_spans=len(common), tiers={
        tier: summarize_pairs([r for r in pairs if r["tier"] == tier and
                               (r["id"], r["error_start"]) in common],
                              scores, threshold, bootstrap, thresholds)[0]
        for tier in TIERS},
        note="Same error spans across tiers, but normal controls differ. Not a causal decomposition.")


def summary_text(report):
    lines = ["# 结构相近的错误 / 正常片段审计", "",
             "正常=数据集中无幻觉标注。候选不是经语义核验的完整正确事实。配对不读取模型分数或embedding。", "",
             "| 配对层级 | 配对数 | 配对内 token AUROC（宏平均） | 正确聚集片段 token FPR | 错误 token recall |", "|---|---:|---:|---:|---:|"]
    for tier in TIERS:
        row = report["tiers"][tier]
        values = row["pair_macro"]
        def text(key):
            return "缺测" if values[key] is None else f'{values[key]:.4f}'
        lines.append(f'| {tier} | {row["pairs"]} | {text("within_pair_token_auroc")} | {text("normal_cluster_token_fpr")} | {text("error_token_recall")} |')
    lines.extend(["", "先检查 balance.csv 和 unmatched.csv：匹配不到不表示正常聚集不存在；只适用于已配对区域。",
                  "三层级可能覆盖不同的错误span，不把层级间成绩变化直接称为机制贡献。",
                  "每个 control 都在自己的完整共同配对上与原模型比较，缺分数不补零。",
                  "原阈值用于冻结对照；独立重训模型使用各自在校准集保存的阈值。",
                  "head_differences.csv 是逐头关联，不是重要性排序，也不证明模型使用了该关系。",
                  "representation_geometry.csv 比较两类片段都是否聚集；相似度变化本身不是分类增益。",
                  "模型是否依赖邻居/多头组合/中继，要看相同配对下的控制结果和独立重训对照。",
                  "固定模型干预有分布偏移；即使图区分有效，也不等于发现了LLM产生幻觉的原因。"])
    return "\n".join(lines)


def run_one(root, args):
    samples, settings = read_predictions(root, args.completed_only)
    output = Path(root) / args.output_name
    if output.resolve() in (Path(root).resolve(), (Path(root) / "samples").resolve()):
        raise ValueError("Output must be a new audit subdirectory")
    output.mkdir(parents=True, exist_ok=True)
    config = dict(DEFAULTS, position_caliper=args.position_caliper)
    protocol = dict(settings=settings, sample_ids=[s["id"] for s in samples], matching=config,
                    completed_only=args.completed_only, prepared=args.prepared, compare=args.compare,
                    captures=args.captures, bootstrap=args.bootstrap)
    config_file = output / "settings.json"
    if config_file.exists() and read_json(config_file) != protocol:
        raise ValueError("Different audit inputs/settings: choose a new --output-name")
    write_json(config_file, protocol)
    comparisons, thresholds, comparison_info = read_comparisons(args.compare, samples, root)
    captures = args.captures or str(Path(root) / "deep_audit" / "captures")
    values = process_answers(samples, args.prepared, captures, config, comparisons)
    pairs, status, skipped, profiles, geometry, scores, capture_info, heads = values
    threshold = float(settings["threshold"]["value"])
    tiers, rows, balance = group_results(pairs, scores, threshold, args.bootstrap, thresholds)
    error = np.concatenate([s["score"][s["gold"].astype(bool)] for s in samples])
    normal = np.concatenate([s["score"][~s["gold"].astype(bool)] for s in samples])
    report = dict(protocol="label_assisted_posthoc_matched_clusters_v1", settings=protocol,
                  threshold=settings["threshold"], population=population(samples),
                  original_pooled=ranking(error, normal) if len(error) and len(normal) else None,
                  tiers=tiers, captures=capture_info, comparisons=comparison_info,
                  common_error_cohort=common_cohort(pairs, scores, threshold, args.bootstrap, thresholds),
                  interpretation="Matched observational discrimination, not causal or full-benchmark evaluation.")
    write_outputs(output, samples, report, pairs, rows, status, skipped, balance, profiles, geometry, heads, args.bootstrap)
    print(summary_text(report), flush=True)
    print("Saved matched cluster audit:", output, flush=True)
    return report


def write_outputs(output, samples, report, pairs, rows, status, skipped, balance, profiles, geometry, heads, bootstrap):
    write_json(output / "report.json", report)
    write_json(output / "pairs.json", pairs)
    write_csv(output / "pair_scores.csv", rows)
    write_csv(output / "matching_status.csv", status)
    write_csv(output / "unmatched.csv", [r for r in status if not r["matched"]] + skipped)
    write_csv(output / "balance.csv", balance)
    write_csv(output / "head_differences.csv", head_rows(profiles, heads, bootstrap))
    write_csv(output / "representation_geometry.csv", geometry)
    write_csv(output / "tokens.csv.gz", token_rows(samples, pairs, report["threshold"]["value"]))
    gallery(samples, pairs, output / "pairs.html")
    (output / "summary.md").write_text(summary_text(report), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Prediction directory or its parent tree")
    parser.add_argument("--prepared", help="Existing prepared directory, only needed to relocate graph paths")
    parser.add_argument("--output-name", default="cluster_audit")
    parser.add_argument("--variant", default="charm_in", help="Only this original model variant is audited")
    parser.add_argument("--completed-only", action="store_true", help="Preview finalized samples; explicitly incomplete")
    parser.add_argument("--captures", help="Existing deep_audit/captures directory, single prediction root only")
    parser.add_argument("--compare", action="append", default=[], metavar="NAME=PREDICTIONS", help="Existing independent model predictions, single root only")
    parser.add_argument("--position-caliper", type=float, default=.25)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args(argv)
    root = Path(args.root)
    files = [root / "prediction_settings.json"] if (root / "prediction_settings.json").exists() else sorted(root.rglob("prediction_settings.json"))
    roots = [p.parent for p in files if read_json(p)["variant"] == args.variant]
    if not roots:
        raise ValueError("No saved predictions for requested variant: " + args.variant)
    if len(roots) != 1 and (args.compare or args.captures):
        raise ValueError("Explicit captures/comparisons require one prediction directory")
    for path in roots:
        run_one(path, args)


if __name__ == "__main__":
    main()
