"""One foreground entry: saved scores, frozen graph controls, and diagnostic probes."""

import argparse
import json
from pathlib import Path

import numpy as np

from .common import load_predictions, metric, write_json
from .scores import analyze_scores, control_comparison
from .learned import attach_learned_controls


def prediction_roots(root):
    root = Path(root)
    if (root / "prediction_settings.json").is_file():
        return [root]
    result = sorted(path.parent for path in root.rglob("prediction_settings.json"))
    if not result:
        raise FileNotFoundError("No CHARM prediction_settings.json under " + str(root))
    return result


def prepared_inventory(samples, prepared=None):
    """Read label members only; no attention arrays, checkpoint or tokenizer."""
    if prepared is None:
        prepared = Path(samples[0]["graph"]).parents[2]
    prepared = Path(prepared)
    index_path = prepared / "index.json"
    if not index_path.exists():
        return dict(status="unavailable", reason="No prepared index; the score report still counts all evaluated tokens.")
    records = json.loads(index_path.read_text())
    groups, sources = {}, {}
    for record in records:
        path = prepared / "graphs" / record["split"] / (str(record["id"]) + ".npz")
        with np.load(path, allow_pickle=False) as saved:
            gold, onset, spans = saved["gold"], saved["onset"], saved["spans"]
        key = "|".join((record["split"], record["task"], record["generator"]))
        group = groups.setdefault(key, dict(answers=0, tokens=0, error_tokens=0, first_error_tokens=0,
                                           unique_onset_tokens=0, raw_annotation_spans=0))
        group["answers"] += 1
        group["tokens"] += len(gold)
        group["error_tokens"] += int(gold.sum())
        group["first_error_tokens"] += int(gold.any())
        group["unique_onset_tokens"] += int(onset.sum())
        group["raw_annotation_spans"] += len(spans)
        sources.setdefault(record["split"], set()).add(str(record["source_id"]))
    for group in groups.values():
        group["error_fraction"] = group["error_tokens"] / group["tokens"]
        group["first_error_fraction_all"] = group["first_error_tokens"] / group["tokens"]
        group["continuation_tokens"] = group["error_tokens"] - group["unique_onset_tokens"]
    return dict(status="completed", scope="prepared cohort, not every RAGTruth model or response",
                groups=groups, official_source_overlap=sorted(sources.get("train", set()) & sources.get("test", set())),
                evaluated_answer_ids=[s["id"] for s in samples])


def paired_interventions(samples, threshold, repeats, bootstrap):
    result = {}
    for repeat in range(repeats):
        coupled = [dict(s, score=s[f"score_coupled_{repeat}"]) for s in samples]
        random_cut = [dict(s, score=s[f"score_matched_random_cut_{repeat}"]) for s in samples]
        result[f"independent_minus_coupled_{repeat}"] = control_comparison(
            coupled, f"score_independent_{repeat}", threshold, bootstrap)
        result[f"oracle_minus_matched_random_{repeat}"] = control_comparison(
            random_cut, "score_oracle_span_cut", threshold, bootstrap)
    return result


def align_alternative(base, alternative, key):
    lookup = {s["id"]: s for s in alternative}
    if set(lookup) != {s["id"] for s in base}:
        raise ValueError("Independently trained models evaluated different answer sets")
    paired = []
    for sample in base:
        other = lookup[sample["id"]]
        for name in ("gold", "offsets", "spans", "source_id", "response"):
            if not np.array_equal(sample[name], other[name]):
                raise ValueError("Model comparison alignment differs: " + name)
        paired.append(dict(sample, **{key: other["score"]}))
    return paired


def compare_retrained(samples, settings, root, bootstrap):
    """Use existing, independently fitted siblings; do not silently train new models."""
    training_file = root.parent / "training.json"
    if root.name != "test" or not training_file.exists():
        return dict(status="unavailable", reason="No identifiable matched retrained siblings.")
    training = json.loads(training_file.read_text())
    result = {}
    for directory in sorted(root.parent.parent.iterdir()):
        candidate = directory / "test"
        other_training = directory / "training.json"
        if directory == root.parent or not (candidate / "predictions.json").exists() or not other_training.exists():
            continue
        recipe = json.loads(other_training.read_text())
        original_recipe = {k: v for k, v in training.items() if k != "variant"}
        candidate_recipe = {k: v for k, v in recipe.items() if k != "variant"}
        if original_recipe != candidate_recipe:
            result[directory.name] = dict(status="not_matched", reason="Training recipes, seeds or partitions differ.")
            continue
        alternative, other_settings = load_predictions(candidate)
        paired = align_alternative(samples, alternative, "alternative")
        measured = control_comparison(paired, "alternative", settings["threshold"]["value"], bootstrap)
        gold = np.concatenate([s["gold"] for s in alternative])
        score = np.concatenate([s["score"] for s in alternative])
        measured["own_saved_threshold"] = other_settings["threshold"]
        measured["own_calibrated_metrics"] = metric(gold, score, other_settings["threshold"]["value"])
        measured["alarm_comparison_note"] = "changes uses reference threshold only as sensitivity; compare own_calibrated_metrics for operating-point performance"
        result[directory.name] = measured
    return dict(status="completed" if result else "unavailable", reference=settings["variant"], models=result,
                note="Frozen deletion is not retraining. Prefer charm_in vs node_only/local_in/rewire_in for graph benefit; charm_out may include future structure.")


def format_number(value):
    return "无可用比较" if value is None else f"{value:.6f}"


def write_summary(result, output, graph_status, probe_status):
    population = result["population"]
    all_tokens = result["all_tokens"]
    within = next(row for row in result["within_answers"] if row["role"] == "all_error" and row["negative_context"] == "all_normal")
    lines = ["# 高 AUROC 来源审计", "", "## 先核对分母", "",
             f"本次实际评价 {population['answers']} 个回答、{population['tokens']} 个 token；其中错误 {population['error_tokens']} 个。",
             f"每个回答的首错共有 {population['roles']['first_error']['tokens']} 个；后续 span 起点 {population['roles']['later_onset']['tokens']} 个；span 延续错误 {population['roles']['continuation']['tokens']} 个。",
             "原始标注、重叠合并片段和连续标签段分别统计；不把标注数量当首错数量。", "",
             "## 总体高分是否也能定位到同一回答内", "",
             f"总体 AUROC={format_number(all_tokens['auroc'])}，AP={format_number(all_tokens['ap'])}。",
             f"回答内平均 AUROC={format_number(within['macro_auroc'])}；回答内按正负配对数加权 AUROC={format_number(within['pair_weighted_auroc'])}。",
             f"同回答邻近正常词对比、按 span 等权的 AUROC={format_number(result['span_macro_nearby_auroc'])}。",
             f"两个错误片段之间的正常 token 误报率={format_number(result['gap_fpr'])}。", "",
             "总体与回答内指标不是同一个统计问题。跨回答配对占多数本身并不能证明作弊或指标虚高。", "",
             "## 每个 token 到底有没有报对", "",
             "tokens.csv 给出原分数、原阈值、TP/FN/FP/TN、首错角色、span 和逐 token AUROC 排序贡献。",
             "AUROC贡献不是二值命中。spans.csv 中片段自身 AUROC 留空，因为一个错误片段内部没有正常负例。",
             "normal_gaps.csv 检查分数是否把两个错误段之间的正常内容也一起抬高。", "",
             "## 图、多头和编码器", "", f"冻结图审计状态：{graph_status}。线性诊断读出状态：{probe_status}。",
             "同头权重分布保持不变时，比较整向量共同置换与各头独立置换；只有后者额外破坏多头共同指向关系。",
             "degree_rewire 同时保持入度和出度；必须结合实际成功改边比例解释阴性结果。",
             "no_relay 保留节点自身逐层更新，但不让邻居传递其继承的邻居信息；不是删除所有历史连接。",
             "oracle_span_cut 使用金标，只能定位已训练模型对片段连接的依赖，不能当无监督检测成绩。",
             "图编码的泛化收益看 independently_retrained.json；冻结删边下降不能替代公平重训对照。", "",
             "## 不应越过的结论边界", "",
             "准备图只保留阈值以上边。retained_entropy 是保留权重条件熵，不是完整注意力熵，也不是词表预测熵。",
             "输入、投影、各消息层的线性读出是有监督诊断探针，不改原模型；未在测试集调阈值或挑最好头。",
             "这些审计回答检测器利用了什么统计和结构，不直接证明大模型事实推理的因果机制。"]
    lines.extend(measured_tables(result, output))
    (Path(output) / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def measured_tables(result, output):
    """Put the actual numbers beside the design, not only in large JSON files."""
    lines = ["", "## 本次实际测得的对照", "",
             "下表每行均在同一批 token 上比较。差值为改动后减原分数的 AUROC，负值表示下降。", "",
             "| 冻结改动 | token数 | 原 AUROC | 改后 AUROC | 差值 | 首错 AUROC 差值 |",
             "|---|---:|---:|---:|---:|---:|"]
    names = dict(score_prefix="仅保留当前前缀（子集）", score_no_messages="不接收消息",
                 score_edge_only_messages="消息不含邻居状态", score_no_relay="禁止多跳继承",
                 score_degree_rewire_0="保度数换邻居，第1次", score_no_rp="删prompt消息",
                 score_no_rr="删history消息", score_zero_edge="边属性清零")
    for key, name in names.items():
        if key not in result["controls"]:
            continue
        row = result["controls"][key]
        a, b = row["baseline"]["auroc"], row["control"]["auroc"]
        first = row["role_rankings"]["first_error"]
        delta = b - a if a is not None and b is not None else None
        onset_delta = (first["control"]["auroc"] - first["baseline"]["auroc"]
                       if first["control"]["auroc"] is not None and first["baseline"]["auroc"] is not None else None)
        lines.append(f"| {name} | {row['common_tokens']} | {format_number(a)} | {format_number(b)} | {format_number(delta)} | {format_number(onset_delta)} |")
    pair_path = Path(output) / "paired_controls.json"
    if pair_path.exists():
        lines.extend(["", "### 区分多头共同指向与普通端点打乱", ""])
        for name, row in json.loads(pair_path.read_text()).items():
            a, b = row["baseline"]["auroc"], row["control"]["auroc"]
            delta = b - a if a is not None and b is not None else None
            lines.append(f"{name}：AUROC差值 {format_number(delta)}；来源重采样区间 {row['delta_ci95']}（两列依次为AUROC、AP）。")
    probe_path = Path(output) / "probes.json"
    if probe_path.exists():
        probes = json.loads(probe_path.read_text())
        if probes["status"] == "completed":
            lines.extend(["", "### 相同线性读出规则下，各类输入包含多少可分类信息", "",
                          "| 输入 | token AUROC | AP | 首错 AUROC |", "|---|---:|---:|---:|"])
            for name, row in probes["probes"].items():
                first = row["roles_vs_all_normal"]["first_error"]["auroc"]
                lines.append(f"| {name} | {format_number(row['auroc'])} | {format_number(row['ap'])} | {format_number(first)} |")
        else:
            lines.append("线性读出未执行：" + probes["reason"])
    return lines


def process(root, args):
    samples, settings = load_predictions(root, args.completed_only)
    output = Path(args.output) if args.output else root / "deep_audit"
    if output.resolve() in (root.resolve(), (root / "samples").resolve()):
        raise ValueError("Output must be separate from the original prediction directory")
    output.mkdir(parents=True, exist_ok=True)
    samples, joined = attach_learned_controls(samples, args.learned_root or root / "learned_audit")
    write_json(output / "learned_audit_join.json", joined)
    write_json(output / "scope.json", dict(predictions=str(root), completed_only=args.completed_only,
                                           evaluated_ids=[s["id"] for s in samples], settings=settings))
    result = analyze_scores(samples, settings, output / "saved_scores", args.window,
                            args.bootstrap if args.stage == "scores" else 0)
    write_json(output / "dataset_inventory.json", prepared_inventory(samples, args.prepared))
    retrained = (dict(status="unavailable", reason="Partial prediction audit; matched full-cohort comparison is not attempted.")
                 if args.completed_only else compare_retrained(samples, settings, root, args.bootstrap))
    write_json(output / "independently_retrained.json", retrained)
    graph_status = probe_status = "not_requested"
    write_summary(result, output, graph_status, probe_status)
    if args.stage != "scores" and settings["variant"] in ("charm_in", "charm_out"):
        import torch
        from .capture import capture_samples, resolve_checkpoint
        from .probes import run_probes

        torch.set_num_threads(args.threads)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        enriched, model = capture_samples(samples, settings, output, args.prepared, args.checkpoint,
                                          device, args.repeats, args.max_lag, args.edge_chunk, args.replay_atol)
        result = analyze_scores(enriched, settings, output / "frozen_scores", args.window, args.bootstrap)
        pairs = paired_interventions(enriched, settings["threshold"]["value"], args.repeats, args.bootstrap)
        write_json(output / "paired_controls.json", pairs)
        graph_status = "completed"
        if args.stage == "all" and not args.skip_probes:
            checkpoint = resolve_checkpoint(settings, args.checkpoint)
            probes = run_probes(samples, model, settings, checkpoint, output, args.prepared, args.probe_epochs)
            probe_status = probes["status"]
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    elif args.stage != "scores":
        graph_status = "not_run_on_control_variant; saved-score and retrained comparisons completed"
    write_summary(result, output, graph_status, probe_status)
    print(json.dumps(dict(predictions=str(root), output=str(output), tokens=result["population"]["tokens"],
                          auroc=result["all_tokens"]["auroc"], graph=graph_status, probes=probe_status), ensure_ascii=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/charm_structure_audit_qa", help="one prediction directory or the experiment output root")
    parser.add_argument("--output", help="only with a single prediction directory")
    parser.add_argument("--stage", choices=("scores", "graph", "all"), default="all")
    parser.add_argument("--prepared", help="relocated prepared directory; original graph paths are used otherwise")
    parser.add_argument("--checkpoint", help="explicit checkpoint override for one prediction directory")
    parser.add_argument("--learned-root", help="relocated preceding learned_audit output; single prediction directory only")
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--edge-chunk", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--max-lag", type=int, default=32)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--replay-atol", type=float, default=2e-5)
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.probe_epochs < 1 or args.window < 1 or args.max_lag < 1 or args.bootstrap < 0:
        raise ValueError("Repeats, epochs, window and max-lag must be positive; bootstrap may be zero")
    roots = prediction_roots(args.root)
    if len(roots) != 1 and (args.output or args.checkpoint or args.learned_root):
        raise ValueError("--output/--checkpoint/--learned-root require a single prediction directory")
    for root in roots:
        process(root, args)


if __name__ == "__main__":
    main()
