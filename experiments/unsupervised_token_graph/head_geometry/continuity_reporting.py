"""Small review bundle for the frozen-score continuity audit."""

import csv
import json
import tarfile
from pathlib import Path

import numpy as np

from ..offline_span.data import write_json
from .reporting import number


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def metric_rows(report):
    rows = []
    for group, result in report["groups"].items():
        for method, measured in result["methods"].items():
            for view, values in measured["views"].items():
                rows.append(dict(group=group, method=method, view=view,
                    tokens=values["evaluated_tokens"], positives=values["evaluated_positives"],
                    coverage=values["coverage"], **values["pooled"],
                    within_auroc=values["within"]["pair_weighted_auroc"],
                    within_macro_auroc=values["within"]["macro_auroc"],
                    within_macro_ap=values["within"]["macro_ap"],
                    cross_answer_auroc=values["cross_answer_auroc"],
                    recall=values["recall"], fpr=values["fpr"]))
    return rows


def comparison_rows(report):
    rows, components = [], []
    for group, result in report["groups"].items():
        for name, pair in result["comparisons"].items():
            for scope in ("pooled", "within"):
                for view, value in pair[scope].items():
                    for index, metric in enumerate(value["order"]):
                        interval = value["ci95"]
                        rows.append({"group": group, "comparison": name, "scope": scope, "view": view,
                            "metric": metric, "delta": value["delta"][metric],
                            "low": interval[0][index] if interval else None,
                            "high": interval[1][index] if interval else None,
                            "bootstrap_valid": value["bootstrap_valid"]})
            for phase, value in pair["decomposition"]["partitions"].items():
                components.append(dict(group=group, comparison=name, phase=phase, **value))
    return rows, components


def group_summary(group, result):
    annotation = result["annotation"]
    lines = [f"## {group}", "",
             (f"标注映射片段 {annotation['token_spans']}；错误 token {annotation['error_tokens']}；"
             f"后续 token 占比 {number(annotation['continuation_fraction'])}。"),
             (f"共同覆盖后错误 token {annotation['scored_error_tokens']}；"
             f"后续占比 {number(annotation['scored_continuation_fraction'])}。"), "",
             "|方法|总体 AUROC/AP|同答 AUROC|起点 AUROC|后续 AUROC|正常匹配 span 误报率|",
             "|---|---|---|---|---|---|"]
    for method, measured in result["methods"].items():
        views = measured["views"]
        all_error = views["all_error"]
        values = [number(all_error["pooled"]["auroc"]) + "/" + number(all_error["pooled"]["ap"]),
                  number(all_error["within"]["pair_weighted_auroc"]),
                  number(views["span_onset_vs_normal"]["pooled"]["auroc"]),
                  number(views["continuation_vs_normal"]["pooled"]["auroc"]),
                  number(measured["matched"]["normal"]["any_alarm_rate"])]
        lines.append("|" + "|".join([method, *values]) + "|")
    return lines + gain_summary(result)


def gain_summary(result):
    pair = result["comparisons"].get("pair_state_smooth_minus_pair_state")
    if pair is None:
        return ["", "本次冻结结果没有所需平滑对照；未补造平滑告警或对应增益。"]
    decomposition = pair["decomposition"]
    lines = ["", "平滑减单点的 ΔAUROC 分解（共同负例；不是因果贡献比例）：", "",
             "|正例阶段|正例数|正例权重|阶段 ΔAUROC|加权增益|", "|---|---:|---:|---:|---:|"]
    for phase in [*decomposition["component_order"], "continuation", "all"]:
        row = decomposition["partitions"][phase]
        values = [phase, str(row["positives"])]
        values += [number(row[key]) for key in ("positive_weight", "delta_auroc", "weighted_delta_auroc")]
        lines.append("|" + "|".join(values) + "|")
    lines += ["", "continuation 是后三组的合计，不可再次与四组相加；AP 不适用此分解。"]
    order = result["order_null"]
    if order is not None:
        lines += [(f"共同覆盖 {order['observed']['tokens']} token 上重算的原顺序平滑 ΔAUROC："
                  f"{number(order['observed']['auroc_gain'])}；"
                  f"成对打乱后的平均值：{number(order['null']['auroc_gain']['mean'])}。"),
                  "此处对已校准单点分数重算窗口并在共同缺口重置，不能冒充冻结平滑分数的复现。",
                  "顺序空模型保留单点分数—标签关系，也破坏精确位置；不是线上检测或模型干预。"]
    return lines


def summary_text(report, matching):
    pairs = sum(len(row["pairs"]) for row in matching)
    missing = sum(len(row["unmatched"]) for row in matching)
    lines = ["# 连续性审计", "", "分数和告警沿用冻结文件；没有训练、重新校准或加载大模型。",
             f"同答等长匹配：{pairs} 对；未匹配：{missing} 段，原因见 matching.json。",
             "全部方法在共同覆盖上比较。缺起点不记为漏检，缺测片段与完整漏检分开。",
             "阈值来自原混合无标签校准集，不能理解成正常 FPR 固定为 5%。", ""]
    for group, result in report["groups"].items():
        lines.extend(group_summary(group, result))
        lines.append("")
    lines += ["判断顺序：", "",
              "1. decomposition.csv：增益是否主要落在后续位置；这只是排序增益的精确归账。",
              "2. comparisons.csv：平滑的增益在同答比较中是否仍存在，来源配对区间是否跨零。",
              "3. matched_spans.csv / matching.json：正常连续片段是否同样容易报警，匹配覆盖如何。",
              "4. audit.json 的 order_null：破坏连续顺序后，保留单点可分性的平滑还能否获益。",
              "5. spans.csv：首告警是否滞后、漏检多少；不只看已检出片段的平均延迟。", "",
              "以上可支持分数层面的持续性解释，不能证明 1024 维头状态重复或特定因果机制。",
              "多次查看过的测试集与未校正多重比较的区间均为探索性结果。"]
    return "\n".join(lines) + "\n"


def save_token_snapshot(root, blocks, methods):
    arrays = {"answer_index": np.concatenate([np.full(block["tokens"], index, int)
                                               for index, block in enumerate(blocks)]),
              "label": np.concatenate([block["views"]["all_error"][0] for block in blocks]),
              "common_finite": np.concatenate([block["common_finite"] for block in blocks])}
    for method in methods:
        arrays[method] = np.concatenate([block["scores"][method] for block in blocks])
        arrays[method + "__alarm"] = np.concatenate([block["alarms"][method] for block in blocks])
    np.savez_compressed(root / "tokens.npz", **arrays)
    records = [{"record": block["record"], "tokens": block["tokens"], "gold": block["gold"].tolist(),
                    "offsets": block["offsets"].tolist(), "text": block["text"]} for block in blocks]
    write_json(root / "answers.json", records)


def archive_audit(output):
    destination = output.parent / (output.name + "_continuity_review.tar.gz")
    partial = destination.with_suffix(".partial")
    with tarfile.open(partial, "w:gz") as archive:
        archive.add(output / "continuity", arcname=str(Path(output.name) / "continuity"))
    partial.replace(destination)
    return destination


def save_audit(output, report, tables, matching, blocks, settings):
    root = output / "continuity"
    root.mkdir(exist_ok=True)
    write_json(root / "audit.json", report)
    write_json(root / "matching.json", matching)
    write_json(root / "input_settings.json", settings)
    write_table(root / "metrics.csv", metric_rows(report))
    comparisons, decomposition = comparison_rows(report)
    write_table(root / "comparisons.csv", comparisons)
    write_table(root / "decomposition.csv", decomposition)
    write_table(root / "annotations.csv", [dict(group=key, **value["annotation"])
                                            for key, value in report["groups"].items()])
    for table in ("answers", "spans", "matched", "profiles"):
        rows = []
        for group, local in tables.items():
            for value in local[table]:
                row = dict(value)
                if table == "profiles":
                    row["phase"] = row.pop("group")
                rows.append(dict(row, group=group))
        filename = "matched_spans" if table == "matched" else table
        write_table(root / (filename + ".csv"), rows)
    save_token_snapshot(root, blocks, report["methods"])
    (root / "summary.md").write_text(summary_text(report, matching), encoding="utf-8")
    return archive_audit(output)
