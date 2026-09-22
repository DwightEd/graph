"""Small review bundle: explicit denominators, frozen alarms, and diagnostic curves."""

import json
import math
import tarfile

from experiments.unsupervised_token_graph.head_geometry.continuity_reporting import (
    write_table,
)


def json_values(value):
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value > 0 else "-inf" if value < 0 else None
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(json_values(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def metric_rows(report):
    rows = []
    for group, result in report["groups"].items():
        for method, settings in result["methods"].items():
            for setting in [settings["frozen"], *settings["descriptive_test_normal_fpr_curve"]]:
                for name, measured in setting["bins"].items():
                    row = {
                        "group": group,
                        "method": method,
                        "setting": setting["setting"],
                        "length_bin": name,
                    }
                    row.update(
                        {
                            key: setting[key]
                            for key in (
                                "requested_normal_fpr",
                                "achieved_normal_fpr",
                                "threshold",
                                "threshold_source",
                            )
                        }
                    )
                    row.update(measured["error"])
                    matched = measured["matched"]["all"]
                    row.update(matched["ranking"])
                    row.update(
                        {
                            "matched_" + key: matched["confusion"][key]
                            for key in ("tp", "fn", "fp", "tn", "recall_minus_fpr")
                        }
                    )
                    row.update(setting["answer_alarms"])
                    row["matched_normal_span_fpr"] = matched["normal"][
                        "before_end_recall_lower_bound"
                    ]
                    row["matched_error_span_recall"] = matched["error"][
                        "before_end_recall_lower_bound"
                    ]
                    rows.append(row)
    return rows


def comparison_rows(report):
    rows = []
    for group, result in report["groups"].items():
        for comparison in result["comparisons"]:
            identity = {key: value for key, value in comparison.items() if key != "differences"}
            rows.extend(
                dict(group=group, **identity, **difference)
                for difference in comparison["differences"]
            )
    return rows


def number(value):
    return "—" if value is None else f"{value:.3f}"


def short_span_table(rows):
    lines = [
        "|任务|方法|阈值口径|正常 token FPR|1–8 段内命中/全部|onset 命中/可测|正常匹配段误报率|",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["length_bin"] != "1-8":
            continue
        setting = (
            "原冻结"
            if row["setting"] == "frozen"
            else f"TEST诊断 {row['requested_normal_fpr']:.0%}"
        )
        values = [
            row["group"],
            row["method"],
            setting,
            number(row["achieved_normal_fpr"]),
            f"{row['before_end_detected']}/{row['spans']}",
            f"{row['onset_alarms']}/{row['observed_onsets']}",
            number(row["matched_normal_span_fpr"]),
        ]
        lines.append("|" + "|".join(value.replace("|", " / ") for value in values) + "|")
    return lines


def answer_alarm_table(rows):
    lines = []
    lines.extend(
        [
            "",
            "## 原冻结阈值下的回答误报与迟到报警",
            "",
            "token FPR 不控制完整回答至少一次报警的概率。后续报警只是描述性关联，不证明来自前一错误。",
            "",
            "|任务|方法|全正常回答有报警/全部|短段完整漏检后15步有报警|",
            "|---|---|---:|---:|",
        ]
    )
    for row in rows:
        if row["length_bin"] == "1-8" and row["setting"] == "frozen":
            values = [
                row["group"],
                row["method"],
                f"{row['all_normal_answers_with_alarm']}/{row['all_normal_answers']}",
                str(row["late_only_alarm_spans"]),
            ]
            lines.append("|" + "|".join(value.replace("|", " / ") for value in values) + "|")
    return lines


def summary_text(report, rows):
    lines = [
        "# 短 span 检验",
        "",
        "只评价已冻结分数，没有训练新的检测器。",
        "frozen 使用原混合 CAL 报警；同 FPR 曲线使用 TEST 正常标签，只是离线诊断，不能当部署校准。",
        "全部方法使用共同覆盖。缺起点不顺移，段结束后报警不算命中，缺失未决与完整漏检分开。",
        "",
    ]
    lines.extend(short_span_table(rows))
    lines.extend(answer_alarm_table(rows))
    lines.extend(
        [
            "",
            "细分 1–2、3–4、5–8 的结果见 metrics.csv；逐段与未决原因见 spans.csv。",
            "正常匹配只控制长度、粗词面、位置和重复程度，不声称已控制实体/关系语义。",
            "paired_spans.csv / audit.json 区分此前 15 步有错误的 recovery 与 clean_history。",
            "comparisons.csv 同时报配对短段召回差、正常段误报差及 recall−FPR 差；按 source 重采样，固定当前阈值。",
            "metrics.csv 的 deadline 0/1/3/7 用全部 span 作分母，缺测提供上下界；长段另分 9–16、17+。",
            "spans.csv 记录错误结束后的连续报警长度及缺测/下一错误/回答结尾截尾，后续报警不回填成功。",
            "answers.csv 保存每回答 TP/FN/FP/TN；全正常回答至少一次误报率及缺测上下界见 metrics.csv。",
            "cohort.json 保存短段和正常配对的文本、位置，供 teaching 单目标贡献采集使用。",
            "reanchor.csv（如提供旧归档）只记录确认的 local→old 事件；未确认不是确认不存在。",
            "反复查看过的 TEST 及未多重校正的区间均属于探索性分析。",
        ]
    )
    return "\n".join(lines) + "\n"


def save_report(output, report, cohort, matching, settings, reanchors):
    output.mkdir(parents=True, exist_ok=True)
    spans, pairs = report.pop("span_rows"), report.pop("paired_rows")
    answers = report.pop("answer_rows")
    rows = metric_rows(report)
    write_json(output / "audit.json", report)
    write_json(output / "cohort.json", cohort)
    write_json(output / "matching.json", matching)
    write_json(output / "input_settings.json", settings)
    write_table(output / "metrics.csv", rows)
    write_table(output / "spans.csv", spans)
    write_table(output / "paired_spans.csv", pairs)
    write_table(output / "answers.csv", answers)
    write_table(output / "comparisons.csv", comparison_rows(report))
    write_table(output / "reanchor.csv", reanchors)
    (output / "summary.md").write_text(summary_text(report, rows), encoding="utf-8")
    destination = output.parent / (output.name + "_review.tar.gz")
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.add(path, arcname=f"{output.name}/{path.name}")
    return destination
