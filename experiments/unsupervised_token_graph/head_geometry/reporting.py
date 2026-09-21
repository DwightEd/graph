"""Report RAGTruth tasks separately; pooled results remain a supplementary view."""

import csv
from collections import defaultdict

from ..offline_span.data import write_json

FIELDS = ("dataset", "task", "generator", "group", "scope", "method", "view",
          "tokens", "positives", "coverage", "auroc", "ap",
          "source_weighted_auroc", "recall", "fpr")
TASKS = ("QA", "Summary", "Data2txt")


def evaluation_groups(blocks):
    groups = defaultdict(list)
    for block in blocks:
        record = block["record"]
        groups[record["task"]].append(block)
        groups[record["task"] + "|" + record["generator"]].append(block)
    groups["ALL"] = blocks
    return dict(groups)


def group_identity(blocks):
    tasks = sorted({block["record"]["task"] for block in blocks})
    generators = sorted({block["record"]["generator"] for block in blocks})
    return {"dataset": "RAGTruth", "task": tasks[0] if len(tasks) == 1 else "ALL",
            "generator": generators[0] if len(generators) == 1 else "ALL"}


def metric_rows(groups, identities):
    rows = []
    for group, result in groups.items():
        for view, methods in result["views"].items():
            for name, metric in methods.items():
                scope, method = name.split("__") if "__" in name else ("all", name)
                rows.append(dict(identities[group], group=group, scope=scope, method=method,
                    view=view, tokens=metric["evaluated_tokens"], positives=metric["evaluated_positives"],
                    coverage=metric["coverage"], auroc=metric["pooled"]["auroc"], ap=metric["pooled"]["ap"],
                    source_weighted_auroc=metric["source_fixed_full_answer"]["auroc"],
                    recall=metric.get("recall"), fpr=metric.get("fpr")))
    return rows


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return "—" if value is None else f"{value:.4f}"


def task_summary(report, rows):
    selection = report["head_selection"]
    lines = ["# RAGTruth 分任务评估", "",
             f"实际层：{selection['layers']}；每层 head：{selection['heads']}。",
             "scope=all 表示全部已选 head，不表示模型全部层。", "",
             "|任务|状态|回答数|", "|---|---|---:|"]
    for task in TASKS:
        result = report["groups"].get(task)
        lines.append(f"|{task}|{'已评估' if result else '未评估'}|{result['answers'] if result else '—'}|")
    lines += ["", "|任务|生成器|范围|方法|token 数|阳性数|覆盖率|AUROC|AP|召回|FPR|",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if "|" not in row["group"] or row["view"] != "all_error":
            continue
        identity = [row[key] for key in ("task", "generator", "scope", "method", "tokens", "positives")]
        values = [number(row[key]) for key in ("coverage", "auroc", "ap", "recall", "fpr")]
        lines.append("|" + "|".join(map(str, [*identity, *values])) + "|")
    lines += ["", "主候选沿用冻结记录：" + report["primary"] + "。",
              "各任务的参考/校准按来源分开拟合；ALL 为补充汇总，不是独立实验。",
              "阈值来自混合无标签校准集，不代表正常 token 的 FPR 固定为 5%。",
              "首错、起点、延续、前后半段及配对差值见各任务的 metrics.csv / evaluation.json。"]
    return "\n".join(lines) + "\n"


def save_reports(root, report, identities):
    rows = metric_rows(report["groups"], identities)
    write_json(root / "evaluation.json", report)
    write_metrics(root / "metrics.csv", rows)
    for task in sorted({value["task"] for value in identities.values()} - {"ALL"}):
        names = [name for name in report["groups"] if name == task or name.startswith(task + "|")]
        selected = dict(report, groups={name: report["groups"][name] for name in names})
        directory = root / "tasks" / task
        write_metrics(directory / "metrics.csv", [row for row in rows if row["group"] in names])
        write_json(directory / "evaluation.json", selected)
    (root / "task_summary.md").write_text(task_summary(report, rows), encoding="utf-8")
