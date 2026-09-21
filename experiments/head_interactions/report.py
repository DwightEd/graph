"""Read completed panels, retain raw effects, and bootstrap source-level means."""

import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from state_audit.storage import read_arrays, read_json, write_csv, write_json

from .trials import unpack_messages

IDENTITY = (
    "case_id",
    "source_id",
    "dataset",
    "task",
    "generator",
    "panel",
    "query_offset",
    "side",
    "variant",
)
STRATA = ("dataset", "task", "generator", "panel", "query_offset", "side", "variant", "readout")
MEASURES = {
    "singles": ("native_effect", "boost_effect"),
    "interactions": ("interaction", "a_with_b", "a_without_b", "b_with_a", "b_without_a"),
    "adaptation": (
        "total_effect",
        "fixed_downstream_effect",
        "adaptation",
        "absolute_effect_reduction",
    ),
}


def identity(context):
    row = {name: context["id" if name == "case_id" else name] for name in IDENTITY[:-2]}
    row["side"] = context["metadata"].get("side", "unspecified")
    row["variant"] = context["metadata"].get("variant", "natural")
    return row


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path, rows, fields or list(IDENTITY))


def collect(root):
    tables = {name: [] for name in (*MEASURES, "controls", "inventory", "conditions", "messages")}
    for marker in sorted((root / "cases").glob("*/*/*/complete.json")):
        directory = marker.parent
        context, result = (
            read_json(directory / "context.json"),
            read_json(directory / "results.json"),
        )
        common = identity(context)
        tables["inventory"].append(dict(common, valid=result["valid"], **context["metadata"]))
        for name, check in result["controls"].items():
            tables["controls"].append(dict(common, control=name, **check))
        for name in MEASURES:
            for row in result[name]:
                tables[name].append(dict(common, valid=result["valid"], **row))
        collect_trials(directory, context, tables)
    return tables


def collect_trials(directory, context, tables):
    common = identity(context)
    sites = {site["name"]: site for site in context["sites"]}
    for path in sorted((directory / "trials").glob("*.json")):
        record = read_json(path)
        scores = record["scores"]
        tables["conditions"].append(
            dict(
                common,
                condition=record["condition"],
                sum_margin=scores["sum_margin"],
                mean_margin=scores["mean_margin"],
            )
        )
        messages = unpack_messages(record["messages"], read_arrays(path.with_suffix(".npz")))
        for name, values in messages.items():
            site = sites[name]
            for position_index, position in enumerate(site["positions"]):
                for head_index, head in enumerate(site["heads"]):
                    mass = values["mass"]
                    row = dict(
                        common,
                        condition=record["condition"],
                        site=name,
                        layer=site["layer"],
                        head=head,
                        query=position,
                        selected_mass=None
                        if mass is None
                        else float(mass[position_index, head_index]),
                    )
                    for field in ("write", "total_write"):
                        row[field + "_norm"] = float(
                            np.linalg.norm(values[field][position_index, head_index])
                        )
                    tables["messages"].append(row)


def source_summary(rows, metric, draws, seed=0):
    sources = defaultdict(list)
    for row in rows:
        sources[row["source_id"]].append(float(row[metric]))
    values = np.asarray([np.mean(sources[name]) for name in sorted(sources)])
    result = dict(
        sources=len(values), rows=len(rows), mean=float(values.mean()), ci_low=None, ci_high=None
    )
    if len(values) >= 2 and draws > 0:
        random = np.random.default_rng(seed)
        means = np.empty(draws)
        for draw in range(draws):
            means[draw] = random.choice(values, len(values), replace=True).mean()
        result["ci_low"], result["ci_high"] = np.quantile(means, [0.025, 0.975]).tolist()
    return result


def aggregate(tables, draws):
    rows = []
    for kind, metrics in MEASURES.items():
        fields = (*STRATA, "group") if kind == "singles" else (*STRATA, "a", "b")
        if kind == "interactions":
            fields = (*fields, "reference")
        groups = defaultdict(list)
        for row in tables[kind]:
            if row["valid"]:
                groups[tuple(row[name] for name in fields)].append(row)
        for values, selected in groups.items():
            group = dict(zip(fields, values))
            for metric in metrics:
                rows.append(
                    dict(
                        group, table=kind, metric=metric, **source_summary(selected, metric, draws)
                    )
                )
    return rows


def pattern_candidates(tables, minimum):
    """Explicit numerical screens, not causal labels or hallucination detector predictions."""
    singles = {
        tuple(row[field] for field in (*IDENTITY, "readout", "group")): row
        for row in tables["singles"]
        if row["valid"]
    }
    rows = []
    for row in tables["interactions"]:
        if (
            not row["valid"]
            or row["reference"] != "zero"
            or row["readout"] == "candidate_sum_margin"
        ):
            continue
        common = {field: row[field] for field in (*IDENTITY, "readout")}
        b = singles[tuple(row[field] for field in (*IDENTITY, "readout")) + (row["b"],)]
        rows.append(
            dict(
                common,
                a=row["a"],
                b=row["b"],
                conditional_effect=abs(row["interaction"]) >= minimum,
                a_sign_reversal=row["a_with_b"] * row["a_without_b"] < 0
                and min(abs(row["a_with_b"]), abs(row["a_without_b"])) >= minimum,
                b_attenuates_a_candidate=row["a_without_b"] >= minimum
                and row["interaction"] <= -minimum,
                protective_b_rival_preferred=row["b_with_a"] >= minimum and row["F11"] <= -minimum,
                removal_b_recovers_margin=row["F11"] <= -minimum and row["F10"] >= minimum,
                boost_b_recovers_margin=row["F11"] <= -minimum and b["boosted"] >= minimum,
            )
        )
    return rows


def matched_random_differences(tables):
    """Compare identical cases/queries/head-group templates before source aggregation."""
    differences = []
    for kind, metrics in MEASURES.items():
        keys = (
            ["case_id", "query_offset", "readout", "group"]
            if kind == "singles"
            else ["case_id", "query_offset", "readout", "a", "b"]
        )
        if kind == "interactions":
            keys.append("reference")
        selected = {
            tuple(row[key] for key in keys): row
            for row in tables[kind]
            if row["valid"] and row["panel"] == "selected"
        }
        for row in tables[kind]:
            if not row["valid"] or not row["panel"].startswith("random_"):
                continue
            original = selected.get(tuple(row[key] for key in keys))
            if original is None:
                continue
            for metric in metrics:
                differences.append(
                    dict(
                        {field: row[field] for field in (*IDENTITY, *keys)},
                        table=kind,
                        metric=metric,
                        selected=original[metric],
                        random=row[metric],
                        difference=original[metric] - row[metric],
                    )
                )
    return differences


def semantic_differences(rows, cases):
    """Difference of interactions with unchanged candidate identities, not flipped support signs."""
    candidates = {case["id"]: case["candidates"] for case in cases}
    keys = ("source_id", "dataset", "task", "generator", "panel", "query_offset", "a", "b")
    selected = [
        row
        for row in rows
        if row["valid"] and row["reference"] == "zero" and row["readout"] == "candidate_sum_margin"
    ]
    bases = {tuple(row[key] for key in keys): row for row in selected if row["variant"] == "base"}
    differences = []
    for row in selected:
        if row["variant"] == "base":
            continue
        baseline = bases.get(tuple(row[key] for key in keys))
        if baseline is None:
            continue
        if candidates[row["case_id"]] != candidates[baseline["case_id"]]:
            raise ValueError("Semantic interaction comparison requires fixed candidate token IDs")
        differences.append(
            dict(
                {key: row[key] for key in (*IDENTITY, "a", "b")},
                base_interaction=baseline["interaction"],
                variant_interaction=row["interaction"],
                interaction_change=row["interaction"] - baseline["interaction"],
            )
        )
    return differences


def write_review(root, summary, aggregate_rows):
    lines = [
        "# 头组条件作用审计",
        "",
        f"已完成 {summary['completed_panels']}/{summary['expected_panels']} 个 panel；"
        f"有效 {summary['valid_panels']}；失败控制 {summary['failed_controls']}。",
        "",
        "先看 inventory.csv 和 controls.csv，再看 interactions.csv 的四个 F。",
        "sum_margin 为主读出，mean_margin 仅为长度敏感性检查；自然两侧前缀不同。",
        "",
        "patterns.csv 只是数值候选，不把负交互自动叫抑制，不把候选偏好当自然回答标签。",
        "adaptation.csv 中 adaptation=删除A后自然重算−删除A后恢复B；需结合删除效应方向解释。",
        "messages.csv 的 write 是当前 A·V 路线分量；total_write 包含实际注入。",
        "未发生路由读取时的零效应不证明不存在该机制。随机头匹配层/数量/来源，不匹配能量。",
        "",
        "区间按 source bootstrap，未做多重比较校正；两个来源只能作为案例审计。",
        "",
        "|任务|panel|A|B|交互均值（nats）|来源数|",
        "|---|---|---|---|---:|---:|",
    ]
    for row in aggregate_rows:
        if (
            row["metric"] == "interaction"
            and row["readout"] == "sum_margin"
            and row["reference"] == "zero"
        ):
            lines.append(
                f"|{row['task']}|{row['panel']}|{row['a']}|{row['b']}|{row['mean']:.4f}|{row['sources']}|"
            )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(root: Path, draws=1000):
    settings = read_json(root / "settings.json")
    tables = collect(root)
    aggregates = aggregate(tables, draws)
    tables["aggregate"] = aggregates
    tables["patterns"] = pattern_candidates(tables, settings["minimum_effect"])
    tables["selected_minus_random"] = matched_random_differences(tables)
    tables["semantic_differences"] = semantic_differences(tables["interactions"], settings["cases"])
    for name, rows in tables.items():
        table(root / f"{name}.csv", rows)
    summary = dict(
        purpose=settings["purpose"],
        sources=len({row["source_id"] for row in tables["inventory"]}),
        expected_panels=len(settings["cases"])
        * len(settings["query_offsets"])
        * (1 + len(settings["random_seeds"])),
        completed_panels=len(tables["inventory"]),
        completed_cases=len({row["case_id"] for row in tables["inventory"]}),
        valid_panels=sum(int(row["valid"]) for row in tables["inventory"]),
        failed_controls=sum(int(not row["passed"]) for row in tables["controls"]),
        interaction_rows=len(tables["interactions"]),
        adaptation_rows=len(tables["adaptation"]),
        readouts=["sum_margin", "mean_margin", "candidate_sum_margin"],
        statistical_status="exploratory_source_bootstrap_not_corrected_for_multiple_comparisons",
    )
    write_json(root / "summary.json", summary)
    write_review(root, summary, aggregates)
    archive = root.parent / (root.name + "_review.tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(root, arcname=root.name)
    return dict(summary, archive=str(archive))
