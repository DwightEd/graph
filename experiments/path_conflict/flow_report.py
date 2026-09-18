"""Summarize head support, suppression, competition and downstream reversal."""

from pathlib import Path
import json
import tarfile

import numpy as np
import pandas as pd

from .flow_plan import opposition, sign_role


KEYS = ["case_id", "side", "panel"]


def role_table(output):
    writes = pd.read_csv(output / "baseline_head_sources.csv.gz")
    effects = pd.read_csv(output / "head_interventions.csv")

    heads = writes[writes["head"].ge(0)].copy()
    wide = heads.pivot_table(
        index=KEYS + ["layer", "head"],
        columns="source_group",
        values=["attention_mass", "local_lens_support"],
        aggfunc="first",
    )
    wide.columns = [f"{value}_{group}" for value, group in wide.columns]
    wide = wide.reset_index()

    final = effects.pivot_table(
        index=KEYS + ["layer", "head"],
        columns="source_group",
        values="final_support",
        aggfunc="first",
    )
    final.columns = ["final_" + name for name in final.columns]
    table = wide.merge(final.reset_index(), on=KEYS + ["layer", "head"], how="left")

    table["local_role"] = table.local_lens_support_all_context.map(sign_role)
    table["final_role"] = table.final_all_context.map(sign_role)
    table["evidence_role"] = table.final_evidence.map(sign_role)
    table["history_role"] = table.final_history.map(sign_role)
    table["local_source_opposition"] = [
        opposition(evidence, history + wrong)
        for evidence, history, wrong in zip(
            table.local_lens_support_evidence,
            table.local_lens_support_history,
            table.local_lens_support_wrong_source,
        )
    ]
    table["downstream_reversal"] = (
        table.local_lens_support_all_context * table.final_all_context < 0
    )
    table["evidence_downstream_reversal"] = (
        table.local_lens_support_evidence * table.final_evidence < 0
    )
    table["evidence_retention"] = table.final_evidence / table.local_lens_support_evidence.replace(0, np.nan)
    return table


def layer_competition(output):
    writes = pd.read_csv(output / "baseline_head_sources.csv.gz")
    rows = []
    selected = writes[(writes["head"] >= 0) & (writes.source_group == "all_context")]

    for key, group in selected.groupby(KEYS + ["layer"]):
        value = group.local_lens_support.to_numpy()
        positive = value[value > 0].sum()
        negative = -value[value < 0].sum()
        denominator = positive + negative
        balance = 0.0 if denominator == 0 else 2 * min(positive, negative) / denominator
        rows.append(dict(
            zip(KEYS + ["layer"], key),
            positive_support=positive,
            wrong_support=negative,
            competition_balance=balance,
            supporting_heads=int((value > 0).sum()),
            wrong_heads=int((value < 0).sum()),
        ))
    return pd.DataFrame(rows)


def evidence_layers(output):
    writes = pd.read_csv(output / "baseline_head_sources.csv.gz")
    groups = ["evidence", "wrong_source", "history", "all_context"]
    total = writes[(writes.head == -1) & writes.source_group.isin(groups)].copy()
    return total[KEYS + ["layer", "source_group", "attention_mass", "local_lens_support"]]


def propagation_delta(output):
    paths = pd.read_csv(output / "propagation.csv.gz")
    baseline = {}

    for path in (output / "baseline").glob("*.npz"):
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(str(saved["record"]))
            trajectory = json.loads(str(saved["trajectory"]))
        key = tuple(record[name] for name in KEYS)
        baseline[key] = {
            (row["layer"], row["site"]): row["margin"] for row in trajectory
        }

    rows = []
    for row in paths.itertuples():
        key = tuple(getattr(row, name) for name in KEYS)
        base = baseline[key][(row.layer, row.site)]
        rows.append(dict(
            **{name: getattr(row, name) for name in KEYS},
            variant=row.variant,
            source_group=row.source_group,
            layer_cut=row.layer_cut,
            head_cut=row.head_cut,
            layer=row.layer,
            site=row.site,
            margin_delta=row.margin - base,
        ))
    return pd.DataFrame(rows)


def same_question_deltas(roles):
    supported = roles[roles.side == "supported"]
    unsupported = roles[roles.side == "unsupported"]
    keys = ["case_id", "panel", "layer", "head"]
    fields = [
        column for column in roles
        if column.startswith(("attention_mass_", "local_lens_support_", "final_"))
    ]
    merged = unsupported[keys + fields].merge(
        supported[keys + fields], on=keys, suffixes=("_unsupported", "_supported")
    )
    for field in fields:
        merged["delta_" + field] = (
            merged[field + "_unsupported"] - merged[field + "_supported"]
        )
    return merged


def supervised_alignment(roles, path):
    """Join head IDs only when the saved detector has the same geometry."""
    if not path:
        return pd.DataFrame()
    file = Path(path)
    if not file.exists():
        return pd.DataFrame()

    rules = pd.read_csv(file)
    if rules.layer.max() != roles.layer.max() or rules.head.max() != roles.head.max():
        return pd.DataFrame([
            dict(note="different model geometry: head IDs were not aligned")
        ])

    columns = ["layer", "head", "mean_gap", "weight", "fit_gap_contribution"]
    return roles.merge(rules[columns], on=["layer", "head"], how="left")


def write_report(output, roles, alignment):
    ranked = roles.assign(size=roles.final_all_context.abs()).sort_values(
        "size", ascending=False
    )
    columns = KEYS + [
        "layer", "head", "final_all_context", "final_evidence", "final_history",
        "local_source_opposition", "downstream_reversal",
        "evidence_downstream_reversal",
    ]
    lines = [
        "# Evidence ↔ Target 功能流审计",
        "",
        "attention质量、当前层局部候选支持、跑完后续网络后的最终功能支持分开报告。",
        "正支持表示该写入帮助正确候选；负支持表示推动错误候选。",
        "",
        "## 先看",
        "1. head_roles.csv：每个确认head的证据/历史/错误来源作用。",
        "2. layer_competition.csv：同层正负head竞争。",
        "3. evidence_forward.csv：证据在各层的局部支持。",
        "4. propagation_delta.csv.gz：删路径后的影响如何被后续层保留、抵消或反转。",
        "5. same_question_head_deltas.csv：同题有依据/无依据的同head差异。",
        "",
        "## 最终作用最大的已确认head",
        "",
        ranked[columns].head(16).to_string(index=False),
        "",
        "候选head由本次基线局部作用发现，再在同一case确认，属于机制发现而非独立复验。",
        "人工证据角色仅用于这两个局部陈述，不是无监督检测标签。",
    ]
    if not alignment.empty:
        lines += [
            "",
            "## 监督检测器坐标对齐",
            alignment.head(12).to_string(index=False),
            "不同模型几何不会强行按head编号对齐。",
        ]
    (output / "REPORT_FLOW_zh.md").write_text("\n".join(lines), encoding="utf-8")


def report_flow(output, supervised_path=None):
    output = Path(output)
    roles = role_table(output)
    competition = layer_competition(output)
    evidence = evidence_layers(output)
    paths = propagation_delta(output)
    deltas = same_question_deltas(roles)
    alignment = supervised_alignment(roles, supervised_path)

    roles.to_csv(output / "head_roles.csv", index=False)
    competition.to_csv(output / "layer_competition.csv", index=False)
    evidence.to_csv(output / "evidence_forward.csv", index=False)
    paths.to_csv(output / "propagation_delta.csv.gz", index=False)
    deltas.to_csv(output / "same_question_head_deltas.csv", index=False)
    alignment.to_csv(output / "supervised_alignment.csv", index=False)
    write_report(output, roles, alignment)

    archive = output / "flow_review.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(output.glob("*")):
            if path.is_file() and path.name != archive.name and path.suffix != ".npz":
                stream.add(path, arcname=path.name)

    print(roles[KEYS + [
        "layer", "head", "final_all_context", "final_evidence",
        "final_history", "downstream_reversal",
    ]].to_string(index=False))
