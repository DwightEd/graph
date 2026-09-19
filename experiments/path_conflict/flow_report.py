"""Summarize head support, suppression, competition and downstream reversal."""

from pathlib import Path
import json
import tarfile

import numpy as np
import pandas as pd

from .flow_plan import opposition, sign_role
from .binding_analysis import write_binding_robustness


KEYS = ["case_id", "side", "panel"]
SOURCE_GROUPS = (
    "all_context", "evidence", "condition", "value", "wrong_source", "history",
    "query_self", "recent_history", "remote_history", "past_history",
)
EFFECT_TOLERANCE = .02


def paired_numeric_fields(roles):
    fields = []
    for group in SOURCE_GROUPS:
        for prefix in ("attention_mass_", "local_lens_support_", "local_linear_support_", "final_"):
            name = prefix + group
            if name in roles.columns:
                fields.append(name)
    return fields


def role_table(output):
    writes = pd.read_csv(output / "baseline_head_sources.csv.gz")
    effects = pd.read_csv(output / "head_interventions.csv")

    heads = writes[writes["head"].ge(0)].copy()
    measures = ["attention_mass", "local_lens_support"]
    if "local_linear_support" in heads:
        measures.append("local_linear_support")
    wide = heads.pivot_table(
        index=KEYS + ["layer", "head"],
        columns="source_group",
        values=measures,
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

    local = "local_linear_support" if "local_linear_support" in heads else "local_lens_support"
    table["local_measure"] = local
    table["local_role"] = table[local + "_all_context"].map(sign_role)
    table["final_role"] = table.final_all_context.map(sign_role)
    table["evidence_role"] = table.final_evidence.map(sign_role)
    table["history_role"] = table.final_history.map(sign_role)
    table["local_source_opposition"] = [
        opposition(evidence, history + wrong)
        for evidence, history, wrong in zip(
            table[local + "_evidence"],
            table[local + "_history"],
            table[local + "_wrong_source"],
        )
    ]
    table["downstream_reversal"] = (
        (table[local + "_all_context"] * table.final_all_context < 0)
        & (table[local + "_all_context"].abs() > EFFECT_TOLERANCE)
        & (table.final_all_context.abs() > EFFECT_TOLERANCE)
    )
    table["evidence_downstream_reversal"] = (
        (table[local + "_evidence"] * table.final_evidence < 0)
        & (table[local + "_evidence"].abs() > EFFECT_TOLERANCE)
        & (table.final_evidence.abs() > EFFECT_TOLERANCE)
    )
    denominator = table[local + "_evidence"]
    usable = (denominator.abs() > EFFECT_TOLERANCE) & (denominator * table.final_evidence > 0)
    table["evidence_final_over_local"] = table.final_evidence / denominator.where(usable)
    return table


def binding_table(roles, tolerance=EFFECT_TOLERANCE):
    required = [
        "final_condition", "final_value", "final_evidence", "final_wrong_source",
        "local_lens_support_condition", "local_lens_support_value",
    ]
    if any(name not in roles.columns for name in required):
        return pd.DataFrame()

    table = roles[KEYS + ["layer", "head"] + required].copy()
    condition = table.final_condition.to_numpy()
    value = table.final_value.to_numpy()
    joint = table.final_evidence.to_numpy()

    denominator = np.maximum(np.abs(condition), np.abs(value))
    same_direction = condition * value > 0
    table["binding_completeness"] = np.where(
        same_direction & (denominator > tolerance),
        np.minimum(np.abs(condition), np.abs(value)) / denominator,
        0.0,
    )
    table.loc[
        ~np.isfinite(table.final_condition) | ~np.isfinite(table.final_value),
        "binding_completeness",
    ] = np.nan
    table["joint_nonadditivity"] = joint - condition - value
    table["condition_value_interaction"] = condition + value - joint
    table["condition_downstream_change"] = (
        table.final_condition - table.local_lens_support_condition
    )
    table["value_downstream_change"] = (
        table.final_value - table.local_lens_support_value
    )
    table["evidence_vs_wrong_competition"] = [
        opposition(evidence, wrong)
        for evidence, wrong in zip(table.final_evidence, table.final_wrong_source)
    ]

    states = np.full(len(table), "weak_or_mixed", dtype=object)
    states[(condition > tolerance) & (value > tolerance)] = "both_support_candidate"
    states[(value > tolerance) & (np.abs(condition) <= tolerance)] = "value_effect_only"
    states[(condition > tolerance) & (np.abs(value) <= tolerance)] = "condition_effect_only"
    states[(condition < -tolerance) & (value < -tolerance)] = "both_oppose_candidate"
    states[(condition * value < 0) & (np.abs(condition) > tolerance)
           & (np.abs(value) > tolerance)] = "condition_value_opposed"
    states[~np.isfinite(condition) | ~np.isfinite(value)] = "not_tested"
    table["binding_state"] = states
    table["interpretation"] = "direct_position_effects_not_semantic_binding"
    return table


def layer_competition(output):
    writes = pd.read_csv(output / "baseline_head_sources.csv.gz")
    rows = []
    selected = writes[(writes["head"] >= 0) & (writes.source_group == "all_context")]

    for key, group in selected.groupby(KEYS + ["layer"]):
        measure = "local_linear_support" if "local_linear_support" in group else "local_lens_support"
        value = group[measure].to_numpy()
        positive = value[value > 0].sum()
        negative = -value[value < 0].sum()
        denominator = positive + negative
        balance = 0.0 if denominator == 0 else 2 * min(positive, negative) / denominator
        rows.append(dict(
            zip(KEYS + ["layer"], key),
            positive_support=positive,
            wrong_support=negative,
            competition_balance=balance,
            measure=measure,
            additive=measure == "local_linear_support",
            supporting_heads=int((value > 0).sum()),
            wrong_heads=int((value < 0).sum()),
        ))
    return pd.DataFrame(rows)


def evidence_layers(output):
    writes = pd.read_csv(output / "baseline_head_sources.csv.gz")
    groups = [
        "evidence", "condition", "value",
        "wrong_source", "history", "all_context"
    ]
    total = writes[(writes["head"] == -1) & writes.source_group.isin(groups)].copy()
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
    fields = paired_numeric_fields(roles)
    merged = unsupported[keys + fields].merge(
        supported[keys + fields], on=keys, suffixes=("_unsupported", "_supported")
    )
    for field in fields:
        merged["delta_" + field] = (
            merged[field + "_unsupported"] - merged[field + "_supported"]
        )
    return merged


def supervised_alignment(roles, path):
    """Equal L×H shapes do not identify a checkpoint or align its heads."""
    if not path:
        return pd.DataFrame()
    file = Path(path)
    if not file.exists():
        return pd.DataFrame()

    rules = pd.read_csv(file)
    identity = "checkpoint_sha"
    if identity not in roles or identity not in rules:
        return pd.DataFrame([dict(note="no verified checkpoint identity; head IDs not aligned")])
    if roles[identity].nunique() != 1 or rules[identity].nunique() != 1:
        return pd.DataFrame([dict(note="mixed checkpoint identities; head IDs not aligned")])
    if roles[identity].iloc[0] != rules[identity].iloc[0]:
        return pd.DataFrame([dict(note="different checkpoints; head IDs not aligned")])
    if rules.layer.max() != roles.layer.max() or rules.head.max() != roles.head.max():
        return pd.DataFrame([
            dict(note="different model geometry: head IDs were not aligned")
        ])

    columns = [
        "layer", "head", "self_gap", "prompt_gap",
        "self_weight", "prompt_weight",
        "self_risk_contribution", "prompt_risk_contribution",
        "routing_pattern",
    ]
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
        "候选是人工指定对比；首子词margin不等于完整事实选择。最终作用依赖删除操作。",
        "v2的local_linear_support共享FP32局部lens梯度，可相加；旧finite lens删除量不可相加。",
        "history保留旧定义（含query自身）；应同时检查query_self与past_history。",
        "",
        "## 先看",
        "1. head_roles.csv：每个确认head的证据/历史/错误来源作用。",
        "2. layer_competition.csv：同层正负head竞争。",
        "3. evidence_forward.csv：证据在各层的局部支持。",
        "4. propagation_delta.csv.gz：删路径后的影响如何被后续层保留、抵消或反转。",
        "5. same_question_head_deltas.csv：同题有依据/无依据的同head差异。",
        "6. binding_completeness.csv：原始condition/value作用；binding_state仅作方向描述，不作机制结论。",
        "7. binding_sensitivity_counts.csv：0.01/0.02/0.05三档效应阈值的部分绑定敏感性。",
        "8. binding_panel_consistency.csv：natural与forced-common-wording面板是否复现同一binding状态。",
        "9. head_interactions.csv（v2）：full/单头/双头四世界交互，同时报告首词与整段候选。",
        "condition/value是来源位置集合，V已含上下文；直接condition效应弱不证明条件缺失。",
        "binding_completeness旧字段仅为两个直接效应幅度比，不是语义绑定完整度。",
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
            "同形不同模型也不按head编号对齐；缺少已核验checkpoint身份时保持未对齐。",
        ]
    (output / "REPORT_FLOW_zh.md").write_text("\n".join(lines), encoding="utf-8")


def report_flow(output, supervised_path=None):
    output = Path(output)
    roles = role_table(output)
    competition = layer_competition(output)
    evidence = evidence_layers(output)
    paths = propagation_delta(output)
    deltas = same_question_deltas(roles)
    binding = binding_table(roles)
    alignment = supervised_alignment(roles, supervised_path)

    roles.to_csv(output / "head_roles.csv", index=False)
    competition.to_csv(output / "layer_competition.csv", index=False)
    evidence.to_csv(output / "evidence_forward.csv", index=False)
    paths.to_csv(output / "propagation_delta.csv.gz", index=False)
    deltas.to_csv(output / "same_question_head_deltas.csv", index=False)
    binding.to_csv(output / "binding_completeness.csv", index=False)
    write_binding_robustness(binding, output)
    alignment.to_csv(output / "supervised_alignment.csv", index=False)
    write_report(output, roles, alignment)

    archive = output / "flow_review.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(output.glob("*")):
            if path.is_file() and path.name != archive.name and path.suffix != ".npz":
                stream.add(path, arcname=path.name)

    print(roles[roles.final_all_context.notna()][KEYS + [
        "layer", "head", "final_all_context", "final_evidence",
        "final_history", "downstream_reversal",
    ]].to_string(index=False))
