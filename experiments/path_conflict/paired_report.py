"""Source-paired summaries; distinguish observed effects from semantic mechanisms."""

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from .paired_exports import adaptation_coverage, export_readouts


def classify_interactions(frame, minimum_effect):
    result = frame.copy()
    left, right = result.left_support, result.right_support
    result["opposed_finite_effects"] = (left * right < 0) & (left.abs() > minimum_effect) & (right.abs() > minimum_effect)
    result["left_direction_changes"] = (left * result.left_conditional < 0) & (
        np.minimum(left.abs(), result.left_conditional.abs()) > minimum_effect)
    result["right_direction_changes"] = (right * result.right_conditional < 0) & (
        np.minimum(right.abs(), result.right_conditional.abs()) > minimum_effect)
    result["nonadditive_effect"] = result.interaction.abs() > minimum_effect
    result["small_singles_large_joint"] = (left.abs() <= minimum_effect) & (
        right.abs() <= minimum_effect) & (result.joint_support.abs() > minimum_effect)
    return result


def paired_differences(frame, keys, value):
    index = ["source_id", "case_id", *keys]
    # A pair is admitted only when the SAME head/source/phase exists on both
    # sides. No unobserved head, missing phase, or failed sham becomes a zero.
    table = frame.pivot(index=index, columns="side", values=value)
    if not {"supported", "unsupported"}.issubset(table.columns):
        return pd.DataFrame(columns=[*index, "unsupported_minus_supported"])
    table = table.dropna(subset=["supported", "unsupported"])
    table["unsupported_minus_supported"] = table.unsupported - table.supported
    return table.reset_index()


def source_intervals(paired, keys, repeats=2000, seed=0):
    rows = []
    rng = np.random.default_rng(seed)
    for identity, group in paired.groupby(keys, dropna=False):
        values = group.groupby("source_id").unsupported_minus_supported.mean().to_numpy()
        low, high = np.nan, np.nan
        if len(values) >= 2:
            samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(samples, [.025, .975])
        identity = identity if isinstance(identity, tuple) else (identity,)
        rows.append(dict(zip(keys, identity), sources=len(values), pairs=len(group),
                         effect=float(values.mean()), lower=low, upper=high,
                         inference="exploratory_source_bootstrap" if len(values) >= 2 else "one_source_no_interval"))
    return pd.DataFrame(rows)


def collect(output, name):
    frames = []
    for path in sorted((output / "pairs").glob("*/*/*/" + name + ".csv")):
        frame = pd.read_csv(path, dtype={"numeric_ok": "boolean"})
        # Before/onset can have no persistence or eligible adaptation pairs.
        # Their identity-only headers must not turn bool columns into objects.
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def control_counts(tables):
    counts = {}
    for name in ("effects", "interactions", "adaptation", "persistence"):
        frame = tables[name]
        flags = frame.numeric_ok if not frame.empty else pd.Series(dtype="boolean")
        counts[name] = dict(rows=len(frame), passed=int(flags.eq(True).sum()),
                            failed=int(flags.eq(False).sum()), missing=int(flags.isna().sum()))
    return counts


def compare_relationships(output, tables):
    comparisons = {
        "interactions": (["left", "right", "relation", "dose"],
                         ["interaction", "left_conditional", "right_conditional", "joint_support"]),
        "adaptation": (["early", "late", "site"], ["restoration_gain"]),
        "persistence": (["layer", "head", "selection"], ["support"]),
    }
    for name, (keys, metrics) in comparisons.items():
        frame = tables[name]
        if frame.empty:
            continue
        frame = frame[frame.numeric_ok.eq(True).fillna(False)]
        rows = [paired_differences(frame, ["phase", "readout", *keys], metric).assign(metric=metric)
                for metric in metrics]
        pd.concat(rows, ignore_index=True).to_csv(output / ("paired_" + name + ".csv"), index=False)


def report_pairs(output, minimum_effect=.05, repeats=2000):
    tables = {}
    for name in ("coverage", "effects", "interactions", "adaptation", "persistence"):
        frame = collect(output, name)
        if name == "interactions" and not frame.empty:
            frame = classify_interactions(frame, minimum_effect)
        frame.to_csv(output / (name + ".csv"), index=False)
        tables[name] = frame
    compare_relationships(output, tables)
    effects = tables["effects"]
    if not effects.empty:
        selected = effects[effects.numeric_ok.eq(True).fillna(False) & effects.control.fillna("").eq("")]
        keys = ["phase", "layer", "head", "source_group", "dose", "readout", "selection"]
        paired = paired_differences(selected, keys, "support")
        paired.to_csv(output / "paired_effects.csv", index=False)
        source_intervals(paired, keys, repeats).to_csv(output / "source_intervals.csv", index=False)
    inventory = pd.read_csv(output / "pair_inventory.csv")
    controls = control_counts(tables)
    adaptation_coverage(output, tables["adaptation"]).to_csv(output / "adaptation_coverage.csv", index=False)
    readouts = export_readouts(output)
    summary = dict(sources=int(inventory.source_id.nunique()), pairs=int(inventory.case_id.nunique()),
        measured_phases=int(effects[["case_id", "side", "phase"]].drop_duplicates().shape[0]) if not effects.empty else 0,
        finite_effect_rows=len(effects), interaction_rows=len(tables["interactions"]),
        failed_control_rows={name: counts["failed"] for name, counts in controls.items()},
        control_rows=controls,
        readout_export=readouts,
        row_count_note="rows include repeated heads, doses and controls; independent unit is source",
        minimum_effect_nats=minimum_effect, purpose="label_assisted_mechanism_audit_not_detector_evaluation")
    write_report(output, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def write_report(output, summary):
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = ["# 重采样正负配对机制审计", "", "```json", json.dumps(summary, indent=2), "```", "",
        "- onset：人工核验的完整候选 log 概率差；长度和措辞仍是限制。",
        "- before/back_half/post：实际下一 token 的 log 概率，不冒充事实真假 readout。",
        "- post_claim 只是所核验子句之后，不自动视为已经恢复正确。",
        "- 同一物理 head、来源分组和阶段先在同题内作差，再按独立 source 汇总。",
        "- 正负侧自然历史不同；局部 supported 不代表此前历史全部正确。",
        "- history 包括已输入回答的当前 query；另报 query_self 与 prior_history，避免把 self 效应叫作远历史复用。",
        "- 单头作用、条件作用、联合删除和 J 分开保存；J 正负不直接命名协同/抑制。",
        "- interactions.numeric_ok 继承两项单独干预的 sham；没有单独执行联合 sham。",
        "- adaptation：删除上游后恢复下游，sham 通过才解释作用；不等于唯一自然中介比例。",
        "- adaptation_coverage.csv 单列跨层配对覆盖；没有合格跨层配对不等于没有下游恢复。",
        "- persistence：只在起点干预后固定 teacher-forcing 文本，不证明自由生成自我强化。",
        "- screen 中 prefix-RAUQ 是因果化对照，不是原文完整回答选 head 的复现。",
        "- 不同 phase 的 readout 不同，不将其 margin 连接为同尺度事实曲线。", "",
        "baseline.json 与 worlds/*.json 从已有 NPZ 的 record 导出，包含各候选逐 token/总和/均值概率，",
        "不包含激活张量；仅运行 report 即可导出，无需重跑模型。readout_inventory.csv 标记缺失，",
        "旧结果包没有 NPZ 或已导出的 JSON 时不能补算这些概率。", "",
        "指针、地址、载荷的语义分离仍需独立交换实验；当前结果不自动提供这种命名。",
        "完整检测对象、证伪条件和文献依据见 docs/PAIRED_MECHANISM_20260920.md。"]
    (output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    with tarfile.open(output / "paired_review.tar.gz", "w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".json", ".md"):
                archive.add(path, arcname=str(path.relative_to(output)))


def review_previous(input_directory, output, minimum_effect=.05):
    """Recompute the uploaded v3 finite effects; do not reinterpret proposals."""
    source = Path(input_directory)
    effects = pd.read_csv(source / "transport_interventions.csv")
    effects = effects[(effects.treatment == "cut") & (effects.dose == 1)].copy()
    effects["gradient_finite_disagreement"] = (effects.final_linear_support * effects.sequence_support < 0) & (
        effects.final_linear_support.abs().gt(minimum_effect) & effects.sequence_support.abs().gt(minimum_effect))
    effects.to_csv(output / "legacy_finite_effects.csv", index=False)
    pairs = pd.read_csv(source / "transport_interactions.csv")
    pairs = pairs[(pairs.metric == "sequence_margin") & (pairs.dose == 1)].copy()
    pairs = pairs.rename(columns={"left_when_right_absent": "left_conditional",
                                  "right_when_left_absent": "right_conditional", "kind": "proposal_kind"})
    pairs = classify_interactions(pairs, minimum_effect)
    pairs["numeric_control_status"] = "joint_sham_not_saved_in_v3"
    pairs.to_csv(output / "legacy_conditional_effects.csv", index=False)
    mediation = pd.read_csv(source / "transport_mediation.csv")
    mediation = mediation[mediation.metric == "sequence_margin"].copy()
    mediation["restore_exceeds_own_sham"] = mediation.restoration_gain.abs() > mediation.sham_delta.abs() + minimum_effect
    mediation.to_csv(output / "legacy_restoration_review.csv", index=False)
    summary = dict(sources=int(effects.source_id.nunique()), panels=int(effects.groupby(["case_id", "side", "panel"]).ngroups),
        finite_deletions=len(effects), gradient_finite_disagreements=int(effects.gradient_finite_disagreement.sum()),
        tested_pairs=len(pairs), opposed_finite_pairs=int(pairs.opposed_finite_effects.sum()),
        proposal_opposed_but_not_finite_opposed=int(((pairs.proposal_kind == "opposed") & ~pairs.opposed_finite_effects).sum()),
        max_restoration_sham=float(mediation.sham_delta.abs().max()),
        minimum_effect_nats=minimum_effect, inference="descriptive_existing_results_no_new_model_run")
    (output / "legacy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
