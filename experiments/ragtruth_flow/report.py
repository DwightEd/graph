"""Population report: routing screen plus held-out functional confirmation."""

from pathlib import Path
import json
import tarfile

import pandas as pd


def paired_functional(table):
    roles = {
        "onset": ("error_onset", "control_onset"),
        "continuation": ("error_continuation", "control_continuation"),
    }
    rows = []
    for phase, (error_role, control_role) in roles.items():
        error = table[table.role == error_role]
        control = table[table.role == control_role]
        keys = ["response_id", "source_id", "layer", "head", "source_group"]
        merged = error.merge(control, on=keys, suffixes=("_error", "_control"))
        for row in merged.itertuples():
            rows.append(dict(
                phase=phase,
                source_id=row.source_id,
                layer=row.layer,
                head=row.head,
                source_group=row.source_group,
                final_support_difference=row.final_support_error - row.final_support_control,
                local_support_difference=row.local_support_error - row.local_support_control,
                attention_mass_difference=row.attention_mass_error - row.attention_mass_control,
                reversal_error=row.downstream_reversal_error,
                reversal_control=row.downstream_reversal_control,
            ))
    return pd.DataFrame(rows)


def competition(table):
    rows = []
    for key, group in table.groupby(["response_id", "role", "source_group"]):
        values = group.final_support.to_numpy()
        positive = values[values > 0].sum()
        negative = -values[values < 0].sum()
        total = positive + negative
        rows.append(dict(
            response_id=key[0],
            role=key[1],
            source_group=key[2],
            positive_support=positive,
            negative_support=negative,
            competition=0 if total == 0 else 2 * min(positive, negative) / total,
        ))
    return pd.DataFrame(rows)


def summarize_pairs(pairs):
    if pairs.empty:
        return pd.DataFrame()
    by_source = pairs.groupby(
        ["source_id", "phase", "layer", "head", "source_group"], as_index=False
    ).mean(numeric_only=True)
    return by_source.groupby(
        ["phase", "layer", "head", "source_group"], as_index=False
    ).agg(
        final_support_difference=("final_support_difference", "mean"),
        local_support_difference=("local_support_difference", "mean"),
        attention_mass_difference=("attention_mass_difference", "mean"),
        reversal_error=("reversal_error", "mean"),
        reversal_control=("reversal_control", "mean"),
        sources=("source_id", "nunique"),
    )


def report(output):
    output = Path(output)
    effects = pd.read_csv(output / "functional_effects.csv.gz")
    pairs = paired_functional(effects)
    pair_summary = summarize_pairs(pairs)
    comp = competition(effects)

    pairs.to_csv(output / "functional_pair_differences.csv.gz", index=False)
    pair_summary.to_csv(output / "functional_pair_summary.csv", index=False)
    comp.to_csv(output / "functional_competition.csv", index=False)

    screen = pd.read_csv(output / "screen_head_effects.csv")
    selected = pd.read_csv(output / "selected_heads.csv")
    targets = pd.read_csv(output / "confirm_targets.csv")
    status = dict(
        screen_groups=len(screen),
        selected_heads=len(selected),
        confirm_targets=len(targets),
        confirm_sources=int(targets.source_id.nunique()),
        source_token_coverage=float((targets.source_tokens > 0).mean()),
        interpretation=(
            "screen=train attention routing; confirm=test teacher-forced Llama-3.1 observer. "
            "Functional support is support for the actually generated token, not a corrected candidate."
        ),
    )
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    lines = [
        "# RAGTruth Evidence ↔ Target population audit",
        "",
        "TRAIN：全量attention cache按金标错误span与同答正常区间筛查逐head路由差异。",
        "TEST：冻结筛出的head，在匹配错误/正常token上回放Llama-3.1 observer，删除真实A·V·W_O写入。",
        "final_support>0表示该来源/head帮助当前实际生成token；<0表示抑制该token。",
        "错误token上的正history support表示帮助已经生成的幻觉token，不代表事实正确。",
        "",
        "先看 screen_head_effects.csv、functional_pair_summary.csv、functional_competition.csv。",
        "",
        "RAGTruth没有正确替代文本；本实验不伪造correct-vs-wrong candidate。",
        "source组只使用source_info在保存prompt中能精确定位的token；覆盖为0时不把整个prompt冒充证据。",
        "这里审计统一Llama-3.1 observer对既有回答的teacher-forced处理，不是六个原生成器自身的内部因果机制。",
        "TRAIN筛查和TEST确认都属于标签辅助机制研究，不是无监督检测成绩。",
    ]
    (output / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")

    archive = output / "ragtruth_flow_review.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(output.glob("*")):
            if path.is_file() and path.name != archive.name and path.suffix != ".npz":
                stream.add(path, arcname=path.name)

    print(status, flush=True)
    print(pair_summary.to_string(index=False), flush=True)
