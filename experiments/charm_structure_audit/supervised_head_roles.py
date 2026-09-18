"""Explain the supervised detector's statistical role for each LLM head."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .lda_data import read_audit_inputs


def load_weight(path):
    with np.load(path, allow_pickle=False) as saved:
        return saved["weight"]


def head_statistics(values, prompt, labels, self_weight, prompt_weight, heads):
    normal = labels == 0
    error = labels == 1
    self_gap = values[error].mean(0) - values[normal].mean(0)
    prompt_gap = prompt[error].mean(0) - prompt[normal].mean(0)
    channels = np.arange(values.shape[1])

    frame = pd.DataFrame(dict(
        layer=channels // heads,
        head=channels % heads,
        self_gap=self_gap,
        prompt_gap=prompt_gap,
        self_weight=self_weight,
        prompt_weight=prompt_weight,
        self_risk_contribution=self_gap * self_weight,
        prompt_risk_contribution=prompt_gap * prompt_weight,
    ))
    frame["routing_pattern"] = np.select(
        [
            (frame.self_gap > 0) & (frame.prompt_gap < 0),
            (frame.self_gap < 0) & (frame.prompt_gap > 0),
            (frame.self_gap > 0) & (frame.prompt_gap > 0),
            (frame.self_gap < 0) & (frame.prompt_gap < 0),
        ],
        ["self_up_prompt_down", "self_down_prompt_up", "both_up", "both_down"],
        default="mixed_or_flat",
    )
    return frame


def summarize(frame):
    ranked = frame.assign(
        size=frame.self_risk_contribution.abs()
    ).sort_values("size", ascending=False)
    total = ranked.self_risk_contribution.abs().sum()
    rows = []
    for count in (8, 16, 32, 64):
        selected = ranked.head(count)
        rows.append(dict(
            top_heads=count,
            contribution_mass=float(selected.self_risk_contribution.abs().sum() / total),
            self_up_prompt_down=int(
                (selected.routing_pattern == "self_up_prompt_down").sum()
            ),
            self_down_prompt_up=int(
                (selected.routing_pattern == "self_down_prompt_up").sum()
            ),
        ))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/charm_structure_audit_qa/QA/seed_0")
    parser.add_argument("--prepared", default="outputs/charm_structure_audit_qa/data")
    parser.add_argument("--lda-output")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    folder = Path(args.lda_output) if args.lda_output else Path(args.root) / "audit_lda_prompt"
    output = Path(args.output) if args.output else Path(args.root) / "audit_supervised_head_roles"
    output.mkdir(parents=True, exist_ok=True)

    config = type("Args", (), dict(
        root=args.root,
        prepared=args.prepared,
        lda_prompt=True,
    ))()
    data, _ = read_audit_inputs(config)
    values, table, prompt, geometry = data["test"]
    labels = table.gold.to_numpy()

    self_weight = load_weight(folder / "parameters/full.npz")
    prompt_weight = load_weight(folder / "parameters/prompt_only.npz")
    frame = head_statistics(
        values, prompt, labels, self_weight, prompt_weight, geometry[1]
    )
    frame.to_csv(output / "head_roles.csv", index=False)
    summarize(frame).to_csv(output / "summary.csv", index=False)

    top = frame.assign(
        size=frame.self_risk_contribution.abs()
    ).sort_values("size", ascending=False).head(24)
    lines = [
        "# 监督检测器的head统计角色",
        "",
        "self_gap/prompt_gap是TEST错误减正常的均值差；weight来自FIT标签训练的LDA。",
        "self_risk_contribution=self_gap*weight，解释监督读出，不是原LLM因果贡献。",
        "prompt是逐head保留prompt质量，没有区分适用证据与无关prompt。",
        "head编号只属于该检测器输入模型，不能与另一模型族按编号对齐。",
        "",
        "## 贡献最大的head",
        "",
        top.drop(columns="size").to_string(index=False),
    ]
    (output / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(top[[
        "layer", "head", "self_gap", "prompt_gap",
        "self_risk_contribution", "routing_pattern",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
