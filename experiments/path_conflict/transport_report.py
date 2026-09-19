"""Report finite effects without turning local interactions into truth labels."""

import json
import tarfile

import pandas as pd


def collect_panels(output, name):
    frames = [pd.read_csv(path) for path in sorted((output / "panels").glob("*/" + name))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def report_transport(output):
    config = json.loads((output / "transport_config.json").read_text())
    tables = {}
    for name in ("edges", "interventions", "interactions", "mediation"):
        table = collect_panels(output, name + ".csv")
        table.to_csv(output / ("transport_" + name + ".csv"), index=False)
        tables[name] = table
    pairs = tables["interactions"]
    selected = pairs
    if not pairs.empty:
        selected = pairs[(pairs.metric == "sequence_margin") & (pairs.dose == 1)]
        selected = selected[["case_id", "side", "panel", "left", "right", "kind",
                             "left_support", "right_support", "joint_support", "interaction"]]
    text = ["# 原生消息运输与多头交互", "", f"协议：`{config['protocol']}`。", "",
            "选点来自最终候选差值的原模型梯度，来源角色在选点之后追加。",
            "这是人工核验候选上的机制实验，不是无监督自然检测成绩。", "",
            "- `edges`：逐层、逐头、接收位置、来源位置；正负门导数、质量、value energy。",
            "- `interventions`：两个删除剂量及等范数随机方向；包含首词概率和完整候选 logp。",
            "- `interactions`：四个世界的原始 margin、条件作用及 J = D_A + D_B - D_AB。",
            "- `mediation`：上游删除后恢复下游消息或同位置 MLP；同时报告同世界恢复误差。", "",
            "正 J 只表示该对比、该干预下的非加和作用，不自动代表正确性协同。",
            "恢复某消息不等于恢复一条唯一因果路径；多个恢复量不可相加。",
            "`kind` 是干预前梯度符号，仅用于配对，不随结果重新命名。",
            "小效应需与 sham_delta、重放误差及随机方向控制一起看；不靠阈值挑成功。",
            f"选点目标为 {config['objective']}；首词和整段效应分别保存。",
            "整段候选仍有长度/措辞差异；不同自然前缀不是只改变真假的控制实验。", "",
            "## 完整候选的双头联合结果", "", "```csv", selected.to_csv(index=False), "```"]
    (output / "REPORT_TRANSPORT_zh.md").write_text("\n".join(text), encoding="utf-8")
    with tarfile.open(output / "flow_review.tar.gz", "w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".json", ".md"):
                archive.add(path, arcname=str(path.relative_to(output)))
    if not selected.empty:
        print(selected.to_string(index=False))
    print(f"Review: {output / 'flow_review.tar.gz'}", flush=True)
