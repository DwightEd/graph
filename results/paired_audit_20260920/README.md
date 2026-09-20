# 上传 v3 输出的配对机制重分析

输入：用户提供的 `flow_review (1).tar.gz`，读取其中
`transport_interventions.csv`、`transport_interactions.csv`、`transport_mediation.csv`。
本目录仅写派生表，没有改写原始结果或重新运行 8B。

固定口径：完整候选 logp 差、删除剂量 1、显著幅度描述阈值 .05 nats。
这里的“显著幅度”是预设数值门槛，不是统计显著性。

| 文件 | 结果 |
|---|---|
| legacy_finite_effects.csv | 60 次删除，2 次梯度/有限效应在门槛以上异号 |
| legacy_conditional_effects.csv | 24 组双删，11 组单独作用异号；2 个 opposed 提议未通过有限异号标准 |
| legacy_restoration_review.csv | 保留恢复量及各自 sham；最大绝对 sham 为 .07370379 nats |
| legacy_summary.json | 上述计数及实验范围 |

仅有 2 个独立来源、6 个面板。不能将 11/24 当作幻觉中冲突机制的流行率。
两个单删异号表示对所选候选差的相反作用，不证明组件物理上相互抑制。
v3 没有为这里所有双删保存联合 sham，表中显式标记此限制。
恢复量超过自身 sham 的描述标记也不代替新协议原值写回校验。

新方案和一键运行见 [PAIRED_MECHANISM_20260920.md](../../docs/PAIRED_MECHANISM_20260920.md)。
