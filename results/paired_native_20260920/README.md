# 本轮 paired_review 的 CPU 复核

输入为 2026-09-20 上传的 `paired_review.tar.gz`，126 个归档成员，解压后共 5,714,290 bytes。复核脚本只读该包；没有新 LLM 运行。原始归档保留不变。

解释和实验设计见 [完整报告](../../docs/PAIRED_RESULTS_20260920.md)。

| 文件 | 内容 |
|---|---|
| onset_effects.csv | 所有 onset 单头/来源作用，保留两种剂量和随机控制；仅省去冗长来源索引列 |
| onset_four_worlds.csv | 所有 onset 四世界，保留读出及左右条件作用 |
| restoration.csv | 所有恢复试验；增加恢复前后与原分数的距离和越过原值标志 |
| persistence_vs_current.csv | 同一案例/侧/位置/head 的起点传播效应与当前删除效应 |
| phase_readouts.csv | 16 个阶段的基线及候选长度；本包的明细概率不可用 |
| adaptation_coverage.csv | 头饰测了两个跨层 pair；洋葱没有合格跨层 pair |
| numeric_checks.json | 四张原表的控制计数和干预公式残差 |
| head_review.png | 逐 head 的对照图；D 面板采用后续实际 token 读出 |

图仅展示明确指定的候选头及代表性关系；完整表保留随机选头和等范数扰动。`supported` 仅表示人工复核的当前子句有依据，不是整条回答为正确。两来源不提供泛化性能或因果贡献百分比。
