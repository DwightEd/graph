# 执行追踪

| ID | 目的 | 状态 | 位置/证据 |
|---|---|---|---|
| R0 | 设计→规划先行 | DONE | METHOD_DESIGN_v1及PLAN_v1早于源码/新评分；后续修正独立文件 |
| R1 | 53答同世界原生完整A/V/head捕获 | DONE | outputs/flow_latent_20261007_capture/execution.json；53forward、565秒、17GiB |
| R2 | v1直接条件/node/rewire | DONE | 初轮及一致MAP各3seed；所有分数先冻结后评价 |
| R3 | v2来源有限路径 | DONE | 初轮及一致MAP各3seed；未确认C1/C2 |
| R4 | v3删除历史解释 | DONE | 初轮及一致MAP各3seed；均未确认真实边增量 |
| R5 | 独立数值复算 | DONE | numeric_audit.json：114 AUC/AP、69拟合density（含重现检查）、4样例全head/raw-source及派生路径复算 |
| R6 | 高保真度追加第三版 | DONE | head16派生53答DONE（0新8B）；3seed评价完成，真实图均值.598814/乱图.600508 |
| R6b | 同容量节点对照与绑定修正重现 | DONE | 3+3模型，旧分数完全不变；重现新增分数最大误差0；cache/model/容量契约PASS |
| R7 | fresh实验审计 | DONE | /root/flow_latent_audit；WARN，同家族provisional；保留收敛/原始freeze范围等限制；main交付验证记录归共享HANDOFF.md |

原始目录均位于graph/outputs。原user archives/delete及routing_likelihood未修改。本轮不将统计异常叫事实概率；完整attention行Dirichlet生成模型尚未实现。

实际69模型=初轮27（非一致MAP）＋修正27＋head16 9＋初次对照3＋绑定重现3。53原生8B前向，0后续8B，GPU释放。匹配乱边/同容量节点对照均未确认真实端点收益，C1/C2未获支持。
