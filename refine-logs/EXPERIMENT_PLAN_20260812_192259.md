# 实验计划

**问题**：注意力中与幻觉相关的有效信号究竟来自路由状态、时序变化，还是精确 token 邻接与 GNN 消息传递？  
**方法主张**：先用 MART 在无 GNN 条件下建模 prompt/history 路由、集中度、层间漂移和时间创新；只有精确 source 拓扑带来可复现增益时才保留 GNN。  
**日期**：2026-08-12

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：机制对齐的路由轨迹可用于无监督幻觉检测 | 将既有统计发现转化为全 token、可泛化的检测模型 | MART 在 held-out test 上优于位置/长度基线，且 source-cluster bootstrap 的 AUPRC 增量置信区间不跨 0 | B1, B2 |
| C2：GNN 只在精确 source 拓扑有额外信息时才必要 | 防止把 attention 特征有效误写成图结构有效 | source-GNN 优于 MART 与 no-message；保持权重、target、RP/RR 的 source shuffle 显著降低性能 | B3 |

需要排除的替代解释：结果只来自相对位置、整体 attention 质量、参数量，或在 test 上重新拟合异常分布。

## Paper Storyline

- 主文必须证明：MART 的 inductive train-fit/test-score 结果；GNN/no-message/source-shuffle 的配对比较。
- 附录支持：attention floor、position bins、kNN 邻居数、不同 generator/data source 的稳健性。
- 明确删除：单样本 t-SNE、整链均值、未显著的 history lag/density 作为核心输入。

## Experiment Blocks

### B1：MART sanity 与主结果

- Claim tested：C1。
- Dataset / split：RAGTruth canonical train 拟合，canonical test 冻结打分；所有 response token。
- Compared systems：relative-position only、旧 scalar statistics、MART。
- Metrics：token AUROC/AUPRC（主），response top-20%-mean（次），按 source cluster bootstrap 的差值区间。
- Setup：MART 使用原始 CSR；不读 labels；3 seeds 仅用于并列/采样算法，确定性 MART 报单次结果。
- Success criterion：MART 相对 position-only 的 AUPRC 增量区间不跨 0，并在主要 data source 上方向一致。
- Failure interpretation：现有路由发现是回顾性相关，不能组成有效异常表征。
- Priority：MUST-RUN。

### B2：MART 表征消融

- Claim tested：C1 的机制归因。
- Compared systems：完整 MART；去掉 prompt fraction/anchor；去掉 entropy；去掉 layer drift；去掉 EMA innovation；跨 layer/head 直接平均。
- Metrics：与完整 MART 的 paired ΔAUROC/ΔAUPRC；onset 对齐分数轨迹仅作解释。
- Success criterion：至少一个机制块产生稳定增益，且不是 relative position 单独解释。
- Failure interpretation：MART 可能仅是高维密度估计或位置校准，应简化模型。
- Priority：MUST-RUN。

### B3：图与 GNN 必要性

- Claim tested：C2。
- Compared systems：MART；threshold/no-message；typed-mass-cover/no-message；threshold/GNN；typed-mass-cover/GNN；最佳 GNN 的 source-shuffle。
- Metrics：同一 token 上 paired ΔAUROC/ΔAUPRC 和 source-cluster bootstrap。
- Success criterion：GNN 同时优于 MART、同 support 的 no-message，且 source-shuffle 使性能下降。
- Failure interpretation：保留无 GNN MART；结论限定为路由状态有效，不能声称精确图拓扑有效。
- Priority：MUST-RUN。

### B4：稳健性与失败分析

- Dataset：按 task type、data source、generator 分组；floor sensitivity 需要重新抽取时放入附录。
- Metrics：分组 prevalence/AUPRC、长度匹配结果、误报/漏报轨迹。
- Priority：NICE-TO-HAVE。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 验证数据与指标 | MART 小规模 train/test、覆盖率检查 | score NPZ 覆盖全部 test token 且 labels 只在 evaluate 打开 | CPU/GPU 分钟级 | 训练参考库内存；必要时分块/子采样 |
| M1 | 建立无 GNN 锚点 | position-only、statistics、MART | MART 未优于简单基线则停止扩展 GNN | 小时内 | test leakage；严格冻结 checkpoint |
| M2 | 机制消融 | B2 五个删除实验 | 确认有效块，否则简化 | 小时级 | 多重比较；预先冻结主指标 |
| M3 | 判断 GNN 必要性 | 2×2 support/message + shuffle | 满足 B3 全部条件才保留 GNN | 数个 GPU-hour/seed | GNN 训练方差；使用同 seed/source split |
| M4 | 稳健性 | 分组、长度匹配、floor | 不改变主结论范围 | 视重抽数据而定 | floor 重抽成本高 |

## Compute and Data Budget

- MART：16 维 token 表征；特征可在 GPU 计算，密度拟合在 CPU/scikit-learn。
- GNN：RTX 4090，3 seeds 只用于通过 M1 后的决定性比较。
- 最大瓶颈：kNN 训练参考向量与精确逐 token leave-one-out GNN 打分。

## Risks and Mitigations

- 回顾性发现泄漏到分数方向：MART 使用双侧 train-only novelty，不手工规定“prompt 高/低即异常”。
- 位置/长度混杂：position-bin calibration，并加入 position-only、长度匹配对照。
- 图性能来自边属性而非拓扑：no-message 与 source-shuffle 分别隔离消息传递和 source endpoint。
- t-SNE 误导：可视化只解释冻结 embedding，不作为成功标准。

## Final Checklist

- [x] Main paper tables are covered
- [x] Novelty is isolated
- [x] Simplicity is defended
- [x] Frontier contribution is explicitly conditional, not assumed
- [x] Nice-to-have runs are separated from must-run runs
