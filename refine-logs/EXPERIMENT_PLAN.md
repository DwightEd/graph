# 实验方案：无监督因果关系通道图检测

**Problem**: 如何把全层全头 attention 边聚合成 token 节点向量，并只用无标签数据检测 RAGTruth 幻觉。
**Method Thesis**: 先用关系/层头感知的 masked graph autoencoder 学习节点表示，再在表示的时序创新量上拟合抗污染多模态密度；标签只用于最终评估。
**Date**: 2026-08-12

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: attention 的具体连接、边权和 layer/head 通道经过 GNN 聚合后含有可用于无监督幻觉检测的信号 | 这是构图必要性的核心主张 | OOF token AUPRC 相对最强 label-free baseline 的 prompt-cluster bootstrap CI 下界大于 0，并超过预注册最小实用增益 | B1, B2 |
| C2: 异常是条件于多种正常生成动态的时序偏离，而不是单一“正常流形”距离 | 避免正常与错误共享局部变化时失效 | K>1、时序上下文和抗污染项同时改善 held-out unlabeled NLL 与 OOF 检测 | B3 |
| Anti-claim: 提升只来自 attention 边缘统计、位置、task/source 或参数量 | 排除把手工特征或域差异误称为图机制 | no-message、edge-marginal MLP、rewire、relation/channel collapse、分层及 leave-one-domain-out 对照 | B2, B4 |

## Paper Storyline

- Main paper must prove: 完整稀疏 attention 边经过 learned message passing 比手工边缘统计更有效；分数来自 held-out、label-blind 的节点表示。
- Appendix can support: 不同 K、污染率、embedding dimension 和 mask rate 的敏感性。
- Experiments intentionally cut: 联合训练 encoder+density、normalizing flow、DeepSVDD、超图、多种任意 residual 加权。

## Method Contract

### 输入图

直接使用 canonical CSR，不复制第二份 graph 数据：

```text
token_ids, response_idx, attention_diagonal,
response_row_ptr, response_column_indices, response_values
```

每条保留项解码为 `(source, target, layer, head, weight)`。仅允许 `source < target`；relation 为 prompt→response 或 response→response。缺失项表示 `<= attention_floor`，不能当作精确零。

### 节点编码器

- 初始节点属性：全层全头 attention diagonal、prompt/response role、归一化位置。
- edge embedding：稀疏 channel embedding 的 attention-weighted sum，加 relation embedding 和边质量；不在 prompt/history 分支内分别归一化。
- 两层有向 message passing，用 `index_add_` 在 GPU 聚合；只从更早 token 传到当前 token。
- 输出 32D response-token embedding `z_t`。

“causal”仅指 prefix-causal/online，不是因果推断。必须通过 indexing audit 与 prefix-invariance test，证明 token `t` 的表示没有未来边。

### 自监督训练

按 `(target, relation)` 分层遮蔽 15% 边，并遮蔽 layer/head 通道。编码器必须从剩余图重建：

1. 被遮蔽的 causal support（正边与同 relation 的合法负边）；
2. 被遮蔽的逐 channel attention weight；
3. source distribution 与 censored `OTHER` mass。

重建目标的输入中不得保留对应 masked edge/channel payload，避免旧 autoencoder 的目标泄漏。

### 无监督异常模型

冻结 encoder。用前缀 GRU 和位置形成 context，拟合 K=4 的对角 Student-t mixture：

```text
p_in(z_t | z_<t, position) = Σ_k pi_k StudentT(z_t; context_shift + mode_k, scale_k)
```

训练密度为 `(1-epsilon) p_in + epsilon q0`，`q0` 是固定宽尾分布，MVP 设 `epsilon=0.05`。最终异常分数只用 `-log p_in`，污染后验仅作诊断。用无标签 calibration token 转成经验尾概率。

## Experiment Blocks

### B1: 主检测结果

- Claim tested: C1。
- Dataset / split / task: RAGTruth canonical train/test，所有样本；按 `source_id` 分组，禁止同一 prompt 跨 fold。
- Compared systems: edge-marginal robust GMM、旧 32D 手工 graph features + Isolation Forest、完整方法。
- Metrics: primary token AUPRC；secondary AUROC、span AP、answer AUPRC、onset recall@固定每千 correct-token 误报、检测延迟。
- Setup: 5-fold OOF，3 seeds；训练/选择只看 reconstruction 与 held-out unlabeled NLL。
- Success criterion: 对最佳无监督 baseline 的 cluster-bootstrap ΔAUPRC 95% CI 下界 > 0，且绝对 AUPRC 超过 prevalence 和预注册实用阈值。
- Failure interpretation: 若不满足，不能声称构图解决了幻觉检测。
- Table / figure target: 主结果表。
- Priority: MUST-RUN。

### B2: 构图必要性

- Claim tested: C1 与 anti-claim。
- Compared systems: no-message；edge-marginal MLP；unweighted topology GNN；attention-weighted GNN；full channel-aware GNN；relation-preserving source rewire；relation collapse；layer/head mean。
- Metrics: 与完整方法同一 OOF token 上的 paired cluster-bootstrap ΔAUPRC，Holm 校正。
- Success criterion: full 模型优于 no-message/edge-marginal，rewire 明显破坏性能；否则不把结果归因于邻接关系。
- Failure interpretation: relation/channel collapse 无损则收缩对应机制主张。
- Table / figure target: 消融表与 occlusion Δscore 图。
- Priority: MUST-RUN。

### B3: 多模态时序异常

- Claim tested: C2。
- Compared systems: K=1 vs K=4；无 GRU；time shuffle；无 q0；同一 `z` 上 robust GMM/Isolation Forest/LOF。
- Metrics: held-out unlabeled NLL、OOF token AUPRC、epsilon 敏感性排名相关。
- Success criterion: K=4 和真实时间顺序同时改善 NLL 与检测；epsilon 扰动后 score rank correlation >= 0.8。
- Failure interpretation: time shuffle 无损则不声称时序机制；K=1 无损则不声称多模态必要。
- Table / figure target: density-head 消融表。
- Priority: MUST-RUN。

### B4: 全量分层统计与可视化

- Claim tested: anti-claim 与机制解释。
- Evaluation population: 所有 OOF response tokens，不只错误 token；标签仅在 scores/embeddings 固定后载入。
- Statistics: prompt/source cluster bootstrap；task、data_source、generator、长度分层 forest；leave-one-task/source-out；每层报告 n、base rate、AUPRC 与 Δ。
- Visualization: OOF score onset event study；单一 held-out fold 内由 train-fold embedding 拟合 PCA 后投影 trajectory；whitened innovation；relation/channel occlusion 后重新编码的 Δscore。
- Constraint: 不直接拼接不同 fold 的 latent 坐标；raw attention heatmap 只能作输入诊断，不作主证据。
- Priority: MUST-RUN。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 数据与无泄漏 sanity | toy overfit、mask audit、prefix invariance、label access audit | 被遮边重建 loss 可下降且截断表示一致 | <1 GPU-hour | teacher-forcing indexing 偏一位 |
| M1 | 强无监督 baseline | 手工 features、edge marginals、untrained/no-message | 复现并固定 all-token OOF evaluator | CPU/GPU 数小时 | 域校准造成虚高 |
| M2 | 主方法 | 32D、2-layer、K=4、3 seeds | ΔAUPRC CI 与绝对门槛同时通过 | 约 15 个 fold-runs | 单图长度导致显存波动 |
| M3 | 决定性消融 | rewire、relation/channel collapse、K/time/q0 | 只保留有证据的机制主张 | 约 6 variants × folds | run 数较多 |
| M4 | 图与稳健性 | 三张 learned-embedding 主图、分层/LOO | 结果不由单一 domain 驱动 | CPU 为主 | 跨 fold latent 旋转 |

## Compute and Data Budget

- Data preparation: 现有 canonical CSR 已足够；不再生成 dense adjacency 或重复 graph archive。
- Main bottleneck: 每折 encoder 训练；首轮只跑 seed 0，过 M0/M1 后再扩三 seed。
- GPU memory: 单图/小批图在 GPU；数据按样本流式加载，避免把全量 canonical 数据放入内存。

## Risks and Mitigations

- 混合正确/错误训练会吸收异常：冻结 encoder 后用 q0 抗污染，并报告 contamination sensitivity。
- 重建难度不等于幻觉：主分数来自条件 embedding density；重建 energy 只诊断，不任意相加。
- task/source shortcut：按 source 分组 OOF、条件校准、分层 FPR 与 leave-one-domain-out。
- top-k 改变自然度数：主方法默认直接从 canonical threshold support 读边；top-k 仅作压缩消融。
- attention floor censoring：distribution decoder 明确建 `OTHER` bucket，不把缺失边视为零。

## Final Checklist

- [x] Main paper tables are covered
- [x] Novelty is isolated
- [x] Simplicity is defended
- [x] 不强行加入 frontier primitive
- [x] Nice-to-have runs are separated from must-run runs
