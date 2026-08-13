# Attention Graph Hallucination Detection

本项目直接读取已有压缩稀疏 attention，为每条样本构造 token 因果图，并在不训练 GNN、不反向传播的条件下生成每个 response token 的结构表征。

当前唯一主线遵守两个约束：已验证的图统计量必须在最终节点表征中逐列精确恢复；图传播必须保留边质量、连接数量和路径质量，不能通过行归一化将它们消掉。

```text
compressed CSR attention
  -> exact historical graph scalars + exact direct Lookback
  -> train-only position-conditioned median/MAD
  -> raw RP/RR matrices (not row-normalized)
  -> 1/2-hop path mass + conditional ancestors + self-ancestor residuals
  -> deterministic token graph representation
  -> fixed one-sided mechanism score
  -> labels opened only for evaluation and node coloring
```

## 一键运行

```bash
git pull --ff-only origin main && CUDA_VISIBLE_DEVICES=0 bash run_token_representation.sh
```

指定样本图或输出目录：

```bash
SAMPLE_IDS=11445,11289 OUTPUT_DIR=/path/to/output \
CUDA_VISIBLE_DEVICES=0 bash run_token_representation.sh
```

程序分七个阶段显示逐样本进度，不转换、复制或重新抽取 attention。

## 节点初始状态

每个 response token 直接复用 `attention_graph/statistics.py` 的 exact pair-graph 统计：retained mass、prompt/history mass fraction、entropy、top-1 share、concentration、degree/density、history edge fraction/lag、channel coverage、mean edge strength，以及 exact direct Lookback。完整原值直接写入 `.npz`，PCA 不参与检测。

固定主分数只使用五个预注册机制：

```text
prompt_mass_fraction ↓
edge_density ↓
retained_concentration ↑
mean_edge_strength ↑
history_lag ↓
```

训练集只拟合相对位置分箱内的 median/MAD，每维只累计符合预注册方向的正偏移。没有 RR 边时，`history_lag=0` 会被 mask，不会误判为“极度局部”。其余 exact 特征只单独报告，不混入主分数。

## 构图与多跳传播

Prompt→Response 为 RP，历史 Response→当前 Response 为 RR。pair 边权为所有层头 retained attention 的总和除以层头数：

\[
A_{tj}=\frac{1}{LH}\sum_{l,h}a_{lh,tj}\mathbf 1[a_{lh,tj}\ge\tau].
\]

`A_RP` 和 `A_RR` 都不做行归一化。默认两跳：

\[
M^{(k)}=A_{RR}M^{(k-1)},\quad q^{(k)}=A_{RR}^{k}\mathbf 1,
\]

\[
\bar z_t^{(k)}=M_t^{(k)}/q_t^{(k)},\quad
\Delta_t^{(k)}=z_t-\bar z_t^{(k)}.
\]

程序同时保存 raw message、RR path mass、conditional ancestor mean、self/ancestor residual 和可达祖先数。邻居残差按训练集同位置/同 hop 正路径中位质量 (q_0) 使用 (q/(q+q_0)) 连续降权：无路径严格为 0，极弱路径趋近 0，强路径才接近 1。Prompt 证据按

\[
p^{(0)}=A_{RP}\mathbf 1,\qquad p^{(k)}=A_{RR}p^{(k-1)}
\]

沿 RR 路径继承，所以非直接相邻节点也能显式建模，同时绝对路径质量不丢失。

## 对照与指标

四个视图共享同一训练校准和固定评分公式：

- `token_only`：五个 exact 基础机制；
- `token_graph`：基础机制 + 质量门控的 RR innovation + RP path weakness；
- `no_rp`：实际删除 RP 边、重算节点统计，再仅保留 RR 传播；
- `no_rr`：实际删除 RR 边、重算节点统计，再仅保留直接 RP 证据。

`top1_share` 与 concentration 同属集中度家族，所以前者只保留在完整节点向量和逐特征报告中，不重复进入主分数；Lookback 也作为 exact 独立基线报告，不靠 PCA 恢复。组合分数不能被解释为自动继承任一单特征的 0.7。每个 exact 特征分别报告 signed raw AUROC、`separability=max(AUC,1-AUC)`、两类中位数和方向。`separability=0.7` 可能对应 raw AUROC `0.3`，表示反向可分，不能写成“AUROC=0.7”。报告同时给 token 和 response 聚合结果，以确认历史结果的粒度。

RR path deficit 只作为诊断，不进入主分数，因为“RR 总质量越低越异常”尚未被独立 validation 证实。对第 t 个 response token，只有 `t>=k` 时第 k 跳才有因果资格；天然不可能的 hop 不计分，有资格但没有路径则保留为弱路径证据。

这里属于同一 RAGTruth test 上的机制探索。若这五个方向来自此前查看同一 test 标签，则运行时延迟读取标签并不能消除研究者层面的 test-set selection；正式无泄漏结果必须在独立 validation 上冻结方向、特征与 hop 数，再一次性评估新的 held-out test。

## 输出

- `token_representations_label_free.npz`：全部 test token 的 exact 标量、完整图表征、冻结分数和绘图坐标，不含标签；
- `sample_graphs/*.npz`：每条样本的边、路径质量、多跳消息、残差和节点表征，不含标签；
- `token_representation_report.json`：exact feature、四视图、任务/数据源分组和消融结果；
- `population_token_representations.png`：机制坐标、图表征 PCA 和分数分布；
- `sample_*_token_graph.png`：直接图、非相邻路径、冻结机制坐标上的全部 token 图和机制热图。

二维 PCA 只用于画图，改变投影不会改变异常分数。
