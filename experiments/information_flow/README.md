# Information Flow Sketch

这个目录验证《Information Flow Reveals When to Trust Language Models》给当前项目带来的核心启发：不要只问一条 attention 边有多大，而要追踪一个 token 的状态如何沿 Transformer depth 被逐层运输。

## 先说明原论文是否“没有神经网络”

原论文没有训练一个新的神经 information-flow encoder。它在已有语言模型内部确定性地计算：

```text
attention weight
× value vector
× output projection
× residual-aware contribution
```

再逐层相乘得到 contribution layout。可是完整方法仍然依赖三个神经模型或监督环节：

```text
被分析的预训练语言模型
外部 neural reranker + SHAP relevance layout
使用正确性标签训练的 XGBoost calibrator
```

所以它不是“几个简单统计量组成的无监督方法”，而是确定性白盒归因加外部相关性估计和监督校准。

## 当前数据为什么不能直接复现论文

当前 RAGTruth cache 只有稀疏 attention weights、diagonal 和 unresolved mass，没有：

```text
per-head value vectors
W_O 后的 source contribution
attention / MLP 前后的 residual states
```

因此不能诚实地把 raw attention 边称为论文中的 functional contribution edge。本目录实现的是一个 attention-only 可证伪实验，而不是冒充原论文复现。

## 方法：GCN state + typed attention transport

已有一阶 GCN 节点表征是目前最可靠的无标签图基线：

```text
PCA-kNN       AUROC 0.6982  AUPRC 0.1617
linear probe  AUROC 0.7865  AUPRC 0.2999
```

GCN embedding 被当作每个 token 的初始内容状态。随后使用原始逐层、逐 head attention 图运输这些状态。

对 layer `l`、head `h` 和 response target `t`：

\[
M_{t,l,h}
=
\sum_s A^{l,h}_{t,s}Z_s^l
+
(d_{t,l,h}+u_{t,l,h})Z_t^l.
\]

`d` 是精确 diagonal，`u` 是 cache 中未解析的质量。未知端点不会被伪造成零边；它只保留在当前 token 上。每个 row 再除以已知总质量，形成 row-stochastic transport。

## 为什么不直接平均 heads

`mean` control 直接对 32 个 head states 取平均。

主方法 `sketch` 给每个 `(layer, head)` 一个固定的 sign-permutation 正交映射，再求和：

\[
Z^{l+1}_t
=
\frac1{\sqrt H}
\sum_h R_{l,h}M_{t,l,h}.
\]

这种 CountSketch / random-feature 形式不训练新参数，但不同 heads 不会在相同坐标中直接相消。相同 seed 在所有样本上使用同一组映射，因此节点维度可以跨样本比较。

prompt query rows没有缓存，所以 prompt token 只经过相同的 head sketch 做 residual transport；response token 使用真实 incoming attention rows。处理完 25%、50%、75% 和 100% depth 后保存快照：

\[
Z_t=
[Z_t^{GCN};Z_t^{L/4};Z_t^{L/2};Z_t^{3L/4};Z_t^L].
\]

Llama-3.1-8B 的最终节点表征是 `5 × 64 = 320` 维。这里没有定义 entropy、lookback、closure 等幻觉特征；表示完全由 GCN 状态和层间图算子产生。

## 这轮实验回答什么

统一评估同时比较：

```text
sketch flow     保留 layer/head identity 的运输轨迹
mean-head flow  构图前不丢 layer，但在每层平均 heads
gcn             不做逐层 typed transport 的已有基线
```

所有方法使用同一组 node-only readers：

```text
PCA-kNN
Isolation Forest
LOF
Autoencoder
Deep SVDD
source-disjoint linear / MLP readability probes
```

最终 detector 不读取边，也不运行第二个 GNN。

## 运行 QA

前提是 GCN calibration/test graph artifacts 已存在：

```text
experiments/dbgnn_reference/outputs/qa_compare/gcn/
├── calibration/index.npz
├── calibration/graphs/*.pt
├── test/index.npz
└── test/graphs/*.pt
```

运行：

```bash
bash experiments/information_flow/run_qa.sh
```

输出：

```text
experiments/information_flow/outputs/qa/
├── sketch/calibration/index.npz
├── sketch/test/index.npz
├── mean/calibration/index.npz
├── mean/test/index.npz
└── evaluation/report.json
```

每条样本还会保存：

```text
base_embedding  [response, 64]
trajectory      [response, 4, 64]
node_embedding  [response, 320]
```

## 判定标准

只有出现下面任一结果，attention-only layerwise flow 才值得继续：

1. `sketch flow` 的无监督 AUROC/AUPRC 超过 GCN；
2. `sketch flow` 的监督可读性上限超过 GCN；
3. `sketch flow` 稳定超过 `mean-head flow`，说明 head identity 有增量。

如果三条都不成立，不能继续靠增加聚合模块来修饰结果。下一步应改变数据采集，保存 value/output contribution 和 residual states，才能真正复现论文的 functional information flow。
