# Attention Route-Delta Baseline

这个目录是 `Information Flow Reveals When to Trust Language Models` 的 attention-only 对照，不是论文算法的复现。

## 原论文到底有没有神经网络

论文没有训练新的 information-flow encoder。它重放预训练语言模型，使用 hidden states、(W_V/W_O)、attention 和 residual 解析计算每层 source contribution，再按 layer 顺序相乘 contribution matrices。

完整 trust estimator 仍然包含：

```text
被分析的预训练 LLM
neural reranker + SHAP relevance layout
correctness-label XGBoost calibrator
```

因此其 attribution core 是无新参数的白盒算子，完整系统不是无神经网络，也不是无监督方法。

当前缓存没有 OV contribution、hidden/residual states 或 prompt-query rows，所以本目录只能研究 routing structure，不能把输出称为 functional information flow。

## 当前对照

GCN 64D embedding 只作为独立 base channel 保存。它已经聚合全部 layers，不能再作为 progressive-flow 的 layer-zero state，否则会造成 depth-mechanism leakage。

route branch 从固定、图无关的 role/position source basis (B^0) 开始。对 layer (l)、head (h)、response target (t)：

\[
\Delta_{t,h}^{l}
=
\sum_{s<t}a_{t,s}^{l,h}
R_{l,h}(B_s^l-B_t^l).
\]

```text
mean mode    R is identity, then uniform head mean
sketch mode  fixed signed permutation per (layer, head), then 1/sqrt(H) sum
```

再更新

\[
B_t^{l+1}=B_t^l+\operatorname{merge}_h(\Delta_{t,h}^{l}).
\]

这个 delta 形式有三个必要性质：

- 没有 off-diagonal endpoint 时严格 identity；
- diagonal 在 delta 写法中产生 zero increment；unresolved 也被保守地设为 zero increment，这是无法定位 censored endpoints 时的 proxy，不代表未知质量真实来自 self；
- prompt-query rows没有缓存，因此 prompt basis 始终固定。

处理 25%、50%、75%、100% depth 后，输出：

\[
Z_t=[Z_t^{GCN};B_t^{L/4};B_t^{L/2};B_t^{3L/4};B_t^L].
\]

对于 64D base 是 320D。artifact 中 `base_embedding` 与 `trajectory` 分开保存，便于确认增量来自 route branch。

## 这项对照能回答什么

统一 evaluator 比较：

```text
sketch route delta   保留 typed head identity 的固定随机特征
mean-head delta      不编码 head identity
frozen GCN           不做 progressive route transport
```

它只能检验 ordered attention topology 是否在 frozen GCN 之外增加可读信号。即使 sketch 胜出，也必须再做 multiple seeds、reverse layer、self-only、endpoint-rewire 和 weight-shuffle controls，才能归因于 head/path structure。

## 运行

前提是 GCN calibration/test graphs 已存在：

```bash
bash experiments/information_flow/run_qa.sh
```

输出包括：

```text
base_embedding  [response, 64]
trajectory      [response, checkpoints, 64]
node_embedding  [response, 320]
```

## 停止条件

若 ordered route 不优于 reverse/self-only，真实 endpoints 不优于 rewiring，或结果随 sketch seed 大幅波动，就停止扩展 attention-only transport。下一步应补采每层 hidden state、per-head OV source contribution、attention/FFN 前后 residual，并明确 next-token 对齐。
