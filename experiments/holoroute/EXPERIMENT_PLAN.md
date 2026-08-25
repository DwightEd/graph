# P-Cut 实验计划

## 第一轮必须报告

1. `closure` raw residual 和条件校准后的 final score；
2. absolute / relative position baseline 与 Spearman；
3. prompt necessity 与 response-closed necessity 的独立分布；
4. provenance interval width、cut fallback 和 unresolved mass 覆盖；
5. source-group bootstrap CI；
6. 每样本导出的 token embeddings 与 edge P/R/Q partition。

## 必做对照

```text
direct prompt mass / Lookback
RR spectral residual
旧 HoloRoute baseline result
P-Cut closure
```

下一步结构对照：

```text
equal-mass random cut
matched exact-endpoint rewire
no provenance propagation（只按 prompt/response role 切）
head-identity shuffle
layer-order shuffle
```

## 接受标准

- P-Cut 必须超过 absolute-position baseline；
- 必须超过 direct prompt/response ratio；
- real endpoints 必须优于 matched rewire；
- 分数不能主要由 removed mass、position、length 或 sparse fallback 解释；
- QA、Summary、Data2txt 和多个模型上的方向必须可复现。

若这些条件不满足，就停止 response-closure 假设，而不是继续增加更多分数或神经模块。
