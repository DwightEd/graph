# Routing dynamics 远端更新审查

审查对象是合并历史中的 `d332218`（`origin/agent/routing-dynamics-audit`）。代码先被完整合并以保留来源与 diff，再退役不安全的 runnable prototype；其中科学问题被缩成无训练、可解释的观测量并接入当前 non-neural pipeline。

## 决策

| 原始想法/实现 | 决策 | 当前落点或理由 |
|---|---|---|
| prompt-origin 与 response-origin 分开观察 | 保留 | `prompt_transition_volatility`、`response_transition_volatility` 与 `origin_transition_gap` |
| 跨 layer 的变化而不只看最终层 | 保留 | 相邻 step 的 head-mean absolute transition magnitude |
| layer order 对照 | 保留并扩展 | lineage 与 transition relations 都进入独立 non-identity layer shuffle；不影响 endpoint-null relations |
| head fracture / off-diagonal 与 diagonal 的关系 | 探索性保留 | `offdiagonal_diagonal_transition_gap`；只有描述性含义，尚无 conditional nuisance control |
| source-disjoint、label-late evaluation | 保留 | 继续使用现有 frozen source plan、artifact hash 与标签边界 |
| 神经 next-layer reconstruction / repairability | 退役 | 候选 topology 由所有层的 union edges 构造，会泄漏未来层 edge support；当前数据也不足以把 reconstruction 叫作计算恢复 |
| learned hidden state、message gain、sufficiency 命名 | 退役 | 原型是不用幻觉标签的 next-layer graph reconstruction 训练，但数值仍依赖训练参数与架构先验；它没有证明 hidden state 对应真实图表征、message passing 或机制充分性 |
| 全量 maps/embeddings/round outputs 累积后 concatenate | 退役 | 30 个 synthetic samples 的原始 scoring 控制流在保存前稳定增长约 312 MiB；真实最大样本下训练中间量下界更高 |
| token-IID Mann–Whitney 与 cohort-fitted normalization | 退役 | 忽略 QA 内相关性，且在目标 cohort 重拟合 median/MAD；当前统计按 source group 聚合并只用 train reference |

## 当前新增量的精确定义

对 token `t`、layer step `l`、head `h` 的 prompt routing mass `p`，定义：

```text
transition_prompt[t, 0] = 0
transition_prompt[t, l] = mean_h(abs(p[t, l, h] - p[t, l-1, h]))
```

history 与 diagonal 同理。随后冻结四个 relation：

- `prompt_transition_volatility`：prompt transition；
- `response_transition_volatility`：history transition；
- `origin_transition_gap`：history transition − prompt transition；
- `offdiagonal_diagonal_transition_gap`：`0.5 * (prompt + history) transition − diagonal transition`；它是差值，不是统计交互项。

这些量没有模型参数、没有 hidden=16/32/96、没有重建目标，也不读取 hallucination label。它们与其他 relation 一样使用 train-only task/position robust reference，再在标签边界之后评估；layer shuffle 对真实与乱序轨迹都使用全 layer transition 均值，只用于检查对层序的依赖。

## 仍不完善、不能越界解释的部分

1. prompt 尚未可靠拆成 evidence/question/instruction，所以 origin 只能叫 prompt，不可叫 evidence provenance。
2. 只有 sparse attention，没有 value/output/residual/FFN；transition 与 lineage 都不是实际 hidden-state contribution。
3. A0b gold trace/token alignment 与 A0c 全 pipeline label-permutation sanity 尚缺，正式 A1–A10 仍统一 `BLOCKED_BY_A0`。
4. endpoint null 的已观察合法交换覆盖率偏低，且不保持 weighted source strength；A2 仍是无效/不充分 pilot。
5. transition relations 尚缺 position、token class、known-mass matching 和同强度无序层基线，不能据此授权 GRU/SSM。
6. off-diagonal/diagonal gap 尚缺在 direct role mass 条件下的增量检验，且不是统计交互项，不能据此授权 head-resolved neural aggregation。
7. 当前 layer shuffle 是审计算子的重排，不是 base LLM layer intervention；真实机制结论仍需 A10 matched activation/KV intervention。

因此，值得保留的是“跨层 routing 如何变化、这种变化是否依赖层序、是否在 source-disjoint 标签后置评估中稳定”的审计问题；不值得保留的是目前会泄漏未来 topology、全量驻留并把重建分数过度命名为 message gain/repairability 的具体神经实现。
