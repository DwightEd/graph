# 从结构审计到无监督图模型：研究路线与停止规则

## 0. 本文目的

本文件用于防止项目把“结构审计”误当成最终方法，也防止在没有证据时继续堆叠 GNN、GRU、超图、随机游走或更多手工指标。

当前仓库中存在两条历史路线：

- `agent/routing-dynamics-audit`（提交 `d332218`）实现过神经式 next-layer graph reconstruction、prompt/response 双路消息、`RoutingTransitionCell` 和多种 reconstruction gains；
- 当前 `agent/graph-structure-audit`（基线提交 `51829a1`）退役了该可运行神经原型，新增 `experiments/non_neural_structure_audit/`，用于在标签后置、结构 null 和严格协议下先确认哪些关系值得建模。

当前非神经审计是 **scientific instrument**，不是论文最终模型。

---

## 1. 当前版本保留了什么，丢掉了什么

### 1.1 保留的研究问题

当前审计仍然研究：

1. prompt 与 response-history routing 是否具有不同统计行为；
2. response token 是否可通过 response endpoints 追溯到 prompt-connected lineage；
3. 一跳与多跳 response-base lineage 是否存在增量；
4. head 间是否出现 prompt/response 角色分裂；
5. layer 顺序是否不可交换；
6. 首个错误前是否出现结构变化，错误后是否存在持续状态；
7. 多种关系是否需要非线性联合。

### 1.2 当前退役的神经实现

以下内容没有进入当前可运行主路径：

- learned graph embeddings；
- `CrossOriginRoutingDynamics`；
- `RoutingTransitionCell` / GRU；
- prompt/response learned transport；
- next-layer edge/support/diagonal reconstruction；
- learned message gain、prompt gain、response gain、closure；
- learned per-layer/per-head prediction maps。

退役原型不等于否定这些方向。它表示当时的任务、候选 topology、统计协议和资源行为还不足以支持这些模块。

### 1.3 当前代码并非只有“几个平面标量”

当前代码仍实现了：

- exact sparse causal attention endpoints；
- `[token, layer, head]` 的 prompt/history/self/unresolved 守恒 routing；
- 六类有限层 lineage 传播；
- response endpoint null；
- layer-order permutation；
- train-only task/position robust reference；
- query `t` 到 token `t+1` 的标签后置对齐；
- source-group bootstrap、circular shift、discovery/confirmation 冻结协议。

但这些结构最后被投影到预注册的可解释 relation coordinates。它们只能用于确认“某种关系是否值得建模”，不能替代可学习的联合图表示。

---

## 2. 对当前非神经审计的评价

### 2.1 有价值的部分

当前版本解决了过去方法中几个反复出现的问题：

- 不再用测试标签筛层、定方向或拟合标准化；
- 不再把稀疏 cache 中未保存的边直接当成真实零；
- 不再因为一个人工 corruption 可被识别，就声称它对应真实幻觉；
- 不再把 token 当作独立样本做显著性推断；
- 不再用一个未经审计的 GNN 结果反推图结构有效；
- 明确规定每个结构 Gate 通过后才能授权相应神经模块。

### 2.2 与目标研究仍有明显距离

当前版本没有充分触及我们最终要研究的对象：

1. **Prompt 不是 evidence。** 问题、指令、系统文本、检索证据和标点目前属于同一类 prompt source。
2. **Lineage 是角色级代理。** 它不保存具体 prompt evidence endpoint，也不能验证语义忠实的 relay。
3. **Heads 最终被压缩。** 当前 relation 多数对 heads 或 layers 求均值，无法学习 head coalition、layer-local fracture 和 recoupling。
4. **没有学习联合图结构。** relation 之间只做探索性线性/二阶交互比较；没有图编码器自动学习 source、head、layer、path 的联合模式。
5. **没有功能性消息传递。** 当前 lineage 使用固定转移规则，不是 learned transport，也没有验证不同来源信息应如何组合。
6. **没有完整的 exact-topology 证据。** 现有 endpoint null 覆盖率低，且没有保持 weighted source strength 或证明 mixing。
7. **没有实际模型计算贡献。** Attention 不包含 value、output projection、residual、MLP，也不能单独识别 MLP 的因果作用。
8. **A0 尚未完成。** Gold token/span alignment 与完整 pipeline label-permutation sanity 未实现；正式 Gate 仍被阻断。

因此，当前代码应被定位为：

> 用于筛选候选结构假设、排除混杂并授权后续模型组件的非神经审计框架。

不能将其写成最终无监督幻觉检测方法。

---

## 3. 当前最值得检验的候选机制

现有 exploratory 结果中，相对较强的候选关系包括：

- `origin_transition_gap`；
- `inherited_response_base`；
- `multihop_response_base`；
- `direct_role`。

而简单的低 prompt-connected lineage、lineage ratio 和 endpoint concentration 没有表现出同样方向。

因此当前不应把机制写成：

> 幻觉只是减少 prompt attention，或只是集中于少数 response endpoints。

更有价值、但仍需证伪的候选机制是：

## Fracture–Accumulation（路由重组—多跳累积）

正常回答也会发生强烈的跨层变化，也会使用 response-history。真正可能异常的是：

1. response-origin routing 相对 prompt routing 出现更强的跨层重组；
2. 同时，response token 自身的基础 lineage 经一跳、两跳及以上 response endpoints 累积；
3. 这种重组没有重新接回 evidence-aware prompt lineage，而是形成 response-local 的持续状态。

该机制包含三个相互独立的竞争假设：

- **H1：动态不对称。** `origin_transition_gap` 在控制总体 volatility 后仍有增量；
- **H2：深度增量。** multi-hop lineage 在 direct role 和 one-hop lineage 之外仍有增量；
- **H3：联合必要性。** fracture 与 accumulation 的交互优于二者的线性加法。

只有 H1–H3 至少得到稳定支持，才值得据此构造最终图模型。

---

## 4. 下一阶段必须补齐的审计

### P0：先完成 A0，不再绕过对齐问题

实现并通过：

- A0b：RAGTruth span、tokenizer offsets、BOS/EOS、assistant prefix、cache query 的 gold trace 对齐；
- A0c：完整 pipeline 的 sample/span-level label permutation sanity；
- 输出 unmatched、boundary、shift 和 leakage 审计。

在 A0 完成前，后续结果均为 exploratory。

### P1：条件化的 origin-transition 审计

当前：

\[
F_{t,l}=\Delta^R_{t,l}-\Delta^P_{t,l}.
\]

必须控制：

- 总 transition magnitude `ΔR + ΔP`；
- diagonal transition；
- known/unresolved mass；
- prompt/response edge count；
- causal position；
- token class；
- task 和 response length。

要回答的是：

> 在总体变化幅度相同的条件下，response-vs-prompt 的不对称是否仍然有信号？

### P2：Lineage depth ablation

冻结四个嵌套表示：

1. direct role；
2. direct + one-hop response-base；
3. direct + one-hop + multi-hop；
4. full lineage。

使用 source-group nested CV 比较：

\[
\Delta\mathrm{AUPRC}_{1},\qquad
\Delta\mathrm{AUPRC}_{2+}.
\]

若 multi-hop 不优于 direct/one-hop，后续不能授权多跳 message passing 或超图随机游走。

### P3：逐层、逐头结构审计

不再只对 layers/heads 求平均。保存并评估：

- fixed early/middle/late bands；
- layer-local `origin_transition_gap`；
- head-specific prompt/history transition；
- head coalition 与 head-mean conditional increment；
- fracture 的 peak layer、持续层数和 recoupling depth。

不能使用 confirmation 标签选择最佳 layer/head。

### P4：Evidence-aware prompt decomposition

至少将 prompt source 拆成：

- evidence/context；
- user question/query；
- instruction/system/template；
- unknown/other。

Lineage 扩展为 evidence-connected、question-connected、response-base 和 unresolved。只有 evidence-aware lineage 有增量时，才允许使用 grounding、evidence ancestry 等表述。

### P5：更有效的 exact-endpoint null

替换低覆盖率 pilot：

- 保持 `(layer, head, target role, lag)`；
- 保持 target row mass；
- 保持 source degree与 weighted strength；
- 保证因果和无重复边；
- 使用 degree/strength-preserving double-edge swap 或有 mixing 诊断的 MCMC；
- 报告 coverage、burn-in、autocorrelation 和 invariant errors。

若 exact endpoint null 不能稳定消除增量，则不能把 exact-token graph 作为核心构图。

### P6：Aggregation-form 审计

在固定 topology 下比较：

- sum / mean；
- max / top-k；
- concentration/second moment；
- role-typed aggregation；
- multiset interaction。

使用 matched-capacity、相同 split、相同训练预算的对照。只有复杂集合形式超过 sum/mean，才授权 Set Transformer、AllSet 或 typed hyperedge aggregator。

### P7：Temporal recoupling audit

正常回答也可能出现 transition spike。需要区分：

- productive transition：变化后 evidence lineage 恢复；
- temporary exploration：短暂失配后 recoupling；
- persistent fracture：失配持续；
- response lock-in：response-base accumulation 持续上升。

Discovery 阶段冻结 change-point threshold、窗口、最短持续时间与 FPR，再在 confirmation 一次评估。

### P8：Fracture–Accumulation 联合形式

只比较预注册的少量形式：

\[
F,\quad C,\quad F+C,\quad [F]_+[C]_+.
\]

其中 `F` 为条件化 origin fracture，`C` 为 multi-hop accumulation。不要把所有关系自动展开成大量任意交互后再在测试集选最佳模型。

---

## 5. 何时恢复神经模型

神经模型不是被永久放弃，而是必须由审计结果授权。

| 审计证据 | 才允许实现的模块 |
|---|---|
| A3：head-resolved 增量 | head-set / head-attention encoder |
| A4：layer order 超越无序基线 | GRU、SSM 或 layer transition operator |
| A5：multi-hop 超越 direct/one-hop | 多跳 message passing / path composition |
| A6：sum/mean 不足 | typed multiset / hyperedge aggregator |
| A7：错误后有 persistent lock-in | gated/hysteretic temporal memory |
| A8：首错前有稳定 change point | online detector |
| A9：interaction 稳定超过 additive | 联合神经图模型 |
| A2：exact endpoint 通过强 null | exact-token topology |

如果对应 Gate 不通过，保持更简单的模型。

---

## 6. 后续神经方法的目标架构（暂不实现为主方法）

若审计通过，建议建立独立目录：

```text
experiments/routing_lineage_model/
├── graph_data.py
├── evidence_roles.py
├── lineage_encoder.py
├── typed_aggregator.py
├── depth_transition.py
├── temporal_state.py
├── objectives.py
├── experiment.py
├── evaluation.py
└── tests/
```

### 6.1 无损输入

每个 exact token pair 保留：

\[
E_{s,t}\in\mathbb R^{L\times H},
\]

同时保留 observed mask，不把 censored channel 当零。

### 6.2 Typed sources

至少区分：

- evidence prompt；
- question/instruction prompt；
- grounded response relay；
- response-base / ambiguous relay；
- unresolved。

### 6.3 Relation-specific transport

不同 source 类型使用不同 transport，而不是全部消息直接求和：

\[
m^r_{s\to t,l}=T_r(E_{s,t,l})h_{s,l}.
\]

### 6.4 由审计决定 aggregation

默认从最简单的 role-typed mean 开始。只有 P6 证明有必要，才升级到 second moment、top-k 或 learned multiset interaction。

### 6.5 由审计决定 depth operator

默认不使用 GRU。只有 A4 通过，才比较：

- ordered MLP；
- GRU；
- state-space model；
- explicit transport recurrence。

### 6.6 无监督目标

优先采用与机制一致的任务：

- evidence-lineage consistency；
- direct-vs-relay path agreement；
- layer-local transport prediction；
- multi-hop depth discrimination；
- recoupling/change-point likelihood。

不再使用“真实图 vs 容易识别的人工假图”二分类，也不默认高 reconstruction error 就是幻觉。

---

## 7. 明确的停止规则

停止或降级某条路线的条件：

1. direct/one-hop/multi-hop 的条件增量不足 `ΔAUPRC = 0.01`；
2. layer-order 结果不能超过强无序基线；
3. endpoint null coverage 或 invariants 不合格；
4. head-resolved 信号不能超过 head mean；
5. temporal effect在 pseudo-onset 中同样出现；
6. interaction 模型不能稳定超过 additive；
7. evidence-aware lineage 不优于粗 prompt role；
8. 跨任务、跨模型方向不一致；
9. 结果只在 test-label 选择的层、头或方向上成立。

这些情况下，不应通过增加网络复杂度“制造”结果。

---

## 8. 当前执行优先级

按以下顺序推进，不并行扩张方法：

1. 完成 A0b/A0c；
2. 实现 P1 条件化 origin-transition；
3. 实现 P2 lineage depth ablation；
4. 输出 P3 layer/head/band maps；
5. 实现 P4 evidence-aware prompt segmentation；
6. 修复 P5 exact-endpoint null；
7. 实现 P7 recoupling/change-point；
8. 运行 P8 少量预注册联合形式；
9. 根据 Gate 决定是否创建 `routing_lineage_model`。

在以上步骤完成前：

- `experiments/non_neural_structure_audit/` 只作为审计框架；
- `experiments/graph_structure_audit/` 只作为旧 reconstruction baseline；
- `d332218` 只作为神经原型参考；
- 不把任何一个目录直接包装成最终论文方法。
