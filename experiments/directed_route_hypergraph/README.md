# Directed Route Hypergraph + Ordered Endpoint Layout

当前实验研究一个受限问题：稀疏 attention 中按 Transformer 层序组合的真实 source endpoint，能否为 token-level hallucination detection 提供比局部 row 或三态 provenance 更有用的无标签表征。

方法由两部分组成：

```text
deterministic teachers
  local attention rows
  + ordered P/R/U provenance
  + ordered token-endpoint layout
                  |
                  v
corrupted graph -> neural directed-hypergraph encoder -> 64D token embedding
                                                        -> PCA-kNN score
```

最后的 encoder 和 detector 都不读取 hallucination label。这里的“无标签”不等于“没有神经网络”：`SourceToHyperedge`、slot attention、head pooling、GRU 和三个 decoder 都通过反向传播训练。

## 与 Information Flow Reveals When to Trust Language Models 的关系

论文先在冻结 LLM 上计算每层 value-aware contribution：

\[
a_{j\to i}^{(l)}
=
\mathbf 1[j=i]x_i
+\sum_h W_O^{l,h}A_{ij}^{l,h}W_V^{l,h}x_j,
\]

再用 ALTI 风格距离归一化得到 \(C^{(l)}\)，并按真实层序组合：

\[
C^{\mathrm{total}}=C^{(L)}\cdots C^{(1)}.
\]

它的信息流抽取本身不训练新 encoder，但完整 trust detector 并不只是“简单算几个特征”：论文还使用 Qwen3 reranker、SHAP relevance，构造 9 个 RBO、2 个 concentration 和 1 个 relevance 特征，最后用 correctness labels 训练 XGBoost。

本实现只迁移层序路径代数，不复现论文的 functional contribution。现有 cache 没有 hidden state、\(W_V/W_O\)、真实 residual message 或 prompt-query rows，所以代码和论文中统一称为 **layer-ordered attention transport endpoint layout**，而不是 Information Flow、causal contribution 或真实 grounding。

## 1. Exact typed graph

一个样本是一张独立 `TokenGraph`：

```text
node: token
edge: (source, response target, layer, head, retained attention weight)
row mass: retained + diagonal + unresolved = 1
```

构图前不平均 layer/head，不把 cache 没保存的边当作零。prompt-query rows 不可用，因此 prompt token 在跨层 transport 中保持 identity endpoint。

## 2. P/R/U ordered provenance

每个 token 的来源是：

```text
P  path starts at a prompt token
R  path remains response-origin / response-closed
U  path has entered unresolved cached mass
```

对 response row \((t,l,h)\)：

\[
\rho_{t,h}^{l}
=
\sum_{s<t}a_{t,s}^{l,h}\pi_s^{l-1}
+d_{t,h}^{l}\pi_t^{l-1}
+u_{t,h}^{l}e_U.
\]

head-uniform attention state 与显式 residual proxy \(\alpha\) 合并：

\[
\pi_t^l=
\frac{\alpha\pi_t^{l-1}+H^{-1}\sum_h\rho_{t,h}^{l}}
{\alpha+1}.
\]

P 只表示路径能追溯到某个 prompt token，不表示该 token 与问题相关或提供了事实证据。

## 3. Ordered endpoint layout

P/R/U 是完整 endpoint layout 的粗粒化。新目标保留每个 token endpoint，并增加一个 unresolved sink \(\bot\)。设 \(Q_t^0\) 为 response token 自身的 one-hot endpoint。每层先用该层全部 typed edges 组成 head-uniform attention transition，再加入 residual proxy：

\[
Q_t^l
=
\frac{
\alpha Q_t^{l-1}
+H^{-1}\sum_h
\left(
\sum_{s<t}a_{t,s}^{l,h}Q_s^{l-1}
+d_{t,h}^lQ_t^{l-1}
+u_{t,h}^le_{\bot}
\right)
}{\alpha+1}.
\]

最终 \(Q_t^L\in\Delta^{N}\) 等于 retained-attention proxy 中所有合法 layer-ordered paths 的权重乘积之和。prompt endpoint 使用隐式 one-hot，response-to-response transport 使用 sparse-dense multiplication，不构造 edge-by-endpoint 三维张量。

训练默认每张图均匀抽取最多 `layout_rows_per_graph=32` 个最终 response rows。实现先从这些 rows 反向追踪每层真实 response relay，构造精确依赖闭包，再只对闭包执行前向 teacher rollout。任意选中 row 的结果与完整 \(R\times(N+1)\) layout 中对应行严格相同；这不是截断近似、top-k 近似或把未计算 endpoint 当作零。

如果随机子集的精确闭包仍超过显式预算，代码按同一随机优先级逐次减半，直至可计算；极端长样本使用首个 response row 作为精确 fallback。只有连该单行都违反用户设置的预算时，该图才跳过 layout 项，local row 与 P/R/U loss 仍继续训练，因此不会因为单个长样本中断整次运行。checkpoint 会保存 `layout_rows_per_graph`，历史中的 eligible-row coverage 可用于审计实际监督覆盖率。

`layout_rows_per_batch` 只切分 neural pointer decoder；`layout_max_elements` 同时限制选中 target/logit 矩阵与 teacher 闭包的峰值 dense state；`layout_max_work_elements` 限制精确闭包的 sparse relay 工作量。完整 layout API 仍保留用于小图测试和 deterministic controls，但默认训练不再强制物化所有 response rows。

unresolved 是吸收式记账：路径一旦进入 cache 未解析质量，就不再假装能恢复到某个已知 endpoint。它不是无效信息或 hallucination 类别。

## 4. 降低 sink/self 捷径的平衡目标

深层 rollout 容易被 unresolved sink 或 residual self endpoint 支配。代码没有把所有列直接塞进一个 categorical CE，而是把 layout loss 分解为：

1. resolved 与 unresolved 的二项质量；
2. resolved 内 self 与 non-self 的二项质量；
3. non-self 条件下的 exact endpoint distribution。

\[
\mathcal L_{layout}
=
\mathcal L_{sink}
+\mathcal L_{self\mid resolved}
+\mathcal L_{endpoint\mid nonself}.
\]

第三项按 token row 平衡，并除以可选 non-self endpoints 数量的对数，减弱回答长度造成的 CE 尺度增长。它不会因为 \(\bot\) 或 self mass 很大而直接消失。质量小于 `layout_min_mass` 的条件项跳过，但其上层二项质量仍被训练。这种分解降低单一 sink/self 分类捷径，不保证排除 position/length shortcut，因此三项 loss 和 eligible-row coverage 都单独记录。

训练时 student 只看 incidence/head 被遮蔽并守恒转入 unresolved 的图；local row、P/R/U trajectory 和 endpoint layout target 都来自干净图：

\[
\mathcal L
=
\mathcal L_{row}
+\lambda_f\mathcal L_{flow}
+\lambda_q\mathcal L_{layout}
+\lambda_v\mathcal L_{variance}.
\]

local row 保留 layer/head，P/R/U 监督每层 trajectory，endpoint layout 监督 retained-attention proxy 内最终全路径的 source identity。三者包含嵌套信息，因此必须通过 `layout_weight=0`、`flow_weight=0` 等消融证明 endpoint 目标有独立价值。权重为零时对应 teacher 和 decoder loss 会真正 bypass，不计算 rollout。

## 5. Neural directed hypergraph student

每个 `(target, layer, head)` 是显式有向超边：

```text
source token -> attention-row hyperedge -> target token
```

模型保留四个 16D route-conditioned slots：P1/P2/R1/R2。response source 若已继承 prompt provenance，仍向 P slots 输送质量；它不会被直接 source role 错算为 response closure。slot 名称不等于数学纯 factorization，必须靠 route gating/head/endpoint controls 验证。

最终四槽展平为 64D token embedding。PCA-whitened kNN 仍是当前 label-free detector，因此新实现只是表征目标优化，还没有真实 QA 结果证明检测更强。

## 6. 为什么没有重新启用 P-Cut

历史 P-Cut 使用相同 token 的 full/no-prompt/no-response 三视图和 closure score。全量 QA 的冻结方向结果为 AUROC `0.4209`、AUPRC `0.0734`，低于位置基线；记录明确禁止通过翻转方向或换名重新包装。

当前改动不做 route cut，不计算 closure，也不把 prompt mass 预设为正确性方向。ordered endpoint layout 只是干净图的连续全路径重构目标；是否改善下游检测必须重新实验，不能从设计推出。

## 7. 运行

```bash
bash experiments/directed_route_hypergraph/run_qa.sh
```

小规模 smoke test：

```bash
EPOCHS=1 TRAIN_LIMIT=32 TEST_LIMIT=16 \
OUT=experiments/directed_route_hypergraph/outputs/smoke \
bash experiments/directed_route_hypergraph/run_qa.sh
```

默认关键参数：

```text
INCIDENCE_DROPOUT=0.15
HEAD_DROPOUT=0.05
FLOW_WEIGHT=0.5
LAYOUT_WEIGHT=0.25
LAYOUT_ROWS_PER_GRAPH=32
LAYOUT_ROWS_PER_BATCH=64
LAYOUT_MIN_MASS=0.0001
LAYOUT_MAX_ELEMENTS=8000000
LAYOUT_MAX_WORK_ELEMENTS=250000000
LAYOUT_ORDER=ordered
RESIDUAL_WEIGHT=1.0
```

最小目标消融：

```text
local only        FLOW_WEIGHT=0 LAYOUT_WEIGHT=0
local + P/R/U     FLOW_WEIGHT=0.5 LAYOUT_WEIGHT=0
local + endpoint  FLOW_WEIGHT=0 LAYOUT_WEIGHT=0.25
all objectives    FLOW_WEIGHT=0.5 LAYOUT_WEIGHT=0.25
reverse endpoint target   LAYOUT_ORDER=reverse
```

默认输出目录包含 `LAYOUT_ROWS_PER_GRAPH`、`FLOW_WEIGHT`、`LAYOUT_WEIGHT`、`RESIDUAL_WEIGHT`、order、variant 和 seed，目标消融不会静默覆盖彼此；其他配置变体可用 `RUN_NAME` 或 `OUT` 显式隔离。

## 8. 解释边界与必须实验

当前实现是 post-hoc same-token routing representation，不是论文的 trust-before-next-token estimator。首个 response token 所需的 prompt 最后一行没有缓存，因此代码不伪造 next-token 对齐。

正式机制结论前至少需要：

- ordered layout 对比 reverse endpoint target、last-layer 和 layer-mean controls；
- `LAYOUT_ORDER=reverse` 只反转 endpoint teacher；encoder 和 P/R/U teacher 保持正序；
- real endpoint 对比 role/lag/mass-matched rewire 与 weight shuffle；
- neural embedding 对比直接 endpoint layout detector；
- `local / flow / layout` 完整目标消融；
- position-only、self+unresolved、response length 和 retained coverage controls；
- 小规模重新采集 value-aware contribution，比较 endpoint JSD/RBO 与 route-token 删除后的 logit drop；
- QA、Summary、Data2txt 分任务、多个 seed、source bootstrap。

如果正确层序不优于 reverse，real endpoint 不优于 rewire，layout 目标只预测 position/self/unresolved，或 attention-only layout 与 value-aware contribution 明显不一致，就不能把增益归因于跨层信息流。此时应停止增加 encoder 容量，转而补采 hidden/OV/residual caches。
