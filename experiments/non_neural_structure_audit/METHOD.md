# Non-neural Structure Audit 方法说明

## 1. 方法目标与边界

该审计不训练 GNN，也不使用幻觉标签构造分数。它把缓存中的稀疏 attention 转成一个可解释、质量守恒的 routing proxy，检查回答 token 对 prompt 与先前回答的依赖结构、这种依赖沿 Transformer 层传播后的来源，以及这些结构是否在标签后置评估中与下一 token 的幻觉有关。

当前方法描述的是 **prompt-connected 与 response-base attention lineage**，不是 evidence grounding，也不是真实模型计算的 causal ancestry。

一键入口默认只选择 RAGTruth `task_type=QA`；`TASK_TYPE=all` 才会纳入其他任务。`limit` 在 task 过滤之后应用。

## 2. 从稀疏 attention 到 routing

入口在 [`experiment.py`](experiment.py) 的 `_fit_sample()` / `_score_sample()`：

```text
sample + loaded attention
  -> collect_routing_edges() + copy response token IDs
  -> release_attention()
  -> build_routing_state()
  -> LineageOperator.run()
  -> build_layer_features()
  -> 无标签标准化并冻结分数
  -> evaluation.py 才读取标签
```

[`../attention_phenomenology/routing.py`](../attention_phenomenology/routing.py) 的 `collect_routing_edges()` 读取正式缓存中的响应 query 行：保留超过 attention floor 的严格因果非对角边 `(layer, head, query, source, weight)`，并单独读取精确对角 attention。`build_routing_state()` 将每个 `[token, layer, head]` 行分成四种质量：

- `prompt_mass`：source 位于 prompt；
- `response_mass`：source 是更早的 response token；
- `self_mass`：精确对角；
- `unresolved_mass`：因稀疏截断而未观测到的剩余 attention 质量。

若观测质量因数值误差超过 1，代码先按行缩放；之后每行满足

\[
p^{P}_{t\ell h}+p^{R}_{t\ell h}+p^{S}_{t\ell h}+p^{U}_{t\ell h}=1.
\]

这里守恒的是归一化 attention probability，不是 hidden-state contribution。

## 3. 六类守恒 lineage

核心传播位于 [`lineage.py`](lineage.py)。每个 response token 在传播前被初始化为 `response_base=1`，并维护六维状态

\[
z=(P_d,P_r,R_b,R_1,R_{2+},U),
\]

分别表示：

1. `prompt_direct`：当前层直接连到 prompt；
2. `prompt_relay`：经至少一个 response token 转接、最终仍连到 prompt；
3. `response_base`：当前 token 自身的基础路径；
4. `response_relay_one_hop`：经一个 response endpoint 继承的 response-base；
5. `response_relay_multihop`：经至少两个 response endpoints 继承的 response-base；
6. `unresolved`：来自被截断 attention 的未知质量。

当消息经过一个 response endpoint 时，`_through_response()` 使用固定映射

\[
P_d\!\to P_r,\quad P_r\!\to P_r,\quad
R_b\!\to R_1,\quad R_1\!\to R_{2+},\quad
R_{2+}\!\to R_{2+},\quad U\!\to U.
\]

令 (T(\cdot)) 表示该映射，则一层中每个 head 的更新为

\[
\tilde z^{\ell,h}_t=
p^P_{t\ell h}e_{P_d}
+p^S_{t\ell h}z^{\ell-1}_t
+p^U_{t\ell h}e_U
+\sum_{j<t}a_{t\leftarrow j}^{\ell h}T(z^{\ell-1}_j),
\]

随后等权平均 heads：

\[
z^\ell_t=\frac{1}{H}\sum_h\tilde z^{\ell,h}_t.
\]

由于 routing 行和为 1，且 (T) 只重新分配类别而不丢质量，所以每个 token、每一步均有

\[
\mathbf 1^\top z^\ell_t=1.
\]

## 4. 可解释特征与无标签参考

[`features.py`](features.py) 生成 `[response token, layer, feature]`：

- 直接 routing：prompt/history/self/unresolved mass、response takeover；
- head 结构：prompt/history head 标准差与角色分歧；
- response endpoint 结构：有效来源数、top-1 share、近期来源占比、平均 lag；
- lineage：prompt-connected total/relay、response-base local/inherited/multihop、lineage unresolved，以及 response-to-prompt log ratio。
- 跨层观测变化：相邻 layer step 的 prompt/history/diagonal head-mean absolute change、history-minus-prompt origin gap，以及 interaction-minus-diagonal gap。

最后一组是从已观测 routing 直接计算的 transition magnitude，不是训练得到的 hidden state，也不是“预测下一层后得到的重建残差”。第一个 layer step 固定为 0；其后对相邻 step 的每个 head 取绝对差，再对 heads 求均值。它只回答“该 routing 量随层顺序变化多少”，不能回答某条边是否被模型恢复、某层是否充分，或该变化是否是因果机制。

[`reference.py`](reference.py) 只用训练 split，按 task 与 response 因果位置桶拟合 robust center/scale；这只是无标签标准化，不是模型训练。`relation_scores()` 将预先指定的单个坐标定向为“越高越风险”，再对层取均值。fit 与 score 阶段不打开标签，冻结产物及 `labels_read=false` 由 [`experiment.py`](experiment.py) 写入。

## 5. 两个结构对照

### Response endpoint null

[`nulls.py`](nulls.py) 只交换 response-history endpoints。交换限制在相同 `(layer, head, coarse log2-lag bin)` 内，并保持边权所在 target row 不变，同时拒绝非因果边与重复边。因此它保持 row mass、response role、粗粒度 lag 分层和非加权 source 次数，但会改变精确 lag、source 与权重的配对及继承到的 lineage。

该 null 是 A2 的 pilot，不是计划中的正式 degree/weighted-strength preserving null。它不保持 weighted source strength，也没有均匀采样或 mixing 证明。评估只把它用于 manifest 中列出的 lineage relations；`changed_fraction` 的分母是全部 retained response-history edges，不包含 prompt edges。若实际可合法交换的比例低于计划阈值 0.7，pilot 质量标记为 null 无效；即使覆盖率达标，在更强 null 完成前也不能授权 exact graph。

### Final-state layer shuffle

[`experiment.py`](experiment.py) 对完整 layer operator 做多次随机排列。每次仅取全部乱序层执行后的最终 lineage state，并重算跨层 transition features；这些 layer-order-sensitive 坐标替换真实样本的 final-layer 对应坐标，直接 routing、head 和 endpoint 特征保持真实值。这样避免把“实际第 20 层”拿去和“正常第 0 层”标准化，检验的是该有限传播算子及观测 transition 对层顺序的敏感性，而不是对原 LLM 做了真实 layer intervention。

## 6. 标签后置评估

[`evaluation.py`](evaluation.py) 在确认分数已无标签冻结后，先对所有选中 NPZ 的 hash、schema、relation 列、形状和 token IDs 做 label-free 预校验，随后才调用 `prepare_evaluation_labels()`。缓存 trace 是 `post_token_query_at_same_position`；评估使用

\[
\text{score at query }t \longleftrightarrow \text{label of response token }t+1,
\]

即 `score[:-1]` 对齐 `labels[1:]`。首个 response token 没有对应的前置 response query，最后一个 query 也没有样本内 next-token 标签。可选 tokenizer 当前只用单 token decode 后是否含字母、数字或中日韩文字来近似 content mask；它还不是 entity/number/boundary 分层。统计按 canonical `source_id` 分组，并输出 relation、时间趋势、null、bootstrap/permutation 与 decision gates。

这里的数组对齐单测不等于完整 A0。当前 A0a（artifact/source/null invariant binding）已实现；A0b（raw RAGTruth span、tokenizer offset、BOS/EOS/assistant prefix 与真实 cache query 语义的 gold trace）和 A0c（完整 pipeline 的 sample/span label-permutation sanity）未实现。总 A0 是三者的 AND，因此当前正式 A1–A10 gate 全部 `BLOCKED_BY_A0`。

## 7. 不能由当前代码支持的声明

- **不能称为 evidence grounding**：所有 prompt token 仍是同一类型，尚未区分 evidence、question、instruction 和 system；
- **不能称为模型计算 ancestry**：只有 raw attention，没有 value/output projection、residual stream、FFN 与向量抵消；head 也是等权平均；
- **不能称为生成因果机制**：trace 来自 teacher-forced 缓存，endpoint null 和 layer shuffle 都只作用于审计算子；
- **不能由缓存证明干预效果**：要验证 detached lineage 是否导致事实错误，仍需在 base LLM 上做等质量、匹配 layer/head/lag 的真实 attention 或 KV intervention。

参数集中在 [`config.py`](config.py)；content-token 筛选位于 [`token_classes.py`](token_classes.py)。

## 8. 冻结协议与精确 null 统计

[`protocol.py`](protocol.py) 把 label boundary 变成可验证 artifact，而不是依赖命令行中的一个 `scope` 字符串：

1. reference 保存 train dataset manifest hash、sample IDs 与 source IDs；
2. score 保存 test dataset manifest hash、reference hash、每个 sample NPZ hash、完整 sample/source scope 与 null invariants；
3. `plan` 在读标签前按完整 source group 冻结 discovery/confirmation IDs；
4. `freeze-confirmation` 冻结 discovery report、audit 本包和共享 runtime source digest、tokenizer 文件与 evaluation config；
5. confirmation 在打开目标标签前重新验证摘要、sample→source 映射、全集覆盖和 source-group 不相交性。

`freeze-confirmation` 还要求 discovery decision 中总 A0 已为 `PASS`；当前 A0b/A0c 缺失，所以命令会按停止规则拒绝创建 confirmation plan。

该 source digest 覆盖当前 audit、attention routing、dataset/cache 与共享 protocol 代码，但本地文件摘要不能证明操作者没有重新生成 plan；论文级运行仍须把 Git commit/tree、plan 和首次 confirmation 命令写入只追加记录。

endpoint/layer null 的主 p 值保持原始 pooled-AUPRC 定义。若真实 AUPRC 为 (m(x,y))，第 (k) 个 null ensemble 为 (x^{(k)})，则

\[
p=\frac{1+\sum_{k=1}^{K}\mathbf 1[m(x^{(k)},y)\ge m(x,y)]}{K+1}.
\]

[`bounded_ensemble.py`](bounded_ensemble.py) 用临时磁盘矩阵逐 replicate 精确计算 pooled AUPRC；没有用可分解的均值差近似替换 AUPRC。由于 endpoint 生成器没有均匀性或交换性证明，这只是 Monte Carlo constrained-null p-value，不是 exact permutation test。source-group bootstrap 只用于 real-vs-null-mean 的 effect CI，两种统计的职责分开。

[`bounded_samples.py`](bounded_samples.py) 同时把跨样本的 compact real/final/null-mean/layer-mean 矩阵放到临时 memmap。这样原始 attention、replicate ensemble 和紧凑样本矩阵都不会在 Python heap 中随样本数累计；各统计仍可通过 `FrozenSample` 的切片接口复用相同数据。

## 9. 当前 gate 为什么保守

当前代码可以产生 A1 association、A2 response-endpoint pilot、A4 layer-permutation、A7 temporal 和 A9 discovery-CV 描述量，但总 A0 尚未完成，所以正式 A1–A10 首先统一 `BLOCKED_BY_A0`。这些数值只用于完善审计设计，不能作为 confirmation 结论。即使未来 A0 通过，以下缺口仍会阻止模块授权：

- A1 尚缺 position/token-class/known-mass matched nuisance control 与预注册 `d_z`；
- A2 的 pilot 只改变 response-history endpoints，不保持 weighted strength 或证明 mixing，不能授权 exact-token graph；最新两条 QA engineering smoke 的 coverage 约为 0.139、0.162，低于计划阈值 0.70，但这不是总体估计；
- A3 尚缺 head-mean conditional increment；
- A4 尚缺 layer-mean/final-layer 等强无序基线；
- A5/A6 尚缺深度消融与 fixed-endpoint weight null；
- A8 尚缺 discovery-frozen change-point threshold；
- A9 当前只在 discovery 做 exploratory grouped CV，confirmation 不会重新拟合；
- A10 必须回到 base LLM 做 matched intervention。

因此当前 `decision_table.csv` 的 A0 是 `INCONCLUSIVE_A0_CONTROLS_MISSING`，正式 scope 的其余行是 `BLOCKED_BY_A0`。若忽略 A0 单看当前 A2 pilot，其 null 质量是 `INCONCLUSIVE_NULL_INVALID`；这不是实际顶层 decision。代码不会用 fallback 生成一个看似完整的结论。
