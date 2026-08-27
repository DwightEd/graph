# Information Flow 方法审计与当前优化决策

## 结论先行

`Information Flow Reveals When to Trust Language Models` 不能概括成“完全不用神经网络，只算几个简单特征”。准确分层是：

1. information-flow extraction：冻结生成模型上的确定性白盒归因，不训练新 encoder；
2. relevance reference：使用冻结的 neural reranker 和 SHAP；
3. trust calibration：使用 correctness labels 训练 XGBoost。

因此它是 **training-free attribution + neural relevance model + supervised tabular calibrator**。其中全路径 contribution layout 是最值得迁移的机制对象；12 维特征和监督 XGBoost 不符合本项目严格无标签主方法的边界。

论文正式版本为 Rui Xu, Yi Chen, Jiujiu Chen, Sihong Xie, *Information Flow Reveals When to Trust Language Models*, ICML 2026。它曾以 ICLR 2026 submission 形式公开。

- ICML version: <https://openreview.net/forum?id=vd8HzoFZ7v>
- Earlier ICLR submission: <https://openreview.net/forum?id=gJi0Hp7nLI>
- PDF: <https://openreview.net/pdf?id=vd8HzoFZ7v>
- Author code: <https://github.com/rxu0112/RAG-information-flow>
- Base Information Flow Routes: <https://aclanthology.org/2024.emnlp-main.965/>

## 1. 论文实际计算什么

对 layer \(l\)、head \(h\)，attention 为：

\[
A_{ij}^{l,h}
=
\operatorname{softmax}_j
\left(
\frac{\langle W_Q^{l,h}x_i,W_K^{l,h}x_j\rangle}{\sqrt{d_h}}
+M_{ij}
\right).
\]

源 token \(j\) 对目标 token \(i\) 的向量 contribution：

\[
a_{j\to i}^{l}
=
\mathbf1[j=i]x_i
+\sum_hW_O^{l,h}A_{ij}^{l,h}W_V^{l,h}x_j.
\]

论文用 ALTI 风格 Manhattan attribution 把向量 contribution 变成非负、行归一化的 token transition：

\[
C_{ij}^{l}
=
\frac{
\left[
\|y_i^l\|_1-\|y_i^l-a_{j\to i}^{l}\|_1
\right]_+
}{
\sum_k
\left[
\|y_i^l\|_1-\|y_i^l-a_{k\to i}^{l}\|_1
\right]_+
}.
\]

然后按 Transformer 的真实层顺序相乘：

\[
C^{total}=C^L C^{L-1}\cdots C^1.
\]

最后一行是当前 next-token prediction 对输入 endpoints 的 complete contribution layout。矩阵乘法等于对所有合法 layer-ordered paths 的边权乘积求和；这比 last-layer attention、layer 平均或在折叠图上重复同一 diffusion step 更有机制依据。

论文还使用 Auto-Emergence 从完整图中贪心抽取 principal flow，并记录 token 进入主干流的顺序。

## 2. 最终 12 维 detector

外部 Qwen3 reranker 判断 context 对 question 的相关性，Text SHAP 通过 repeated masking 得到 token relevance layout。论文再计算：

```text
3 layouts/orderings: emergence, complete, principal
x 3 granularities: subword, word, phrase
= 9 RBO simulatability features

+ complete-layout KL concentration
+ principal-layout KL concentration
+ overall relevance score
= 12 features
```

这 12 维特征标准化后输入 XGBoost；正确/错误标签参与训练与验证。因此：

- attribution core 不是一个新 GNN；
- XGBoost 不是神经网络，但它是监督分类器；
- reranker 本身是神经网络；
- 基础生成 LLM 当然也是神经网络；
- 完整方法不是严格无监督，也不是一次简单 attention 统计。

论文任务是 RAG 短答案 sample-level trust estimation，greedy answer 最多约 10 tokens。当前项目研究较长回答的 token-level hallucination，任务边界不同。

## 3. 不能直接复制的内容

### 3.1 当前 cache 不足以复现 contribution

严格复现至少需要：

```text
per-layer input hidden states
per-head attention
per-head W_V and W_O or projected OV messages
residual/post-attention states
prompt query rows
next-token generating-position alignment
```

当前 cache 只有 response-query sparse attention、diagonal、unresolved 和 token IDs。raw attention 很大不等于 transported value 大；不同 heads 还可能在 OV space 抵消。因此只能实现 attention transport/rollout proxy。

### 3.2 concentration 没有固定正确性方向

论文的经验规律是可信答案的信息流更集中。但本项目历史结果显示，幻觉也可能进入简单、稳定、高度集中的错误 attractor。concentration 必须按 prompt/history 来源分解并作为诊断，不能预注册为“越高越正确”。

### 3.3 supervised calibrator 不属于无标签主方法

论文 XGBoost 可以作为 representation upper bound，但不能用来选择本项目的 layer、head、方向、loss weight 或主 detector。所有 label-free scores 必须先冻结，之后才能读取 test labels。

### 3.4 next-token alignment 当前不可伪造

预测 response token \(y_t\) 的 state 位于其前一个输入位置。第一个 response token 需要 prompt 最后一个 query row；当前 cache 没有它。当前实现只能诚实地做 post-hoc same-token routing representation，不能声称 trust-before-generation。

## 4. 论文代码审计注意事项

公开代码适合作为公式定位参考，不应直接复制。审计的公开 commit `263bec3` 存在以下实现/论文不一致或工程问题：

- 正文描述跨生成 token 平均，代码路径出现 coordinatewise max；
- 部分 CLI 参数名、目录名与读取路径不一致；
- reranker choices 列表存在字符串拼接问题；
- StandardScaler 在数据切分前拟合，造成评估分布泄漏风险；
- 最终 XGBoost 重新训练没有完整保留固定参数；
- repository 顶层没有清晰许可证声明。

当前实现依据论文数学对象重新设计，不复制作者代码。

## 5. 历史失败约束

旧 P-Cut 已经检验过 full/no-prompt/no-response 三视图和 closure：

\[
C_t=(L_t^{-R}-L_t^F)-(L_t^{-P}-L_t^F).
\]

全量 QA 的冻结方向 AUROC 为 `0.4209`。条件校准没有修复，位置基线更强。这个结果排除了“换一个 encoder 后把相同 closure 重新作为主方法”的做法。

当前优化必须满足：

- 不做 route cut；
- 不计算 closure；
- 不把 prompt mass 或 response mass 指定为 correctness direction；
- 不用标签翻转方向；
- 新目标必须保留完整 endpoint distribution，而不是再压回二值可达性。

## 6. 当前迁移：ordered endpoint layout

当前代码实现一个明确受限的代理。状态空间包含所有真实 token endpoints 和 unresolved sink \(\bot\)。对 response token：

\[
Q_t^0=e_t,
\]

\[
Q_t^l
=
\frac{
\alpha Q_t^{l-1}
+H^{-1}\sum_h
\left(
\sum_{s<t}a_{t,s}^{l,h}Q_s^{l-1}
+d_{t,h}^lQ_t^{l-1}
+u_{t,h}^le_\bot
\right)
}{\alpha+1}.
\]

它保留：

- actual layer order；
- exact source/target endpoint；
- causal response relay；
- sparse-cache missing mass；
- local encoder 中的 head identity。

它不保留：

- OV value direction/magnitude；
- head cancellation in representation space；
- real residual/FFN adoption；
- semantic relevance；
- next-token causal effect。

完整 layout 在跨层传播时固定均匀合并 heads，因此准确名称是 **head-merged layer-ordered attention transport endpoint layout**。

精确 target 只保存 response rows，内存为 (O(RN))；response relay 的主要代理工作量为 ((N+1)(LR+E_{RR}))。代码分别用 `layout_max_elements` 和 `layout_max_work_elements` 在 rollout 前限制这两项。上限只负责 fail-fast，不能替代正式数据最长样本上的 CUDA runtime/peak-memory 记录。

## 7. 为什么把它用作 target，而不是 hallucination feature

论文的最终 features 有正确性监督和外部 relevance reference。当前项目没有等价对象。如果直接规定“prompt concentration 高更正确”或“response concentration 高更错误”，会重复历史伪方向问题。

因此 endpoint layout 只承担一个无标签 representation objective：student 从 incidence/head 被遮蔽的图恢复 clean full-path layout。它要求 encoder 保留局部 row 之外的跨层 endpoint composition，但不预设哪种 layout 是 hallucination。

为避免 trivial target：

\[
\mathcal L_{layout}
=
\mathcal L_{sink}
+\mathcal L_{self\mid resolved}
+\mathcal L_{endpoint\mid nonself}.
\]

这样可降低 unresolved saturation 和 residual self identity 对 non-self endpoint loss 的直接淹没；它不能排除 position/length shortcut。non-self CE 还按候选 endpoint 数量的对数归一化，三项 loss 与 eligible-row coverage 分开记录。

## 8. 当前完整方法

```text
input: exact sparse typed graph
teacher 1: clean local row distributions
teacher 2: clean P/R/U ordered trajectories
teacher 3: clean ordered exact-endpoint layout
student input: mass-conserving corrupted graph
student: neural source -> row hyperedge -> target encoder
output: 64D response-token embeddings
detector: calibration-only PCA-whitened kNN
evaluation: frozen scores, then labels
```

训练目标：

\[
\mathcal L
=
\mathcal L_{row}
+\lambda_f\mathcal L_{P/R/U}
+\lambda_q\mathcal L_{layout}
+\lambda_v\mathcal L_{variance}.
\]

P/R/U 是 endpoint layout 的粗粒化，local rows 又是 layout 的输入，因此三个目标的信息并非独立。是否有增益必须靠 objective ablation 证明，不能从公式推出。

## 9. 必须运行的实验矩阵

### 9.1 objective ablation

```text
local only
local + P/R/U
local + endpoint
local + P/R/U + endpoint
```

### 9.2 path algebra controls

```text
correct layer order
reverse layer order
random layer permutation
last-layer/direct layout
layer-mean repeated transport
```

### 9.3 topology and shortcut controls

```text
real endpoints
role/lag/mass-matched endpoint rewire
weight shuffle
position-only decoder
self + unresolved decoder
direct deterministic layout + same detector
```

### 9.4 value-aware proxy audit

在可负担的小子集重新提取真实 contribution matrices，比较：

- endpoint JSD；
- top-k RBO；
- prompt/history conditional mass；
- 删除 top/bottom route tokens 后 observed-token logit drop；
- attention-only 与 value-aware layout 的 detector 结果。

如果 attention proxy 与 value-aware contribution 不一致，只能保留为 sparse attention rollout audit。

## 10. 当前允许的 claim

实现阶段只能说：

> We train a label-free directed attention-row hypergraph encoder to recover a sink-conservative, layer-ordered attention-transport distribution over exact token endpoints from mass-conserving corrupted graphs.

不能说：

- 复现了论文 Information Flow；
- 恢复了事实证据链；
- attention route 对 logits 有因果作用；
- ordered layout 已经改善 hallucination detection；
- 该方法不使用神经网络；
- 该方法已达到 SOTA。

真实 QA、完整 controls、多个 seed 和 value-aware faithfulness audit 完成后，才能升级 claim。
