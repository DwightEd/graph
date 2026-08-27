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

## 6. 已完成但失败的迁移：ordered endpoint layout

旧版本实现了一个明确受限的代理。状态空间包含所有真实 token endpoints 和 unresolved sink \(\bot\)。对 response token：

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

精确 target 只保存 response rows，内存为 (O(RN))；response relay 的主要代理工作量为 ((N+1)(LR+E_{RR}))。代码分别用 `layout_max_elements` 和 `layout_max_work_elements` 在 rollout 前限制这两项。上限只负责 fail-fast，不能替代正式数据最长样本上的 CUDA runtime/peak-memory 记录。该 proxy 仍保留作辅助消融，但不再是默认主目标；正式负结果见下一节。

## 7. Ordered layout 已经得到负结果

论文的最终 features 有正确性监督和外部 relevance reference。当前项目没有等价对象，
所以旧实验把 endpoint layout 只作为无标签 representation target，而没有预设哪种
layout 是 hallucination。为降低 sink/self 捷径，它把 layout target 分为 sink、self
和 conditional non-self endpoint 三项。

但这套 target 与 local clean-row、P/R/U flow 联合训练后的正式 QA 结果已经失败。
在 149 个样本、30,619 个 tokens、2,307 个正例上，64D embedding 的 PCA-kNN 为
`0.548162 / 0.083510`，linear readability probe 为
`0.597574 / 0.096996`。absolute-position baseline 为
`0.617076 / 0.112859`，first-order GCN 的 PCA-kNN 和 linear probe 分别为
`0.6982 / 0.1617`、`0.7865 / 0.2999`。

因此不能再把 ordered layout 写成“尚待首次验证”，也不能用 validation loss
`1.946106` 代替检测证据。该 run 的多个嵌套 objective 同时开启，所以结果只否定当前
联合 clean-support 目标，尚未单独否定 layer order。任何后续 ordered claim 都必须把
它作为 endpoint-only 基线上的独立 auxiliary delta，并对比 reverse/shuffled controls。

## 8. 当前完整方法：held-out typed endpoint recovery

当前主线不再把 full ordered layout 作为默认核心目标。它改为一个更直接、可审计的
结构恢复任务：

```text
input: exact sparse typed graph
positive: clean retained (source, target, layer, head) edge
student: graph with every scored positive edge forcibly removed
negative: causal non-edge matched on source role and logarithmic lag
score: final decoder/node latent under the same target/layer/head
loss: weighted positive-versus-negative ranking
output: frozen response-token embedding
detector: calibration-only label-free readers
evaluation: save frozen scores, then open labels
```

核心损失为：

\[
\mathcal L_{endpoint}
=
\frac{\sum_i a_i\operatorname{softplus}(s_i^- - s_i^+)}{\sum_i a_i}.
\]

它复用已有 role/lag-matched negative sampler，避免重新实现另一套近似负采样。这个
目标的必要机制判断是：若最终表征无法区分真实 held-out endpoint 和相同角色、相似
距离、相同 `(target,layer,head)` 的非边，就没有证据声称 exact typed topology 有用。

人工 mask 和 native unresolved 也明确分离：

```text
native unresolved = sparse cache 原本未知的 endpoint mass
masked mass        = teacher 已知、训练时主动隐藏的 retained mass
```

人工遮蔽不得写回 graph 的 `unresolved`，否则模型无法区分数据 censoring 和 recovery
task。可选 P/R/U/layout teacher 始终读取 clean graph。

## 9. VAE 的作用与 claim boundary

确定性 64D endpoint-recovery 是公平基线。VAE 只能作为显式 ablation：训练时 decoder
可读取 reparameterized sample，评估时固定读取 posterior mean；`mean` export 保持
64D，`mean_logvar` 把两者拼接为 128D。维度翻倍不自动构成优势，也不能用不同输出
维度掩盖训练目标本身无效。

VAE 可能提供的只是：对人工结构 censoring 的 stochastic regularization，以及把
逐维 posterior dispersion 暴露给下游 reader。它不能自动修复错误的 graph target，
也不能仅凭 KL 或 reconstruction loss 推出 hallucination detection 改善。

特别禁止以下等同：

```text
posterior variance != factual uncertainty
posterior variance != language-model confidence
posterior variance != hallucination probability
endpoint ranking loss != hallucination score
native unresolved != model uncertainty
```

如果 `mean_logvar` 优于 `mean`，还必须控制 position、response length、retained
coverage 和 native unresolved mass，排除 log-variance 只编码 cache 密度的解释。

## 10. 必须运行的实验矩阵

### 10.1 objective 与 posterior ablation

```text
deterministic endpoint-only
deterministic endpoint + P/R/U
deterministic endpoint + ordered layout
VAE endpoint-only, mean export
VAE endpoint-only, mean_logvar export
```

### 10.2 topology 与 path controls

```text
real endpoint versus role/lag-matched endpoint rewire
real weights versus weight shuffle
correct layer order versus reverse/random/last-layer auxiliary
full encoder versus position-only control; no-message requires a dedicated
clean-teacher/separate-student implementation and is not the generic graph
variant in the current endpoint-recovery runner
first-order GCN under the same evaluator
```

### 10.3 implementation gates

```text
every supervised positive is absent from the student graph
negative endpoint is a verified non-edge in the exact typed row
native unresolved is unchanged after artificial masking
masked mass is conserved in its separate student-only channel
final node latent receives endpoint-loss gradient
VAE evaluation export is deterministic
artifact records actual embedding dimension and architecture version
```

### 10.4 value-aware proxy audit

在可负担的小子集重新提取真实 contribution matrices，比较 endpoint JSD、top-k RBO、
prompt/history conditional mass，以及删除 route tokens 后的 observed-token logit drop。
如果 attention proxy 与 value-aware contribution 不一致，只能保留为 sparse attention
graph audit。

## 11. 当前允许的 claim

实现阶段只能说：

> We train a label-free directed attention-row hypergraph encoder to recover
> forced-held-out typed source endpoints against role- and lag-matched causal
> non-edges, while separating native cache censoring from artificial masks.

不能说：

- 复现了论文 Information Flow；
- 恢复了事实证据链；
- attention route 对 logits 有因果作用；
- VAE/posterior variance 是 hallucination uncertainty；
- endpoint loss 是 hallucination score；
- ordered layout、exact topology 或 VAE 已经改善 detection；
- 该方法不使用神经网络；
- 当前方法已超过 GCN 或达到 SOTA。

正式 QA、matched controls、多个 seed、source bootstrap 和 value-aware faithfulness
audit 完成后，才能升级对应 claim。

## 12. 下一版实现：Functional-Flow DAG

当前 cache 无法通过后处理补出 OV、hidden state 或 prompt-query rows。下一版必须从
冻结 LLM 重新提取，不能把 attention rollout 改名为 functional contribution。

### 12.1 一次 teacher-forced replay 提取什么

对完整的 `prompt + 已生成 response` 做严格 causal replay。对每层保存或在线计算：

```text
token / role / generation-step alignment
all causal query rows of per-head attention A
pre-attention residual x
per-head projected value W_O^h W_V^h LN(x_j)
attention output and post-attention residual
MLP output and post-MLP residual
next-token logits or selected-token unembedding direction
```

预测 response token `y_t` 的 query position 是它的前一个位置；`y_1` 对齐 prompt
最后一个位置。没有这个 shift，得到的是看到 token 之后的 post-hoc 表征，不是生成它
之前的 trust 表征。

不能落盘 `[L,H,T,T,D]`。实现应按 target/source block 在线计算 source contribution，
先把 heads 在 OV space 中合并，再写出：

```text
per-layer scalar transition C_l       CSR top-k + exact dropped-mass sink
per-layer/per-head signed sketch      optional, only for retained endpoints
hidden/residual/attention/MLP sketch  fixed random projection to 64--128D
complete layout checkpoints           products over selected depth windows
```

全 token 的 causal query rows 是跨层全路径所必需的；只存 response-query rows会把
prompt 内部的中继路径冻结掉。若只能承担小子集，先在该子集做 attention-versus-OV
faithfulness audit，不要先大规模训练另一个 attention-only encoder。

### 12.2 构图对象

主图应从折叠 token 图改为 layer-expanded causal DAG：

```text
node                 (layer, token position)
cross-token edge     (l-1, source) -> (l, target), weight C_l[target, source]
residual self edge   (l-1, target) -> (l, target)
MLP self transform   token-local edge carrying the observed state delta/sketch
output node          selected response token, aligned to predecessor query
```

原始 attention 是“读哪里”；`A * OV` 是“搬运什么”；真实 layer order 是“怎样复合”；
selected-token logit direction 是“是否推动了这个输出”。非负 ALTI transition 适合稳定
rollout，但会抹掉 head cancellation。正式方法至少应在 retained edges 上保留 signed
OV sketch 或 selected-logit signed projection，把非负 layout 只作为并行稳定通道。

### 12.3 无标签学习目标

首选确定性 128D/256D layer-wise graph encoder，不先上 VAE。训练目标按以下顺序加入：

1. **masked functional endpoint recovery**：复用当前 role/lag-matched sampler，隐藏真实
   typed endpoint，同时预测 endpoint、贡献强度和符号；teacher 必须来自 clean graph；
2. **state conservation**：聚合的 OV contribution + residual 必须重建 post-attention
   state sketch，MLP self transform 必须重建下一层 state sketch；
3. **path-composition consistency**：若干连续层的预测 transition product 必须匹配直接
   计算的 multi-layer complete layout，而不只重建单层边；
4. **order-sensitive contrast**：真实 layer order 与同一样本的 reverse/shuffle/endpoint
   rewire 区分开，避免只编码 position 或度数；
5. **two-view invariance**：同一 clean flow 的两种 top-k/censoring view 表征接近，
   但 native unresolved 与 artificial mask 始终使用不同状态。

最终对 `y_t` 导出其 predecessor query 的多尺度 flow state，包括 incoming evidence、
depth trajectory、complete-layout summary 和 state sketch。无监督 detector 只读取这个
冻结表征。VAE 只有在确定性 representation 已超过 attention-only、position 和 GCN
后，才作为 corruption uncertainty 的独立消融。

### 12.4 论文机制与可证伪条件

可用的核心叙事是 **Censored Functional-Flow Completion**：attention cache 只观察到
不完整路由；模型必须从上下文和已观察到的跨层动态恢复被遮蔽的功能贡献，并同时满足
局部状态守恒与全路径复合一致性。

这个设计的“非这样不可”来自四个可独立证伪的必要条件：

```text
attention-only < OV-aware       否则 value-aware 机制没有必要
unordered < ordered             否则跨层路径叙事没有必要
edge recovery < + composition   否则完整全链路目标没有必要
rewired < real endpoints        否则 exact topology 没有必要
```

任何一条长期不成立，都应删除相应机制 claim，而不是靠 VAE 或更大 embedding 维度掩盖。
