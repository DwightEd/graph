# Directed Route Hypergraph: Held-out Typed Endpoint Recovery

当前主线研究一个比“从 attention 直接判幻觉”更可检验的问题：在保留
Transformer layer/head 类型的有向 attention-row 超图上，节点表征能否从被明确
遮蔽的局部结构中恢复真实 source endpoint，并优于 role/lag 匹配的因果非边。

```text
clean typed graph
  -> sample real (source, target, layer, head) edges
  -> force-hide sampled real edges
  -> neural directed-hypergraph encoder
  -> final node latent ranks real source above matched causal non-edge
  -> frozen node embeddings
  -> label-free readers / label-only readability diagnostics
```

encoder、训练目标和无监督 detector 都不读取 hallucination label。这里的“无标签”
不等于“没有神经网络”：source-to-hyperedge 聚合、route-conditioned slots、head pooling、
逐层 token update，以及可选的 variational posterior 都通过反向传播训练。

## 1. 为什么停止旧 ordered-layout 联合目标

旧配置同时训练 local row、P/R/U flow 和 ordered endpoint layout：

```text
rows_per_graph=256
layout_rows_per_graph=32
incidence_dropout=0.15
head_dropout=0.05
flow_weight=0.5
layout_weight=0.25
variance_weight=0.05
```

在 149 个 QA 样本、30,619 个 response tokens、2,307 个正例（prevalence
`0.075345`）上，64D 表征的最佳 validation loss 为 `1.946106`（epoch 5），但
检测结果接近随机，且弱于 position baseline：

| Frozen embedding reader | AUROC | AUPRC | AUPRC lift |
|---|---:|---:|---:|
| Autoencoder | 0.540130 | 0.081410 | 1.080 |
| Deep SVDD | 0.526438 | 0.078837 | 1.046 |
| Isolation Forest | 0.550779 | 0.082963 | 1.101 |
| LOF | 0.518836 | 0.076342 | 1.013 |
| PCA-kNN | 0.548162 | 0.083510 | 1.108 |

| Diagnostic | AUROC | AUPRC |
|---|---:|---:|
| Absolute position | 0.617076 | 0.112859 |
| Relative position | 0.606555 | 0.094919 |
| Linear probe on node embedding | 0.597574 | 0.096996 |
| Linear probe on position | 0.606064 | 0.101359 |
| MLP probe on node embedding | 0.552147 | 0.079619 |
| MLP probe on position | 0.568792 | 0.092920 |

在相同 149-sample / 30,619-token QA 数据规模上，已有 first-order GCN 的结果为：

| Reader | Directed hypergraph 64D | First-order GCN |
|---|---:|---:|
| PCA-kNN AUROC / AUPRC | 0.5482 / 0.0835 | **0.6982 / 0.1617** |
| Isolation Forest AUROC / AUPRC | 0.5508 / 0.0830 | **0.6362 / 0.1411** |
| Autoencoder AUROC / AUPRC | 0.5401 / 0.0814 | **0.5649 / 0.0935** |
| Linear probe AUROC / AUPRC | 0.5976 / 0.0970 | **0.7865 / 0.2999** |
| MLP probe AUROC / AUPRC | 0.5521 / 0.0796 | **0.7785 / 0.2760** |

因此旧方法不能继续写成“尚无正式结果”。它已经是负结果：更低的 reconstruction
loss 没有转化为可读的节点几何。训练也存在一个机制缺口：被评分的 clean positive
support 不一定从 student graph 中移除，模型可能从仍可见的同一 support 完成局部
重构。P/R/U 和 full layout 还与 local row 包含嵌套信息，使联合 loss 的下降难以定位
到某个有用的结构机制。

这个结果不单独证明“层序无效”或“超图无效”；它只否定当前 clean-support 联合目标
作为有效表征学习方案。新主线先回到与成功 GCN 更接近、且不会泄漏答案的 endpoint
recovery，再决定是否重新加入长路径辅助目标。

## 2. Exact typed graph

一个样本是一张独立 `TokenGraph`：

```text
node: token
edge: (source, response target, layer, head, retained attention weight)
clean row: retained + diagonal + native unresolved = 1
```

构图前不平均 layer/head，不把 cache 没保存的边当作零。prompt-query rows 不可用，
所以当前表征是 post-hoc same-token routing representation，不是 next-token trust
estimator。

每个 `(target, layer, head)` 是一个显式有向超边：

```text
source token -> attention-row hyperedge -> target token
```

模型保留四个 route-conditioned slots（默认 `4 x 16 = 64` hidden dimensions），
在每层先形成 head-specific row message，再更新 target token。容量不是当前负结果的唯一
解释：同为 64D 的 GCN 明显更强，所以首先必须修正训练问题，而不是只增加维度。

## Attention-routing mechanism gate

在继续训练 endpoint encoder、扩大 embedding 或增加 loss 之前，先运行一个确定性、
无参数、无 hallucination label 的 attention-routing 机制门。它不是复活 P-Cut：不做
`full / no-prompt / no-response` closure；也不把 AE/VAE reconstruction、endpoint ranking
或 posterior variance 当成主分数。该 gate 只回答当前 cache 能否支持下面三个相互区分
的机制轴：

| Axis | 机制问题 | 当前 cache 的观测边界 |
|---|---|---|
| lineage drift | prompt-rooted routing 是否沿层深和生成位置转为 response-rooted routing | 可观测，但只是 attention routing |
| routing dispersion | endpoint 支持是否集中、分散，以及不同 heads 是否选择不同 source role | 可在 censoring bounds 下观测 |
| parametric bias | 被关注内容是否经 OV 写入、被 residual/MLP 保留并推动 chosen-token logit | 不可观测；必须重新 replay OV、residual、MLP 和 logits |

特别地，response-rooted 祖先不等于 parameterized knowledge。artifact 必须记录
`drift_observed=true`、`dispersion_observed=true` 和
`parametric_bias_observed=false`；在 value-aware replay 完成前不得构造或填补 bias 分数。

### D/I/E/U lineage 与生成 token 对齐

对每个 response query 按真实 Transformer layer order 递推四类守恒 routing ancestry：

```text
D  direct prompt-rooted: 当前 query 直接读取 retained prompt endpoint
I  indirect prompt-rooted: prompt ancestry 经更早的 response carrier 到达当前 query
E  endogenous response-rooted: 路径根在 response-position input embedding
U  unresolved: sparse cache 未定位的 censored mass 及其后续传播
```

不插入 cache 没有观测到的 residual transition，也不把 `U` 并入 `E`。预注册的 drift
读数是完整逐层 `[D,I,E,U]` trajectory，以及最终 response takeover：

\[
g_t=\log\frac{E_t+\epsilon}{D_t+I_t+\epsilon}.
\]

对齐必须使用 predecessor query。若 response token 采用零基索引，则缓存 query `i`
预测 token `i+1`：第一个 response token 因缺少 last-prompt query 而显式 unavailable，
最后一个缓存 query 因预测回答外 token 而丢弃。不得用后面的可用 token 替换真实 onset；
same-token row 只作为 post-hoc misalignment control。

### Censoring-aware dispersion

未保留 endpoint 不是 observed zero。对一个 head-row，设 retained endpoint 与 exact
diagonal 的质量为 `p`，unresolved mass 为 `u`，可容纳它的 censored causal endpoints
数为 `m`。entropy 与 HHI 只报告可识别区间：

\[
H_{\min}=\sum_k-p_k\log p_k-u\log u,\qquad
H_{\max}=H_{\min}+u\log m,
\]

\[
Q_{\min}=\sum_kp_k^2+\frac{u^2}{m},\qquad
Q_{\max}=\sum_kp_k^2+u^2.
\]

区间先在每个 `(target,layer,head)` 上计算，再汇总到 predecessor-aligned token；禁止
用 floor、零或均匀分配产生伪 point estimate。另将每个 head 的 role mass 写为
`(prompt, earlier-response, diagonal, unresolved)`，用 generalized Jensen-Shannon
divergence 衡量 head-role disagreement。entropy/concentration bounds 与 head-role JSD
保持独立输出，不用手工权重与 drift 合成一个分数。

### 必要 controls 与停止门

同一批 rows、同一校准集和同一 score orientation 下固定运行：

```text
ordered                  真实 0 -> L-1 层序
reverse                  只反转层组合顺序
random-layer             非 identity 的固定随机层排列
last-layer               从初始 roots 只执行最后一个缓存层
response-carrier-rewire  只重连 R->R carrier，保留 prompt endpoints 与 row nuisances
same-token               将 query 错配给同位置 token 的 post-hoc control
```

carrier rewire 必须报告实际 changed-edge fraction；若稀疏图没有合法 swap，不能把它当作
有效 null。分数先在 source-disjoint、task 与绝对位置分箱的 calibration rows 上冻结，
之后才打开 test labels，并用 paired source bootstrap 比较。机制扩容的硬门是：

1. ordered drift 必须优于 reverse、last-layer 和 same-token；否则删除 layer-order /
   pre-generation claim；
2. drift 或 dispersion 至少一个必须优于 response ordinal、absolute sequence
   position、prompt length 与 offline response-length baselines；否则停止扩大
   attention-only encoder、embedding 和 AE/VAE 容量；
3. ordered 若不优于 random-layer，不能声称真实 chronology 有效；若不优于有效的
   response-carrier-rewire，不能声称 exact response relay 有效。

不满足这些条件时，训练 loss 下降不构成继续扩容的理由。OV/residual/MLP/logits replay
属于新的 parametric-bias 可观测阶段，而不是用更大模型挽救当前 attention-only gate。

QA 一键入口为：

```bash
bash experiments/directed_route_hypergraph/run_lineage_qa.sh
```

它依次生成 label-free calibration/test traces、冻结的 drift/dispersion scores 和
post-hoc evaluation。只生成并冻结机制 artifact、不打开标签时使用：

```bash
EVALUATE=0 bash experiments/directed_route_hypergraph/run_lineage_qa.sh
```

## 3. 新核心目标：强制 held-out typed endpoint recovery

对 clean graph 中采样的真实边
\(e^+=(s^+,t,l,h)\)，训练前强制从 student graph 删除该边。然后复用
`experiments.grounded_route.learning.matched_negative_edges`，在同一个 typed row
`(t,l,h)` 中采样不存在的 source \(s^-\)，并匹配：

- prompt/response source role；
- causal direction \(s^-<t\)；
- logarithmic source-target lag bucket；
- exact layer、head 和 target。

最终节点 latent 直接给真实与负 endpoint 打分：

\[
\mathcal L_{endpoint}
=
\frac{\sum_i a_i\,\operatorname{softplus}(s_i^- - s_i^+)}
{\sum_i a_i},
\]

其中 \(a_i\) 可保留 clean edge 的 attention mass。关键约束是：所有被当作 positive
监督的 edge 都必须出现在 forced holdout 集合中，student 不能看见该 edge；score 必须
读取最终 decoder/node latent，而不是读取遮蔽前的 clean support 或只读某层临时状态。

这比旧目标更接近一个必要的机制命题：如果 exact typed topology 对 token 表征有用，
表征应能在相同 role、相似距离和相同 `(target,layer,head)` 的困难对照中恢复真实
endpoint。若 real 与 matched non-edge 无差异，就不能把任何下游增益归因于 exact
endpoint structure。

## 4. Native unresolved 与 artificial masked mass 必须分离

两个“看不见”具有不同语义：

| Channel | 来源 | 可恢复性 | 含义 |
|---|---|---|---|
| `native unresolved` | sparse attention cache 原本未保存的质量 | endpoint 未知 | 数据采集造成的 censoring |
| `masked mass` | 训练时主动移除的已知 retained edge | teacher 知道 endpoint | endpoint-recovery 任务的 corruption |

旧 corruption 将人工删除质量加进 `unresolved`，会把“原生未知”与“人为遮蔽但可恢复”
混成一个 sink。新实现保持 clean `TokenGraph` 的
`retained + diagonal + native unresolved = 1` 契约，并通过 student-only
`masked_mass` channel 传入人工遮蔽量；corrupted view 满足
`retained_kept + diagonal + native unresolved + masked = 1`。clean teacher 永远使用
未破坏的 graph。

`native unresolved` 不是无效信息、事实不确定性或 hallucination 类别；`masked_mass`
也不是一个新 token endpoint。二者不得在 artifact 或论文叙事中互换。

## 5. Deterministic baseline 与可选 VAE

确定性模式是正式公平基线：默认 hidden state 为 64D，训练和导出使用同一个确定性
latent。只有确定性 endpoint-recovery 已超过旧目标并接近/超过 GCN，才有依据把 VAE
增益解释为 posterior regularization，而不是额外参数或随机训练带来的偶然结果。

VAE 必须显式开启。对 hidden state \(h_t\)：

\[
q(z_t\mid G_{masked})
=\mathcal N(\mu_t,\operatorname{diag}(\exp(\log\sigma_t^2))).
\]

准确名称是“stochastic variational bottleneck + typed endpoint decoder”。当前没有
clean-posterior/corrupted-prior 双编码分支，因此不是完整的 conditional VAE，也不能
声称已经辨识多模态真实路径后验。

训练 decoder 使用 reparameterized sample \(z_t\)，评估固定使用 \(\mu_t\)，避免同一
checkpoint 因采样产生不同 embedding。KL 使用小权重、free bits 和 warmup，不能让
posterior 在 endpoint objective 学到结构之前塌缩。

导出维度由 `VAE_EXPORT` 决定：

| Mode | Export | 默认 hidden=64 时的维度 |
|---|---|---:|
| deterministic | node state | 64 |
| VAE | `mean` | 64 |
| VAE | `mean_logvar` | 128 |

评估器接受任意 embedding dimension；PCA-kNN 内部投影到较低维不等于 encoder 只能
输出 32D。`mean_logvar` 的 128D 只是把均值和逐维 log-variance 拼接，不自动意味着
更强表征。

最重要的 claim boundary：posterior variance 描述的是在当前训练 corruption 和
Gaussian bottleneck 下 latent 的分散程度。它不是事实不确定性、语言模型置信度或
hallucination score，不得单独作为最终 detector，也不得用 label 事后选择符号。

## 6. 与 *Information Flow Reveals When to Trust Language Models* 的关系

论文先在冻结 LLM 上计算 value-aware contribution，再按真实层序组合每层 transition。
它的信息流抽取本身不训练新 encoder，但完整 trust detector 还使用神经 reranker、
SHAP relevance 和 correctness-supervised XGBoost。因此不能概括成“没有神经网络，只
计算简单特征”。

当前 cache 没有 hidden state、\(W_V/W_O\)、真实 residual message 或 prompt-query
rows，不能复现 functional contribution。旧 ordered layout 只迁移了非交换层转移的
组合代数，而且其联合训练结果已经失败。新 endpoint-recovery 主线保留 exact
layer/head/target typing，不把 attention rollout 冒充 contribution、causal effect 或
事实 grounding。

ordered P/R/U 和 endpoint-layout decoder 仍可作为显式 auxiliary ablation，但默认
权重为零；它们只有在 endpoint-only 确定性基线成立后，且 ordered 明显优于 reverse、
last-layer 和 matched rewire 时，才可能恢复为论文机制的一部分。

## 7. 运行与最小实验矩阵

正式 QA 入口保持唯一：

```bash
bash experiments/directed_route_hypergraph/run_qa.sh
```

本次 objective 与 decoder 均已改变，checkpoint 使用新的 method ID 和
`architecture_version=2`。旧 ordered-layout checkpoint 会在加载权重前被拒绝；新实验
必须使用新输出目录并从 `START_STAGE=1` 重新训练，不能只重新执行 encode/evaluate。

当前 endpoint-only deterministic 默认值为：

```text
POSITIVE_EDGES_PER_GRAPH=4096
HOLDOUT_FRACTION=0.15
NEGATIVE_COUNT=1
NEGATIVE_ATTEMPT_FACTOR=8
INCIDENCE_DROPOUT=0
HEAD_DROPOUT=0
FLOW_WEIGHT=0
LAYOUT_WEIGHT=0
VARIANCE_WEIGHT=0.05
SLOT_DIM=16
EDGE_HIDDEN_DIM=64
LATENT_MODE=deterministic
```

forced endpoint holdout 本身已经提供 corruption；默认关闭额外 incidence/head dropout，
避免第一轮公平基线混入第二种遮蔽分布。flow/layout 权重为零时对应 teacher/decoder
目标必须真正 bypass。

VAE 是显式配置，不应覆盖确定性输出目录：

```bash
LATENT_MODE=vae VAE_EXPORT=mean_logvar \
bash experiments/directed_route_hypergraph/run_qa.sh
```

每个 `RUN_NAME`/输出目录必须编码 objective、latent mode、export、slot dimension、KL
权重和 seed，防止 64D deterministic 与 128D VAE artifact 静默互相覆盖。

第一阶段只运行：

```text
A. deterministic endpoint recovery
B. deterministic endpoint recovery + old P/R/U auxiliary
C. deterministic endpoint recovery + old ordered-layout auxiliary
D. VAE(mean) endpoint recovery
E. VAE(mean_logvar) endpoint recovery
```

每个表征都用相同 source split、训练预算、frozen detector 与 seeds。必须报告：

- held-out endpoint pair count、forced mask coverage 和 ranking loss；
- native unresolved 与 artificial masked mass 的守恒测试；
- PCA-kNN、Isolation Forest、Autoencoder、Deep SVDD、LOF；
- position baselines 与 linear/MLP readability diagnostics；
- embedding dimension 和 VAE KL/posterior diagnostics；
- GCN 的同协议结果，而不是只与旧失败模型比较。

## 8. 当前允许的结论与停止规则

实现阶段允许说：

> We train a label-free directed attention-row hypergraph encoder to recover
> forced-held-out typed source endpoints against role- and lag-matched causal
> non-edges, while separating native cache censoring from artificial masks.

当前不能说：

- VAE 已改善 hallucination detection；
- posterior variance 是 hallucination uncertainty；
- 复现了 Information Flow 的 functional contribution；
- exact topology、layer order 或超图结构已经有效；
- reconstruction/ranking loss 是 hallucination score；
- 当前方法达到或超过 GCN/SOTA。

停止规则：

```text
endpoint recovery ~= matched non-edge  -> exact typed endpoint 机制不成立
embedding <= position readability      -> 表征仍被捷径或塌缩主导
deterministic < GCN                     -> 不用 VAE 掩盖公平基线缺口
VAE <= deterministic                    -> 去掉 variational 模块
mean_logvar gain only                   -> 审计 variance 是否只是长度/coverage 编码
ordered auxiliary ~= reverse           -> 不提出 layer-order 信息流 claim
real endpoint ~= matched rewire         -> 不提出 exact-topology claim
```

训练 loss 下降只说明优化目标可拟合；最终结论必须来自冻结表征、统一评估器、多个 seed
和 source bootstrap。
