# Directed Route Hypergraph

本目录检验一个可反证的假设：上下文幻觉可能不是“关注历史 token 太多”，而是生成路径逐层从 prompt-origin routing 转向 response-closed self-conditioning。方法把确定性的跨层路径流作为结构 teacher，再训练一个去噪有向超图编码器；最终仍输出每个 token 一个 64 维神经表征。P 只表示路径追溯到某个 prompt token，不表示该 token 与问题相关或构成有效证据。

当前缓存只有稀疏 attention，因此本文档始终使用 **routing provenance**，不把它称为 functional contribution。

## 与 Information Flow 方法的关系

`Information Flow Reveals When to Trust Language Models` 对每层真实 (W_V/W_O)、hidden state 和 residual 计算 source contribution，再按 Transformer layer 顺序相乘 contribution matrices。它的 attribution 算子不训练新 encoder，但完整系统还使用 neural reranker、SHAP 和 correctness-label XGBoost。

本目录迁移的是“逐层有序路径组合”，没有声称复现论文的 OV contribution。现有缓存缺少 hidden state、(W_V/W_O) 和 residual stream，只能构造 attention-only 近似。显式 `residual_weight` 是预注册 proxy，不是真实 residual attribution。

## 1. 质量守恒的有序路径 teacher

一条保留 incidence 为

\[
(s,t,l,h,a_{t,s}^{l,h}),\qquad s<t.
\]

每个 token 的路径来源为三态分布：

```text
P  prompt-origin
R  response-origin / response-closed
U  unresolved because of sparse-cache censoring
```

初始 prompt token 为 `(1,0,0)`，response token 为 `(0,1,0)`。对 row `e=(t,l,h)`：

\[
\rho_{t,h}^{l}
=
\sum_{s<t}a_{t,s}^{l,h}\pi_s^{l-1}
+d_{t,h}^{l}\pi_t^{l-1}
+u_{t,h}^{l}(0,0,1).
\]

构图保证

\[
\sum_{s<t}a_{t,s}^{l,h}+d_{t,h}^{l}+u_{t,h}^{l}=1,
\]

所以每个 `head_flow[t,l,h]` 也严格和为 1。跨层 token state 使用 head-uniform merge 和显式 residual proxy：

\[
\pi_t^l=
\frac{\alpha\pi_t^{l-1}+H^{-1}\sum_h\rho_{t,h}^{l}}
{\alpha+1}.
\]

prompt-query rows没有缓存，因此 prompt states 始终固定。实现位于 `flow.py`，保存的机制张量为 `[response, layer, head, P/R/U]`。

这个算子不训练参数。它与 contribution-matrix product 的共同点是 layer 顺序不可交换；不同点是边权只有 attention，不含 OV 或真实 residual contribution。

## 2. Provenance-aware node -> hyperedge -> node

每个 `(target,layer,head)` 是一个有向 row hyperedge：

```text
source token -> attention-row hyperedge -> target token
```

编码器保留四个 16 维槽：

```text
P1/P2  prompt-origin-conditioned learned source-set summaries
R1/R2  response-closed-conditioned learned source-set summaries
```

这里按完整路径来源给消息加权，而不是按 source token 的直接角色分流。一个 response source 只要它本身承载 prompt provenance，就继续给 P-conditioned slots 提供质量；这避免把 `prompt -> earlier response -> current response` relay 全部错算成 response closure。

这些槽不是数学上纯净的 P/R factorization：初始 node state、layer/head context 和 GRU recurrence 会被共享保留。准确说法是 route-conditioned slots；其纯度必须通过 zero-route gating 和 slot intervention 另行验证。

对 incidence 的两条可解析质量为

\[
w_{e,s}^{P}=a_{e,s}\pi_{s,P}^{l-1},\qquad
w_{e,s}^{R}=a_{e,s}\pi_{s,R}^{l-1}.
\]

每个 route 内用 learned slot attention 聚合非线性 source messages，attention weight 只作为 prior。聚合后的内容乘回 route mass；diagonal 也按 target 的 P/R provenance 拆分。由 retained paths、diagonal 和当前 row censoring 共同产生的 U mass 以独立 learned message 注入。随后同一 target 的 heads 做 learned pooling，四个 GRU 沿 layer depth 更新 target node。

因此该模型不是邻居向量的简单累加。解析 teacher 不训练，而 `SourceToHyperedge`、slot queries、head pooling、GRU、row decoder 和 flow decoder 都是可训练神经网络。

## 3. 守恒式去噪训练

训练时只使用符合缓存观测模型的 corruption：

- 随机隐藏 retained incidence；
- 随机隐藏完整 `(layer, head)` channel；
- 被隐藏的质量原样转入对应 row 的 `unresolved`；
- 不重连 endpoint，不改变 diagonal，row 总质量仍为 1。

`endpoint_rewire` 和 `weight_shuffle` 只作为独立训练的 null controls，不能作为增强。

student 编码受扰动图，但目标来自干净图：

\[
\mathcal L=
\mathcal L_{row}
+\lambda_f\mathcal L_{flow}
+\lambda_v\mathcal L_{variance}.
\]

`row` 目标在干净图给定的 retained support、SELF 和 UNRESOLVED 候选上恢复质量分配；它不制造 censored non-edge negatives，但也不能证明模型学会了发现未给定的 endpoint support。`flow` 目标从每层 post-update node state 解码干净的 \((P,R,U)\) ordered-flow trajectory：

\[
\mathcal L_{flow}
=-\frac1{TL}\sum_{t,l}\pi_{t}^{l,*}
\log\operatorname{softmax}(g(x_{t,l})).
\]

hallucination labels 不进入 encoder、early stopping 或 detector fitting。去噪重构误差也不作为 hallucination score；一个稳定错误 attractor 可能同样容易重构。

## 4. 导出与检测

最后四槽展平并归一化：

\[
z_t=\operatorname{LayerNorm}
(P1\Vert P2\Vert R1\Vert R2)\in\mathbb R^{64}.
\]

下游只读取冻结的 64D `node_embedding`：

```text
64D node embedding -> robust scaling -> PCA whitening -> mean kNN distance
```

calibration 与 test 按 `source_id` 隔离。test labels 在 token scores 保存之后才打开，用于 AUROC/AUPRC 和 source bootstrap。

每样本 `graphs/*.pt` 还保存 exact typed edges、diagonal、unresolved 和解析式 `[R,L,H,3]` path flow，供机制审计；这些 sidecar 不进入 detector。

## 5. 一键运行 QA

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main
bash experiments/directed_route_hypergraph/run_qa.sh
```

小规模 smoke test：

```bash
EPOCHS=1 \
TRAIN_LIMIT=32 \
TEST_LIMIT=16 \
OUT=experiments/directed_route_hypergraph/outputs/smoke \
bash experiments/directed_route_hypergraph/run_qa.sh
```

默认去噪参数可从环境覆盖：

```text
INCIDENCE_DROPOUT=0.15
HEAD_DROPOUT=0.05
FLOW_WEIGHT=0.5
RESIDUAL_WEIGHT=1.0
```

设 `INCIDENCE_DROPOUT=0 HEAD_DROPOUT=0 FLOW_WEIGHT=0` 可得到同一实现中的 row-only ablation。

## 6. 已实现对照、后续对照与停止条件

当前一键入口已经支持：

```text
row-only          INCIDENCE_DROPOUT=0 HEAD_DROPOUT=0 FLOW_WEIGHT=0
no_message        VARIANT=no_message
endpoint_rewire   VARIANT=endpoint_rewire
weight_shuffle    VARIANT=weight_shuffle
residual alpha    RESIDUAL_WEIGHT=0/0.5/1/2 分别训练
```

正式机制结论前仍需接入同一 evaluator 的 `ordered vs reverse`、one-step、position/length、deterministic-flow-only controls；当前代码和结果不能暗示这些对照已经完成。

若未来 ordered 不优于 reverse、real 不优于 endpoint-rewire/weight-shuffle，结果随 residual proxy 大幅变化，或信号主要由位置、长度、unresolved mass 解释，就不能把增益归因于真实跨层路径。此时应停止增加 GNN 容量，转而采集 OV/residual caches。

## 解释边界

当前实现是 post-hoc same-token routing representation，不是论文的 trust-before-next-token estimator。首个 response token 的生成前状态需要 prompt 最后一行，而当前缓存没有 prompt-query row。

现有数据不能验证：

- token 内容冲突；当前节点初态只有角色与位置，token IDs 不进入 encoder；
- attention route 对 logits 的真实功能贡献；
- FFN parametric memory 与 prompt-origin routing 的竞争；
- value/residual message 的语义冲突、谱熵或有效秩；
- 某条 route 对错误输出的因果作用。

这些主张需要补采 per-head OV message、attention/FFN 前后 residual、logit attribution，并加入 activation patching 或 head/endpoint intervention。
