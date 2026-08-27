# Directed Route Hypergraph

本目录检验一个明确而可反证的假设：在生成每个 response token 时，模型并不是只依赖一个无类型的邻域，而是在 Transformer 的不同 layer/head 中，在两类路由之间动态分配注意力质量：

```text
prompt route     从问题、检索上下文和约束条件读取证据
response route   从已经生成的 response 读取并延续当前生成状态
```

上下文幻觉可能对应 prompt route 减弱、response route 逐步闭合并稳定自我强化。但当前输入只有稀疏 attention 权重，所以这里检验的是 **attention routing signature**，不是 Transformer 的完整 functional information flow。只有真实图显著优于保持边际统计的构图 controls，才能说明具体端点、权重配对和邻居传播确实有用。

## 图与张量

一条 prompt-response 样本是一张因果 token 图。每条保留的 attention incidence 为

\[
(s,t,l,h,a_{t,s}^{l,h}),\qquad s<t,
\]

其中 `s` 是 source token，`t` 是 response target token，`l` 是 Transformer layer，`h` 是 attention head。构图前不平均 layer 或 head。

对每个 row

\[
e=(t,l,h),
\]

创建一个有方向的 attention-row hyperedge。计算路径严格为

```text
source token node
    -> (target, layer, head) row hyperedge
    -> target token node
```

实现中的主要张量为：

```text
source       [I]          incidence 的 source token
hyperedge    [I]          incidence 所属 row
weight       [I]          retained attention weight
role         [I]          prompt / response source
target       [R]          每个 row 的 target token
head         [R]          每个 row 的 head
diagonal     [R]          self-attention mass
unresolved   [R]          稀疏缓存未保留的 censored mass
node_state   [N, 4, 16]   token 的四槽状态
hyperedge    [R, 4, 16]   row hyperedge 的四槽状态
```

`I` 是保留 incidence 数，`R=response_tokens × heads`，`N` 是 token 数。

## 四个 route slots

每个 token 保留四个 16 维槽，最终恰好得到 64 维表示：

```text
slot P1, P2   prompt-source set 的两个独立 learned summaries
slot R1, R2   response-source set 的两个独立 learned summaries
```

P1/P2 或 R1/R2 没有预先指定“实体”“数字”等手工语义。每个角色使用两个独立 slot query，使同一 attention row 可以保留两种互补的 source coalition，而不是把所有邻居压成一个均值。slot 的实际功能必须通过 slot ablation、端点干预和可视化后验解释，不能从名字直接宣称。

## Node -> hyperedge -> target

### 1. Source node 到 row hyperedge

对 incidence \(s\to e\)，先构造带类型的 source message：

\[
m_{e,s,k}=\phi_{r(s),k}
\left(
x_s,\;l,\;h,\;\operatorname{bucket}(t-s),\;\operatorname{lineage}_{s\to e}
\right),
\]

其中 \(k\) 是该 source role 的两个 slots 之一。row 内不是直接求和，而是使用 attention weight 作为 learned slot attention 的先验：

\[
\beta_{e,s,k}
=
\operatorname{softmax}_{s\in S_e^{r}}
\left(
\frac{q_{r,k}^{\top}K m_{e,s,k}}{\sqrt d}
+\log(a_{t,s}^{l,h})
\right).
\]

归一化后的内容和该角色的真实 retained mass 分开计算：

\[
\mu_{e,k}=\sum_{s\in S_e^r}\beta_{e,s,k}V m_{e,s,k},
\qquad
M_e^r=\sum_{s\in S_e^r}a_{t,s}^{l,h},
\]

\[
z_{e,k}=M_e^r\mu_{e,k}
+d_e\,\phi_{\mathrm{self}}(x_t)
+u_e\,b_{l,h,k}.
\]

这里 \(d_e\) 是 diagonal mass，\(u_e\) 是 unresolved mass。先对邻居内容做归一化、再乘回 route mass，可以避免节点度数直接决定表示尺度，同时不丢失 prompt/response 使用量。

### 2. Row hyperedge 到 target node

同一 target 和 layer 下的 heads 仍然分开形成 row states，然后按 slot 做 learned head pooling：

\[
\gamma_{t,l,h,k}
=
\operatorname{softmax}_{h}
\left[
w^{\top}\tanh(Qx_{t,l-1,k}+Kz_{t,l,h,k})
\right],
\]

\[
c_{t,l,k}=\sum_h\gamma_{t,l,h,k}z_{t,l,h,k},
\qquad
x_{t,l,k}=\operatorname{GRU}_k(c_{t,l,k},x_{t,l-1,k}).
\]

这一步沿 Transformer depth 依次更新 response nodes。因此，该编码器不是普通的邻居特征累加：它包含 source-role 分流、多个 source-set slots、row-level hyperedges、head-specific pooling、mass 保留和 layer-wise recurrent update。

训练时每个完整 layer step 使用 non-reentrant activation checkpointing。反向传播会重算该层，避免同时保留 32 层中所有 incidence-sized message 激活；同一层的 edge tensors 在 lineage 与 hyperedge 构造间复用。编码阶段处于 `eval/no_grad`，不会产生重算开销。

## Censoring-aware 守恒 row 目标

稀疏缓存未保存的 attention 只表示低于缓存阈值，不能当作真实零边，也不能自动当作负样本。对 row \(e=(t,l,h)\)，目标分布保留三部分：

\[
q_e(s)=a_{t,s}^{l,h}\quad(s\in E_e),
\qquad
q_e(\mathrm{self})=d_e,
\qquad
q_e(\mathrm{unresolved})=u_e,
\]

并满足 row 质量守恒：

\[
\sum_{s\in E_e}q_e(s)+d_e+u_e=1.
\]

encoder 使用进入当前 layer 前的 node states，对已知 retained support 中的 endpoints、self bucket 和 unresolved bucket 评分，并在同一个 row 内归一化：

\[
\mathcal L_{\mathrm{row}}
=-
\frac{1}{|\mathcal R|}
\sum_{e\in\mathcal R}
\left[
\sum_{s\in E_e}a_{t,s}^{l,h}\log p_e(s)
+d_e\log p_e(\mathrm{self})
+u_e\log p_e(\mathrm{unresolved})
\right].
\]

训练不读取 hallucination labels。这个目标学习的是**给定 retained support 后的质量分配**，要求表示保留可观测 row 的相对端点质量以及 self/unresolved 质量结构；它不把 censored endpoints 伪造为可靠负例，也不把 retained weights 重新归一化后丢掉 self/unresolved mass。由于候选集合本身包含真实 retained support，该目标不能单独证明模型学会了预测哪些 endpoint 应当存在。

## 节点表征与无监督检测

最后一层的四槽状态被展平并归一化：

\[
z_t=\operatorname{LayerNorm}
\left[
x_{t,L,P1}\Vert x_{t,L,P2}\Vert
x_{t,L,R1}\Vert x_{t,L,R2}
\right]\in\mathbb R^{64}.
\]

`encode` 阶段为每个 token 保存一个 64 维 `node_embedding`。下游 detector 不再读取边或执行第二个 GNN：

```text
64D node_embedding
    -> median/MAD normalization
    -> PCA whitening
    -> mean kNN distance
    -> token anomaly score
```

PCA-kNN 只在 source-disjoint 的无标签 calibration embeddings 上拟合。hallucination labels 在分数保存后才用于 AUROC、AUPRC 和 source bootstrap 评价。

## 运行

服务器 QA 全流程：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git pull --ff-only origin main
bash experiments/directed_route_hypergraph/run_qa.sh
```

默认执行：

```text
fit label-free encoder
encode calibration nodes
encode test nodes
fit PCA-kNN and freeze scores
open labels for post-hoc evaluation
```

默认输出目录（不同 variant/seed 自动隔离）：

```text
experiments/directed_route_hypergraph/outputs/qa/real_seed20260827/
├── model.pt
├── calibration/index.npz
├── calibration/graphs/*.pt
├── test/index.npz
├── test/graphs/*.pt
├── detector.npz
├── scores.npz
└── evaluation.json
```

## 必要 controls

单个 real run 不能证明 hypergraph 有效。正式结论至少需要在相同 source split、seed、训练预算和 node-only detector 下比较：

```text
first-order GCN       普通 pairwise graph 基线
no-message            只保留节点自身状态
endpoint-rewire       保持 layer/head/role/lag/degree/row mass，破坏真实 source
weight-shuffle        保持 endpoints 与 row mass，破坏 weight-endpoint 配对
layer-head-average    构图前平均 layer/head
role-merged           合并 prompt 与 response source sets
one-slot-per-role     检验两个 learned slots 是否必要
retained-only loss    去掉 self/unresolved 守恒目标
position-only/no-pos  排除位置捷径
```

应使用 paired source bootstrap 报告 `real - control` 的 AUROC/AUPRC 区间。若 endpoint rewiring、weight shuffling 或 no-message 不降低结果，就不能把增益归因于真实 hypergraph topology。

## Attention-only 的解释边界

当前缓存不含每个 head 的 value/output contribution、FFN output 和 residual-stream change。真实 head message 更接近

\[
a_{t,s}^{l,h}W_O^{l,h}W_V^{l,h}h_{s,l-1},
\]

而本方法只能观察其中的 \(a_{t,s}^{l,h}\)。因此：

- 可以声称学习了 attention routing representation；
- 可以检验 prompt/response route、端点和权重配对是否具有检测信号；
- 不能把 attention edge 直接称为语义贡献或 functional information flow；
- 不能仅凭 head attention 较大断言该 head 因果决定了输出；
- 若要研究 prompt evidence 与 FFN parametric memory 的竞争，必须额外缓存 OV outputs、attention/FFN 前后 residual，或执行 activation intervention。

这个限制与 [Attention is not Explanation](https://aclanthology.org/N19-1357/)、[Information Flow Routes](https://aclanthology.org/2024.emnlp-main.965/) 和 [ReDeEP](https://arxiv.org/abs/2410.11414) 的结论一致。prompt/response 分流受到 [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/) 启发；保留 head identity 和必要干预受到 [Retrieval Head](https://arxiv.org/abs/2404.15574) 支持；方法主张必须通过结构必要性、干扰不变性和因果可编辑性三重验证，这一标准来自 [Grounding latent algorithm routing](https://arxiv.org/abs/2607.24471)。
