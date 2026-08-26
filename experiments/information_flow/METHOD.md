# 从 Information Flow 到 token-level 幻觉检测：可证伪的验证方案

## 1. 论文真正研究了什么

**Information Flow Reveals When to Trust Language Models** 研究短答案 RAG 中的可靠性。文章的关键判断不是“模型看了多少 context”，而是：

> 最终输出真正接收了哪些 context token 的信息；这些内部贡献是否与外部相关性一致；贡献是否集中在少量关键证据上。

官方实现对每个 Transformer layer 手动重放 attention。对 target token `i` 和 source token `j`，先计算各 head 的 value 经 `o_proj` 后对 `i` 的 source-specific 输出，并在 `i=j` 时加入 residual。令完整 attention-block 输出为 `y_i`，单个 source 的输出为 `c_ij`，代码用曼哈顿距离定义：

\[
q_{ij}^{(l)}=
\max\left(
\lVert y_i\rVert_1-
\lVert y_i-c_{ij}\rVert_1,
0
\right),
\]

然后在 source 维归一化，得到一层的 token-to-token contribution matrix：

\[
C_l[i,j]=
\frac{q_{ij}^{(l)}}{\sum_k q_{ik}^{(l)}}.
\]

层间总贡献通过有序矩阵乘积得到：

\[
F=C_LC_{L-1}\cdots C_1.
\]

最后一行描述每个输入 token 通过所有有效路径对最终输入位置的贡献。代码还从后向前贪心展开 principal information-flow route，得到 token 的 emergence order。

文章据此构造两类可靠性依据：

1. **simulatability**：模型内部 contribution/routing 排名与外部 reranker/SHAP relevance 排名的 RBO 一致性；
2. **concentration**：内部 contribution/path layout 相对均匀分布的集中程度。

必须准确说明：贡献提取本身不需要正确性标签，但官方最终 calibrator 使用 train/validation correctness labels 训练 XGBoost，并用 Optuna 选择超参数。因此论文不是严格的 zero-label detector，而是 **白盒信息流表征 + 监督校准**。

## 2. 与其他研究如何拼接

这篇工作的价值和以下结果是一致的。

### ALTI / contextual mixing

raw attention weight 只表示路由系数，不能代表 source 对 residual stream 的实际贡献。value、`W_O`、residual 和 layer normalization 都会改变贡献。因此文章使用 source-specific block output，而不是直接做 attention rollout。

### Information Flow Routes

单层 attribution 只是快照；预测相关信息应当表示为跨层有向路径。最终可靠性更可能取决于完整 route，而不是某个 head 的孤立统计量。

### Reasoning Fails Where Step Flow Breaks

错误推理可能表现为浅层 lock-in 或深层 context decay。关键是信息是否持续跨阶段运输，而不是 attention entropy 单独升高或降低。

### Grounding latent algorithm routing

受控实验支持中间层逐渐形成 episode-conditioned route state。attention 更适合被解释为这种内部路线的执行痕迹；有序层传播应比层平均更接近真实计算过程。

### 当前项目的 GCN 结果

无标签 endpoint prediction 训练的一阶 GCN 已得到有用节点表征：PCA-kNN AUROC 为 0.6982，监督 linear readability 为 0.7865，明显超过位置基线。这说明“把图运输结构压进 token node embedding”是可行的；但该 GCN 在传播前平均了全部 layer/head，因此仍未检验 progressive all-layer flow。

## 3. 当前数据能做什么，不能做什么

当前 formal attention cache 保存：

```text
retained off-diagonal attention edges
exact self diagonal
unresolved censored mass
layer/head/source/target identity
```

没有：

```text
V vectors
W_O
hidden states
attention gradients
prompt-query rows
外部 relevance layouts
```

因此本实验不能复现论文的 `c_ij`，也不能计算其 simulatability。我们只能验证一个更弱但清楚的问题：

> raw attention 所定义的 ordered transport，是否已经包含 progressive information-flow 信号？

若该 proxy 失败，不能据此否定原论文；更可能说明 value/hidden-state contribution 是必要数据。

## 4. Layerwise Flow Sketch

### 4.1 每条样本一张 typed token graph

节点是 prompt 与 response token。每条 retained edge 保留：

\[
(s,t,l,h,A_{t,s}^{l,h}).
\]

prompt-query rows没有保存，因此 prompt node 在传播中保持固定；response node 按 causal attention 更新。

### 4.2 共享 source basis

不同样本 token 数不同，不能直接把每张图的 one-hot contribution vector 当作跨样本表征。我们先给所有图使用相同的低维 basis：

\[
B_i=[\text{role}_i,\text{Fourier}(\text{absolute position}_i),
\text{Fourier}(\text{response position}_i)].
\]

它不包含幻觉标签，也不预先定义 grounding、entropy 或 concentration。它只是对完整 source identity 的固定低维 sketch。

### 4.3 attention-only layer operator

对每个 layer/head，response target 从 source 接收：

\[
M_{t,h}^{(l)}=
\sum_{s<t}A_{t,s}^{l,h}Z_s^{(l)}
+A_{t,t}^{l,h}Z_t^{(l)}.
\]

未保存的 mass 只知道低于 attention floor。默认采用保守处理：

\[
M_{t,h}^{(l)}\leftarrow
M_{t,h}^{(l)}+U_{t,h}^{(l)}Z_t^{(l)},
\]

即将 unresolved mass 留在当前 token；另提供 `renormalize` 控制。

没有 `W_OV` 时不能合理合并 heads 的内容方向，因此使用 head mean 保持行尺度：

\[
M_t^{(l)}=\frac1H\sum_hM_{t,h}^{(l)}.
\]

加入 Transformer residual proxy：

\[
Z_t^{(l+1)}=
\frac{\alpha Z_t^{(l)}+M_t^{(l)}}{\alpha+1},
\qquad \alpha=1.
\]

最终：

\[
Z^{(L)}=T_L\cdots T_1B.
\]

这是完整 contribution product 的低维 sketch：没有显式构造 `N×N` dense product，却保留了所有层的有序多跳路径。

### 4.4 输出节点表征

```text
full_final      有序全部层后的最终状态
full_trace      每一层状态按顺序拼接
reverse_final   反向层序后的最终状态
reverse_trace   反向层序的完整轨迹
last_layer      只应用最后一层
layer_mean      各层独立作用于同一初始 basis 后的单步 ensemble
identity_final  不使用图的初始 basis
identity_trace  将初始 basis 重复 L 次，维度与 full_trace 完全一致
```

`full_trace` 是主要 token node embedding；所有 detector 只读取这些固定表征，不再读取边。

`identity_trace` 与 `full_trace` 具有相同维度，避免把“图运输增益”和“输入维数更高”混在一起。`layer_mean` 只是单步 layer ensemble，不被表述为严格的等深静态 rollout；真正的层序证据主要来自 ordered 与 reverse 的同深比较。

## 5. 同一套评估

复用当前 GCN 的 node-only readers：

```text
PCA-kNN
Isolation Forest
LOF
Autoencoder
Deep SVDD
source-disjoint linear probe
source-disjoint MLP probe
absolute/relative position baseline
source-group bootstrap
```

无监督分数在标签打开前保存。监督 probe 只作为 representation ceiling，不属于主方法。

核心配对比较：

```text
full_trace  - reverse_trace    同维、同深：层序是否重要
full_final  - reverse_final    同维、同深：最终 composition 是否依赖层序
full_final  - last_layer       全部层是否优于最后一层
full_final  - layer_mean       有序多层结果是否优于单步 layer ensemble
full_trace  - full_final       完整轨迹是否优于只保留最终状态
full_trace  - identity_trace   同维：图运输是否超过位置 basis
full_final  - identity_final   同维：最终流状态是否超过位置 basis
```

## 6. 判定与停止条件

保留该方向至少需要满足一项：

1. `full_trace` 或 `full_final` 的无监督结果接近或超过现有 GCN；
2. linear readability 明显超过 position baseline；
3. ordered flow 在 paired source bootstrap 中稳定优于 reverse order；
4. all-layer flow 稳定优于 last-layer；
5. flow 在同维比较中稳定优于 identity control。

若 ordered flow 不优于这些控制，则停止 attention-only rollout。下一步应采集 `V/W_O/hidden state`，复现论文真正的 contribution operator，而不是继续加入 entropy、degree、closure 等手工特征。

## 7. 后续方法如何从验证结果生长

本目录先做确定性的 operator audit，不先训练新的大网络。这样可以把问题拆清楚：

- ordered flow 有信号：再学习 layer/head mixing，或用 normal-only / few-shot calibrator读取 flow trajectory；
- ordered flow 只在监督 probe 中有信号：表示有效，one-class detector需要调整；
- ordered flow 与 reverse、identity 相同：raw attention 不足，应采集 value、`W_O` 和 hidden state；
- 真 contribution 可用后：再实现论文的 relevance-layout alignment，并比较 attention route、value-aware contribution 和 hidden-state route。

## 8. 代码入口

```bash
bash experiments/information_flow/run_qa.sh
```

主要实现：

```text
basis.py       共享 source sketch
transport.py   sparse ordered flow composition
extract.py     按样本保存图与节点轨迹
evaluate.py    相同 reader、位置基线和 paired controls
```
