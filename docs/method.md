# Method: self-supervised attention-graph representation learning

## Training-free provenance pattern discovery

在训练图模型之前，仓库提供一个只检验拓扑机制的探索协议。对每个 response
token，从最后一层开始沿 attention source 逐层反向传播单位质量。Prompt 节点
和 cache 未观测质量是吸收状态，response 节点通过其精确 source endpoint 继续
向前层传播。主曲线在每行已观测 attention 内条件归一化，避免把 cache 缺失量
偷偷当作聚类特征；原始未观测质量使用独立传播副本计算控制曲线。第 $d$ 个反向
深度的两个候选视图为

\[
A_t(d)=\text{cumulative mass absorbed by prompt},
\]

\[
C_t(d)=\sum_j\left(\frac{\pi_{t,j}^{(d)}}
{\sum_k\pi_{t,k}^{(d)}}\right)^2.
\]

$A_t$ 表示 prompt grounding 的传播深度，$C_t$ 表示仍存活的 response
ancestry 是单链集中还是多源分散。默认主实验的节点表示只有

\[
z_t=[A_t(d_1),\ldots,A_t(d_m)].
\]

若单独研究 $C_t$，必须作为另一次实验运行，不允许与 $A_t$ 拼接。未观测质量
曲线单独用于检查 cache censoring，不拼入 $z_t$。Train token 上
使用 robust scaling 和 BIC-selected diagonal GMM 发现结构模式，不使用标签和
反向传播。模式、t-SNE 坐标及代表性 ego graph 冻结后，才读取 test token
labels 计算各模式的 hallucination enrichment，并比较正确/幻觉响应图的模式占比
与相邻 token 模式转移。

## MART non-GNN baseline

MART is the mechanism-aligned non-GNN baseline. It computes per-token features
directly from canonical CSR attention: channel summaries of retained mass,
retained prompt fraction, entropy over retained sources plus diagonal and one
censored OTHER bucket, anchor `q(1-H)`, diagonal mass and censored OTHER mass;
signed late-minus-early layer drift; and a strictly causal EMA innovation.
It then uses train-only robust calibration within relative-position bins, PCA
whitening, and kNN distance; position selects the calibration bin but is excluded
from the scored vector. This tests whether the useful routing findings
already suffice without learned message passing. Statistics motivate the
representation but do not prescribe an error direction.

A graph encoder is scientifically necessary only when it improves on MART and
the same encoder with no messages, and when source-shuffling edges degrades it.

## 1. 研究对象

对一条 prompt + response 序列建立有向图

\[
G=(V,E,\mathcal A).
\]

每个 token 是一个节点。只保留 response query 的严格因果连接 `source < target`：

- RP: prompt \(\rightarrow\) response；
- RR: previous response \(\rightarrow\) response。

对 token pair \((j,i)\)，不是先把所有 layer/head 平均成一个标量，而是保存 sparse trace

\[
\mathcal A_{ji}=\{(c,a^{c}_{ij})\},\qquad c=(l,h).
\]

这使 GNN 自己学习哪些 layer/head channel、哪些 relation、哪些 source neighborhood 重要。

## 2. 节点与边的 learned encoding

节点的原始 attention 属性是所有 channel 的 self-attention diagonal：

\[
x_i=[A^{1}_{ii},\ldots,A^{LH}_{ii}].
\]

它经过 learnable channel basis 聚合为节点初始表示，并加入位置与 prompt/response role context。

每条边的 sparse channel trace 通过两个可学习 basis 聚合：一个编码 attention value，一个编码 channel presence；再加入 edge magnitude 与 RP/RR relation embedding，得到 \(e_{ji}\in\mathbb R^d\)。缺失 channel 保持为“未观察/低于 cache floor”，不会在输入前伪造成完整 dense vector。

## 3. CHARM-style message passing

第 \(k\) 层对每条边计算

\[
m_{j\to i}^{(k)}=\phi_m([h_j^{(k)}\Vert e_{ji}]),
\]

对同一 target 做 mean aggregation：

\[
\bar m_i^{(k)}=\frac{1}{|N(i)|}\sum_{j\in N(i)}m_{j\to i}^{(k)},
\]

再更新：

\[
h_i^{(k+1)}=\operatorname{LN}\left(h_i^{(k)}+
\phi_u([h_i^{(k)}\Vert\bar m_i^{(k)}])\right).
\]

`phi_m` 和 `phi_u` 都是 learnable MLP。和监督 CHARM 不同，它们不由 hallucination BCE 训练，而由下面的 graph reconstruction 训练。

## 4. Target-masked self-supervision

训练随机选择一组 response target token。对每个被选 token \(i\)：

- mask 其 node attention diagonal；
- mask 所有 incoming RP/RR pair edges；
- 可额外 drop 一部分 layer/head channels。

模型从剩余 causal graph 重建被遮蔽的四类内容：

\[
\mathcal L=
\lambda_s\mathcal L_{support}+
\lambda_w\mathcal L_{weight}+
\lambda_d\mathcal L_{distribution}+
\lambda_n\mathcal L_{node}.
\]

`support` 判断真实 pair 与同 target/relation 的 absent causal pair；`weight` 重建 retained channel attention；`distribution` 重建一行 retained attention 的相对分布并加入 OTHER bucket 表示 cache-censored/未选 mass；`node` 重建该 token 的 layer/head diagonal。

这四个目标都来自 attention graph 自身，无需 hallucination label。

## 5. Learned node embedding

训练后，在完整未 mask 图上得到

\[
H=f_\theta(G),\qquad H\in\mathbb R^{N\times d}.
\]

研究可视化使用 response token 的 \(h_i\)，而不是手工 32D/33D feature vector。

## 6. Unsupervised anomaly score

测试时对每个 response token 做 leave-one-target-out：隐藏该 token 的 node attribute 与全部 incoming edges，计算六个可解释 residual：

\[
r_i=[r^{sup}_{RP},r^{sup}_{RR},r^{w}_{RP},r^{w}_{RR},r^{dist},r^{node}].
\]

只用 canonical train split 的独立 calibration source groups 拟合每维 median/MAD：

\[
z_{ik}=\frac{r_{ik}-\operatorname{median}_k}{\operatorname{MAD}_k}.
\]

最终分数使用各 residual 的 one-sided positive deviation 等权平均：

\[
s_i=\frac1K\sum_k\max(z_{ik},0).
\]

因此没有“异常一定是小簇”的假设，也不需要在 test 上拟合 GMM / Student-t mixture。

## 7. Statistics 的角色

`attention_graph/statistics.py` 对传入 split 的**全部样本**计算 retained mass、prompt/history routing、entropy/concentration、density、lag、channel coverage 等诊断量。它们：

- 不作为 GNN node feature；
- 不选择或加权 GNN loss；
- 只用于验证机制假设、报告 feature-wise AUROC、和 handcrafted baseline。

旧实验曾进一步区分 passage/question/answer。当前 canonical archive只保证 prompt/response boundary，因此本仓库不会把整个 prompt 擅自解释成 passage 或 question；若后续需要复现该分析，应新增经过 tokenizer 对齐验证的 segment sidecar，而不是从 token index 猜测。

## 8. 正式实验协议

1. canonical train：source-group train / validation / calibration，全部 label-blind；
2. validation：只以 reconstruction loss early-stop；
3. calibration：只拟合 residual median/MAD；
4. canonical test：冻结模型与 calibration，生成 embedding 和 anomaly score；
5. 最后 `evaluate` 才读取 `labels.jsonl`。

任何使用 test embedding 拟合异常模型的 transductive protocol 都不是默认正式实验。
