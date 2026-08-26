# Method: structural attention routing fingerprints

## 1. 研究定位

这不是最终的幻觉机制方法，而是一个清晰的图表征基线。它回答：在不训练 GNN、不定义多个手工异常分数的情况下，如何把 layer/head-aware attention 图直接变成逐 token 节点特征。

## 2. Multiplex token graph

一条样本是一张图。token 是节点，每条 retained attention incidence 保留：

\[
(s,t,l,h,A^{l,h}_{t,s}).
\]

未保存的 sparse entries 不当作零，而由每个 `(target, layer, head)` 的 unresolved mass 表示。

## 3. 固定结构基

对 response token `t` 和 channel `(l,h)`，将 prompt source 位置分布投影到固定 cosine basis：

\[
q^P_{t,l,h,k}=\sum_{s\in P}A^{l,h}_{t,s}\cos(\pi k x_s).
\]

将 response-history edge 按归一化 lag 投影：

\[
q^R_{t,l,h,k}=\sum_{j<t}A^{l,h}_{t,j}\cos(\pi k\,\mathrm{lag}_{j,t}).
\]

一跳邻居 provenance 使用真实二跳结构：

\[
q^I_{t,l,h,k}=\sum_{j<t}A^{l,h}_{t,j}q^P_{j,l-1,h,k}.
\]

它表示当前 token 通过历史 response source 继承了怎样的 prompt source distribution。历史实验中，一跳 prompt provenance 明显强于直接 prompt ratio 和更长路径，因此第一版只保留一跳。

每个 channel 的结构向量为：

\[
f_{t,l,h}=[q^P,q^R,q^I,d_{t,l,h},u_{t,l,h}].
\]

heads 使用一个固定、可复现的正交投影压缩到若干 modes；所有层按顺序保留。最终节点特征是：

\[
x_t=\operatorname{vec}_{l,m}(f_{t,l,m}).
\]

默认 32 层、6 个 source basis、8 个 head modes 时，节点维度为：

\[
32\times8\times(3\times6+2)=5120.
\]

## 4. 无监督节点检测

训练数据只用于学习正常节点特征子空间：

1. 回归掉 position、response length、unresolved mass 和 incoming-edge count；
2. 用 median/MAD 做 robust standardization；
3. 在 source-disjoint fit groups 上拟合 PCA；
4. 在独立 calibration source groups 上建立残差能量经验分布。

测试节点的主分数只有一个：

\[
E_t=\frac1D\|z_t-UU^\top z_t\|_2^2.
\]

Detector 不读取图边，不预测下一个 token，不做 masked reconstruction，也不联合多个命名残差。

## 5. 必须回答的实验问题

- 该结构节点特征是否超过 position baseline？
- 是否超过只使用直接 prompt/history mass 的低维特征？
- one-hop inherited prompt block 是否提供增量？
- 保留 layer/head 结构是否超过 layer/head mean？
- real endpoints 是否超过 matched endpoint rewiring？
- 同一节点特征交给 PCA residual、kNN、LOF 时结果是否一致？

只有这些问题得到正结果后，节点表示才值得作为后续机制创新的基础。
