# Latent Head Regime：预注册设计

## 问题

为什么监督 LDA 能从单 token physical-head pattern 中读出错误，而 GMM、
Mahalanobis、autoencoder、人工 corruption compatibility 都弱？

当前最小假设不是“错误破坏内部协调”，而是：
错误可能进入一个仍然协调、但不同于常态的持续 head-routing regime。

## 数学形式

每个 token 的第 l 层观察：

x_t^l = [self_{l,1:H}, prompt_{l,1:H}]

状态 z_t in {0,1}。

发射：

p(x_t | z_t=k)
= product_l N(x_t^l ; mu_{k,l}, Sigma_l)

Sigma_l 对两个状态共享，因此两状态 emission log-ratio 对 x_t 是线性的。
这就是 supervised shared-covariance LDA 的无标签对应物。

时序：

p(z_t | z_{t-1}) = A

A 的对角仅加入对称 sticky pseudocount；没有错误状态专属转移率。
状态 1 在拟合后按较低 source-balanced TRAIN occupancy 定义，不看任何自然标签。

## 训练和校准

官方 TRAIN 按 source 拆 reference/calibration。
reference 拟合标准化、HMM emission 与 transition。
每个 source 的所有 token 在 EM 中总权重相同。

calibration 只做三个自然分数的中位数/IQR和95%分位阈值：
iid、filtered、layer_mean_filtered。
high-transition Q90 也只来自无标签 calibration self_jump。

TEST score 全部冻结后才读取 RAGTruth spans。

## 注册对照

P1: filtered > iid，才说明 span continuity 提供增量。
P2: filtered > layer_mean_filtered，才说明 physical-head coordination 提供增量。
P3: previous_gold_0 必须单独有效；只提高 continuation 不能叫 onset mechanism。
P4: sentence-start / high-transition scope 仍需成立，排除普通边界和大跳变。
P5: suspect-state occupancy 必须如实报告；“较少状态”不自动等于 hallucination。

## 停止条件

如果 filtered 在 ALL / QA 的 onset 仍接近随机，
或者 head_structure_gain 的 source-bootstrap CI 覆盖 0，
则停止所有“内部状态无监督定向”路线。

这时已有结果共同说明：自然 hallucination 不是 generic outlier，也不是 generic broken coordination；
要取得方向性，必须加入 RAG evidence 的语义适用性、condition/value binding 或外部验证信号。
