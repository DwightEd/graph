# Latent Regime：直接无标签建模“head协同 + span连续”

Structured Compatibility 的自然 TEST 结果已经表明：人工制造“不协调”不能恢复自然幻觉方向。
因此这里不再训练 corruption classifier，也不再做距离异常。

## 对应已有监督发现

监督审计已经稳定显示两件事：

1. physical-head 的有符号协同很重要：head contrast / layer block 明显强于 layer mean；
2. 错误 span 高度连续，但固定 previous label 后当前 token 仍有独立信号。

这里把监督 LDA 的形式直接改成无标签 switching model。

对每个 token、每层保存 32 个 physical head 的：

- query-self attention
- total prompt attention

因此每层是 64 维自然观察，不使用 source role、人工 evidence 标签或 hallucination 标签。

## 两状态 shared-covariance HMM

对自然 TRAIN 序列拟合：

x_t | z_t=k ~ N(mu_k, Sigma)

z_t ~ Markov(A)

Sigma 在两个状态之间共享、但每层单独保留 64×64 协方差。
因此状态 log-likelihood ratio 是一个按层分块的线性判别方向，
与 supervised layer-block LDA 的结构直接对应，但 mu_0/mu_1、Sigma、A 全部由无标签 EM 学习。

transition 使用对称 sticky pseudocount，只表达“状态通常持续”，不指定哪一状态是错误。
EM 完成后仅按 source-balanced TRAIN occupancy 排序：
占用率更低的状态定义为 suspect state。没有用错误 prevalence 或 TEST label 定方向。

所有 TRAIN sufficient statistics 按 source 等权：
一个 source 的全部 token 总权重为 1，长回答不能支配均值、协方差、occupancy 或 transition。

## 三个必须一起看的分数

- iid：同一 HMM emission，但不使用时间转移；
- filtered：因果 forward filter，只使用当前和过去，主方法；
- layer_mean_filtered：先平均 32 个 head，再拟合同样的 HMM。

因此：

filtered - iid = temporal continuity 的增量

filtered - layer_mean_filtered = physical-head structure 的增量

另报 self_jump 和 position 控制。

如果 filtered 只提高 continuation、不提高 previous_gold_0，
说明 span continuity 有用但没有恢复 onset direction。

如果 filtered 与 layer_mean_filtered 没差异，
说明 head 协同没有被无标签 latent state 恢复。

如果较低 occupancy state 本身仍占很大比例，
也必须报告，不能把它叫“异常状态”。

## 一键运行

先检查接口：

python -u -m experiments.unsupervised_token_graph.latent_regime --phase inspect

完整运行：

python -u -m experiments.unsupervised_token_graph.latent_regime \
  --phase all \
  --output outputs/latent_regime_v1 \
  --resume

统一入口：

python -u -m experiments.unsupervised_token_graph.run regime \
  --phase all \
  --output outputs/latent_regime_v1 \
  --resume

不加载 Llama 权重；prepare 只重读已有 attention cache。
fit 是 CPU/NumPy/Scikit-learn，两种模型（full 与 layer_mean）按 task/generator 独立拟合。

## 结果判断

控制台直接打印：
- all_auroc / all_ap
- onset_auroc = previous_gold_0
- continuation_auroc = previous_gold_1

predictions/evaluation.json 还包含 sentence-start onset、high-transition onset、
source bootstrap、macro within-answer、95% 无标签 calibration threshold recall/FPR，
以及 filtered-iid、filtered-layer_mean 的 source-bootstrap 配对增量。

这个实验是一个明确的可证伪点：
如果它仍不能接近监督 LDA，那么“连续性 + 无标签双状态结构”不足以给监督方向定向，
下一步必须引入外部 evidence/applicability 语义，而不能继续做内部状态异常检测。
