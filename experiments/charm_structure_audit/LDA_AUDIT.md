# LDA究竟利用什么：head身份、相关性、连续状态、prompt读取

基于 main @2ea741f。保留原图、CHARM、混合模型及所有结果。
本轮只做监督诊断和原分数分解，不训练神经网络，不提取新LLM状态，不宣称无监督成绩。

## 1. 先复算已完成的LDA：分数持续还是当前变化？

```bash
python -u -m experiments.charm_structure_audit.main --mode lda_audit --lda-stage scores --bootstrap 2000
```

只读取 `audit_mixture/test_tokens.csv.gz` 中已冻结的 `supervised_lda`，
以及已锁定 `charm_in/test/cluster_audit/pairs.json`。输出 `audit_lda_scores/`。
该步骤不再拟合，不读取原节点或边。不修改已有评分或阈值。

对每个回答，使用之前10个词（开头不足10个时用实际前缀）：

    past(t) = mean(z(t-10), ..., z(t-1))
    increment(t) = z(t) - past(t)
    z(t) = past(t) + increment(t)

首词没有历史，记缺测。三组总体比较用完全相同的可观测token；固定配对也使用双方共同可观测位置。
不按金标边界重置，不把当前词混进历史，不跨回答取均值。`--lda-window`显式修改观察长度。
这里是“当前分数可以被过去分数预测多少”，不能直接证明逐点LDA内部在做时间聚合。

输出 `metrics.csv`、`matched_pairs.csv`、`matched_summary.csv`、`conditional_ranking.csv`、
`history_vs_current.png/.svg`。所有四种标签转移方向都保留；特别检查“之前错误、当前正常”的退出词。
既往错误长度只用于诊断条件，不是任何评分的输入。没有独立报警阈值，因此scores模式只报告排序。
自动打包 `lda_scores_review.tar.gz`。

## 2. 闭式训练并拆开LDA

```bash
python -u -m experiments.charm_structure_audit.main --mode lda_audit --bootstrap 2000
```

默认路径仍是 `outputs/charm_structure_audit_qa/QA/seed_0`，prepared路径不变。
复用 `node_only/training.json` 的FIT/calibration/TEST来源划分。
所有FIT节点参与监督均值/协方差估计，不用原CHARM分数拟合、不选测试头、不重新配正常窗口。
输出 `audit_lda_nodes/`。没有读取edge_attr/edge_index/embedding，没有运行消息网络。

令每个节点x由所有LLM层/head的对角线构成。训练计算：

    mu_y = FIT中标签y的均值
    Sigma = sum_i (x_i-mu_yi)(x_i-mu_yi)^T / N + ridge * diag(FIT_scale^2)
    w = solve(Sigma, mu_1-mu_0)
    b = log(prior_1/prior_0) - 0.5*(mu_1+mu_0)^T w
    z(x) = x^T w + b

默认ridge=1e-3，与mixture中监督LDA一致。常数通道scale设为1，只为同原协议。
`baseline_replay.json`与已存`supervised_lda_diagnostic.npz`对比TEST实际logit；
相同输入与ridge下应接近数值误差。只显示不存在参考，不假称回放过。
LDA类条件高斯是假设；本轮只用其矩估计和线性规则，不以有效分类证明高斯成立。

### A. 哪类head统计起作用？

| 对照 | 唯一目的 |
|---|---|
| raw_mean | 直接按类均值差加权，不校正通道波动 |
| diagonal | 只使用每个head自己的类内方差，不使用跨通道协方差 |
| layer_block | 保留同一LLM层内head协方差，去掉跨层协方差 |
| full | 使用完整类内协方差 |
| layer_mean | 每层先对头平均，只保留L维，检验整体自关注量是否足够 |
| head_contrast | 原x减该token在该层的头均值，保留head身份，检验相对强弱 |
| unordered_heads | 每层逐token按数值排序后重新拟合，保留数值分布/稀疏程度，丢掉head编号 |

前四组使用同一批FIT原始输入、相同均值和正则化协方差，只改变求解所用协方差结构。
后三组明确改变输入表示，使用同样的FIT流程重新计算均值与协方差，没有借用原w声称重训。
full优于diagonal才支持跨通道协方差有预测价值；这种价值可能是消除共同噪声，
不能自动命名为语义head协作。固定ridge不是最优超参证明，极端病态协方差也应看正则敏感性。

### B. 直接读出每个head为什么得到这个系数

由 Sigma*w=Delta，逐行精确得到：

    w_i = Delta_i/Sigma_ii - sum_(j != i) Sigma_ij*w_j/Sigma_ii

代码将第二项再拆为“同层”和“跨层”：

    w = direct + same_layer_compensation + cross_layer_compensation

这不是另训探针，也不是近似积分。`head_rules.csv`列出每个head的两类均值、方差、三项与最终系数。
`covariance_links.csv.gz`按FIT固定每个head最大的3个补偿来源；这些是统计项，不是attention图中的边。
例如x0=B+Y+noise、x1=B+noise，x1即使没有类别均值差，也可获得负系数，用来抵消共同波动B。
负系数不能被命名为原LLM的“纠错head”。

在已有正常/错误配对上：

    contribution_i = w_i * (x_error_i-x_normal_i)
    sum_i contribution_i = z_error-z_normal

所有head的数值、系数和贡献都保存，不只展示有利的通道；`channel_summary.csv`按来源等权，
前后半段分别汇总并给探索性区间，未做多重校正。`pair_head_contributions.csv.gz`保留每对每head原值。
另有两种精确分解的分数：
- part_layer_common + part_head_contrast = 中心化full分数；
- part_direct + part_same_layer + part_cross_layer = 中心化full分数。
分数可相加，AUROC不可相加；分量是当前模型的算式，不是独立训练模块的因果百分比。

### C. 连续历史/位置/词面是否足以解释区分？

固定条件组：回答位置五等分 × 首token粗类别(word/number/punctuation/empty) ×
过去连续错误长度的组{0},{1},{2,3},{4,...,7},{8,...,15},{16+}。
这是token自身的粗词面类别，不是词性或事实实体标注；位置用最终回答长度，仅作诊断控制。

FIT每组至少有3个正常与3个错误才进入以下对照：
- matched_support：仅在共同可用样本上原样拟合；
- balanced_context：同样样本，各组每类权重= min(n0,n1)/n_y。

加权后两类具有完全相同的经验条件组分布，因此不能单靠这些条件组的类别频率区分。
两种对照的样本集合一致，避免将“丢掉了不平衡样本”的效果误认成重权作用。
既往标签只用于训练权重/诊断分层，评分仍只读取当前x。仍是有标签诊断；
它不消除非线性历史状态或语义混杂，也不证明生成因果关系。
没有共同支持时明确报告不可用，不自动放宽。`fit_context_balance.csv`完整报告分母。
测试输出按之前正常/之前错误分开，并在同回答、同条件组内计算AUROC和共同覆盖。

## 3. prompt注意力是不是足够解释这个判别？

```bash
python -u -m experiments.charm_structure_audit.main --mode lda_audit --lda-prompt --bootstrap 2000
```

输出独立的 `audit_lda_prompt/`，不会覆盖node-only审计。
每个回答额外读取原prepared图一次，逐层逐头求该query保留的prompt边权重总和R。
绝不从1-self推断prompt：剩余权重还可流向历史。
`retained_mass_mean`和`prompt_mass_mean`保存在逐词表供核查；不为无保留边伪造完整attention。
这些量已受原缓存/图阈值截断，不是全量attention；prompt也不是经核验的适用证据。

比较：
1. prompt_only：相同来源划分上，以R闭式拟合LDA；
2. self_after_prompt_regression：FIT无标签岭回归X≈R*B+a，减去可线性预测部分后拟合LDA；
3. self_plus_prompt：使用[X,R]联合拟合LDA；
4. 现有self-LDA分数=prompt可线性预测部分+残差，精确分解，同时报告留出预测R²。

第二项只能排除FIT线性prompt总量解释，不排除非线性关系，也不能叫作精确条件互信息。
第三项若改善，支持实际prompt观测提供预测增量；不同模型的估计误差/正则也应保留。
如果prompt_only弱而self残差仍强，就不能继续把高分简单归因于“prompt读取总量较少”。
本轮不判断哪些prompt节点语义适用，不做原LLM QKV干预。

## 输出与验收

- `metrics.csv`：全流、首错、续错、上一词正常/错误的AUROC/AP及独立校准后的召回/FPR。
- `within_summary.csv`：同回答评价，按来源汇总。
- `matched_summary.csv`、`paired_deltas.csv`：相同固定结构配对上的成绩、相对full差异和来源区间。
- `head_rules.csv`、`channel_summary.csv`、`covariance_links.csv.gz`：具体的权重规则与相关性补偿来源。
- `test_scores.csv.gz`：所有token的原文、身份、标签、位置和每个固定评分。
- `figures/`、`history_vs_current.png/.svg`：协方差/头身份比较及历史与当前增量对照。
- `lda_review.tar.gz`：自动导出报告/表格/图片，排除parameters中的大协方差，不含任何checkpoint。

每种分数在独立calibration正常文本token上校准5%FPR，严格score>threshold。
测试FPR不保证恰好5%；条件子集AP受其特殊类比例影响，不能称为准确率。
观察性高分/边界现象与原LLM因果作用分开。已有test反复审计，全部为探索证据。
若用本轮带标签权重设计最后评分，须称监督或蒸馏；无监督先验还需独立来源的无标签获取方式。

再次统计已完成结果（不再拟合）：

```bash
python -u -m experiments.charm_structure_audit.main --mode lda_audit --lda-stage report
# prompt结果加--lda-prompt，或明确--output。
```

四个模块只有本轮所需职责：lda_math.py公式；lda_data.py读取；lda_audit.py实验顺序；lda_report.py统计绘图。
不改既有模块算法、不加神经网络、自动挑头、目标重建或通用兼容框架。

数学出处：scikit-learn官方LDA推导 https://scikit-learn.org/stable/modules/lda_qda.html 。
协方差分项来自Sigma*w=Delta的逐行恒等变换；时间分项是score减过去均值的恒等式。
