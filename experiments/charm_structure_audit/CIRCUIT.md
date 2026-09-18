# 从真实节点模型参数还原判别方向

## 输入与运行

基于 main 5832c99。使用 export_circuit 已生成的 `charm_circuit_inputs.tar.gz`，包含实际 node_only 权重、固定配对的原始 x、分数和文本。没有图边、没有重新训练、没有新提取 attention。标签只用于配对评价。

```bash
python -m experiments.charm_structure_audit.main --mode circuit
```

默认输入为 ROOT/audit_circuit_export/charm_circuit_inputs.tar.gz；输出 ROOT/audit_circuit。ROOT 默认 outputs/charm_structure_audit_qa/QA/seed_0。迁移后的包可以用 `--circuit-input /实际路径.tar.gz`。CPU 即可；`--bootstrap` 控制来源重采样次数。

## 核心数学

原输入投影为 g=ReLU(Px+b)。只对训练后的 P 做奇异值分解，取第一右奇异向量 v，方向符号由最大绝对系数取正固定，没有用真假标签拟合或选方向。

    q(x) = v^T x
    P1 = (P v) v^T
    P = P1 + R

冻结所有其他参数，分别保留 P1、删除 P1。对照始终保留原偏置。另提供保留 R 的样本平均写入（不保留各样本变化）的版本，它是这批配对上的均值消融，不是独立校准模型。

真实配对中，再把错误与正常在这个分量上的差异 d=(P v) v^T(E-N) 单独移除/引入：

    remove = F(ReLU(P E+b-d))
    insert = F(ReLU(P N+b+d))

F 是原来的三轮自身更新及预测头。参照需要已知正常伙伴，成绩只能称为配对干预诊断。
20个随机正交方向的每个token在首层**激活前**的L2改变量与d精确一致。它检验方向效应，不保证ReLU之后的范数、自然性或语义一致。

## 逐头与逐单元贡献

采用 DeepLIFT-style Rescale 差分核算：每个 ReLU 用两端的 ΔReLU/Δpre 作为乘子，在残差分支间相加，沿实际权重反传。Δpre=0时取当前开关，贡献仍为0。

    sum_head (E-N)*multiplier = logit(E)-logit(N)
    sum_projection_unit Δactivation*downstream_multiplier = 同一个差值

逐token核验两个恒等式。此分解依赖网络表示与正常参照，**不同于前一轮沿输入直线数值积分的IG**；不覆盖、混合或冒称复现其逐头数值。算术守恒不是语义因果证明。每个投影单元也做实际双向激活交换；不只看权重范数。

另以 autograd 独立计算真实函数的局部输入梯度，检查它与 v 的对齐程度。P 的SVD依赖参数化，所以不能只看第一奇异值；必须同时看实际保留/删除、原输出排序和函数梯度。

## 输出

- axis_summary.csv / pair_metrics.csv：保留、删除、配对替换、等幅随机对照的配对AUROC、logit差与来源区间。
- head_rules.csv：每个原LLM层/head的v系数、两类输入均值、完整网络差分贡献与有效乘子。
- projection_units.csv / unit_swaps.csv：128个投影单元的权重方向、真实激活、下游作用及实际交换效应。
- projection_weights.csv.gz：全部原投影权重，不隐藏省略项。
- tokens.csv.gz / positions.csv：每个配对词的原分数、q和前后半统计。
- status.json：回放误差、权重能量、函数梯度对齐与排序相关性。两张PNG为方向与主要干预。

主要AUROC：每对内部计算，再同来源配对平均、来源等权；pair_macro_auc另外保存。所有904对齐位置都参与，不限于先前IG通过的894个位置。结果仍只覆盖成功匹配的50对，不是整个RAGTruth测试集。CI重采样来源，不重采样头；单一训练seed无训练方差结论。

## 本次真实数据的结果

50对、40来源、904对齐位置；原分数回放最大误差1.14e-7。实际执行了冻结权重保留/删除、配对方向替换、20种等幅正交控制、128个单元交换。

原配对来源均值AUROC=0.804638；保留首层第一方向=0.804678；删除=0.516331。真实函数梯度与该方向的余弦中位数0.99535，标量q与完整logit的Spearman=0.98509。

只保留第一方向的logit平均绝对误差仍为0.74189；保留正交部分平均写入后为0.17409。说明排序可大幅简化，不代表概率、阈值或整个函数完全等价。

L10H0、L29H25、L16H2在错误中自关注更低，通过负向有效乘子支持更高风险；L8H11、L13H18自关注更高，通过正向乘子支持风险。不是所有head同方向。

第一投影的U85在两侧都开启，错误平均激活降低，而下游作用为负；U93在错误侧平均更强，下游作用为正。两路可以同时推动同一个判别方向；单元没有固有语义名称。

## 结论边界

这个方向从监督训练后的权重抽取，不能称为无监督算法或自然两簇。没有重新训练一个线性分类器，也没有以测试标签选head。测试集已反复探索，泛化需独立来源检验。
一个方向是全部1024通道的加权组合，不是一个head或一个LLM层。保留排序不能证明fact grounding、self lock-in或错误延续的生成原因。原始x已经有上下文信息，当前节点网络没有时间状态。

参考：Shrikumar et al., ICML 2017, Learning Important Features Through Propagating Activation Differences (https://arxiv.org/abs/1704.02685)。这里只借用差分守恒计算，不借论文替自然数据下结论。

## 实际测试

`python -m pytest tests/test_charm_circuit.py tests/test_charm_structure_audit.py -q`
52 passed（11项新测试含参数化、41项原模型测试）。检查实际节点函数、残差/非残差差分守恒、ReLU跨零、rank-one极限、无消息执行、等幅正交构造、来源权重及真实CLI。另实际运行用户本次自然权重和全部导出的配对输入，非合成数据成绩。未运行原LLM或新训练。
