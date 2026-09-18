# 两成分高斯是否解释节点判别方向？

这是**假设检验基线**，不是新的图方法。保留完整 CHARM、所有已有实验及结果。

## 1. 比较的对象

原模型 `h=ReLU(Px+b)` 的 P 为 128×1024 矩阵；此前发现的方向 v 是 P 的第一右奇异向量，长度1024。
本轮比较 v 与 w=Σ⁻¹(μ₁−μ₀) 的方向、在留出输入上的分数排序，不比较 P 与 w 相等。
一个标量线性读出有效，不说明数据只有一维，也不说明原网络只包含P、b。

只有假设 `x|k ~ N(μ_k,Σ)`，且两成分共享Σ时，才有：

    log P(k=1|x)/P(k=0|x)
      = wᵀx + b_g
    w = Σ⁻¹(μ₁−μ₀)
    b_g = log(π₁/π₀) − 0.5 (μ₁+μ₀)ᵀw

高斯共享协方差 => 线性后验；线性可分 => 高斯不成立。均匀分布也可以按线性方向完全分开。
这两成分 k 是隐变量，不预先等同正常/幻觉 y。即使k恢复完美，也可能只分出词类、位置或其他状态。

## 2. 一次运行

在graph根目录、research环境：

```bash
python -u -m experiments.charm_structure_audit.main --mode mixture
```

默认沿用 `outputs/charm_structure_audit_qa/QA/seed_0/node_only/training.json` 的来源划分，
读取 `outputs/charm_structure_audit_qa/data/graphs/<split>/<id>.npz`。
输出 `QA/seed_0/audit_mixture/`，自动生成 `mixture_review.tar.gz`。
CPU NumPy/SciPy，线程上限4。无需8B模型、GPU或新的attention抽取，不训练CHARM。
所有FIT回答的所有response token都进入拟合；不用金标窗口、分数阈值或配对采样。

三个阶段保持明确：

```bash
# 只做无标签拟合与留出评分，连监督checkpoint都不读取。
python -u -m experiments.charm_structure_audit.main --mode mixture --mixture-stage fit

# 从已冻结结果重新生成诊断。没有冻结结果会直接报缺文件，不暗中训练。
python -u -m experiments.charm_structure_audit.main --mode mixture --mixture-stage report
```

默认all先fit再diagnose。全部完成fit后重跑会复用；中断的fit从头再拟合，不是迭代断点续跑。
改变ridge/迭代数/种子时显式使用新的 `--output`；不会改旧结果或缓存。
新拟合只复用无标签来源划分；旧CHARM选模型用过标签，本轮不用其选出的层、向量或分数初始化。

## 3. 按步骤排除混淆

### A. 无标签学习，先冻结

1. 仅加载x、prompt_length和通道布局，不加载gold、span、边数组、embedding或检测分数。
2. FIT中按每通道标准化，保留全部维度，不做PCA截断；常数通道标准差记为1。
3. 拟合一个高斯基线，以及共享协方差的两成分模型，初始化seed=0/1/2；KMeans只用于无标签初始化。
4. 每个种子的所有SELECT/calibration/TEST词先保存密度和成分后验log-odds。
5. 只以SELECT的来源等权平均log密度选一个种子，所有种子都保留。此时写入frozen.json。

ridge固定为标准化坐标下1e-3，最多200次责任概率更新；可用 `--mixture-ridge`、
`--mixture-iterations`、`--mixture-seeds`明确修改，不允许根据TEST标签选参数。
EM未收敛会保留实际迭代数和标记，不能把最优初始化当成已经验证的分布。没有自动强行调整数据来成功拟合。
三起点只是局部最优化稳定性，不代表重新抽取训练来源后的稳定性。

### B. 冻结后检查拟合、真假与方向

1. 两成分在留出来源的密度是否超过单高斯？来源重采样区间只描述密度增益；不把token当独立样本。
2. 12个固定随机方向和拟合出的方向上，用混合CDF检查留出数据；输出经验CDF与均匀分布的差距。
   不给依赖token假设的独立样本p值。小差距不证明整个1024维分布高斯；仅比较K=1和K=2也不证明恰有两个状态。
3. 输入范围是[0,1]，非退化高斯无界，只能是近似模型。输出各通道边际界外概率的平均，不误称为联合界外概率。
4. 最后读取TEST gold，报告两个任意编号成分各自的比例、错误占比、位置与词面组成；报告全流、起点、续错、同答指标。
5. 另用FIT gold计算共享协方差LDA的闭式方向。它是**有监督诊断**，用于区分“均值/协方差方向本身不适用”和“无标签分群没有找到真假”。
6. 读取旧node_only checkpoint的P，比较其v与无标签w、LDA方向。此步骤不会反过来重拟合GMM或选超参。

测试输出component1_odds和component0_odds两种方向，**不选择更高的一个，也不把少数成分称为幻觉**。
高斯负log密度作为单独异常基线；低密度=错误也只是待检验先验，不保证成立。
所有AUROC/AP均在完整自然比例的测试词上计算，未加标签权重、未用配对数据训练、未设测试最优阈值。

## 4. 为什么实现没有丢掉1024维？

为避免每轮对大矩阵重新求逆，使用共享协方差二成分模型的代数恒等式。
令标准化FIT数据为y，T=E[yyᵀ]+λI=LLᵀ，使用可逆坐标z=L⁻¹y；所有1024维都保留。
在每次M步之后，两成分的总均值为0。记p=π₁、d=μ₁,z−μ₀,z，则：

    μ₀,z = −p d
    μ₁,z = (1−p)d
    Σ_z = I − p(1−p) d dᵀ
    Σ_z⁻¹d = d / [1 − p(1−p)dᵀd]

这来自全协方差=组内协方差+组间协方差。**秩一的是两均值的组间协方差，不是输入或组内协方差。**
一次Cholesky与可逆变换后，每轮可用矩阵向量运算更新；与原标准化空间直接的全协方差M步一致。
代码测试逐项对比直接加权均值、完整协方差、线性后验与scipy多元高斯log密度。
责任概率裁到[1e-8,1−1e-8]防数值成分消失。固定对角ridge不是无正则的极大似然，
因此记录真实log密度轨迹与最大责任变化，不承诺每步无正则似然严格增长。

映回原通道：w_raw = diag(scale)⁻¹ L⁻ᵀ w_z。
全部均值、完整协方差、原始单位方向存进raw_parameters_seed.npz。
方向比较同时报告原始单位余弦、标准化单位余弦和留出分数Spearman；余弦受尺度影响。
全局符号由最大绝对原始系数为正固定，与标签、P和测试AUROC无关。

## 5. 结果如何判断

| 结果 | 可以支持的解释 |
|---|---|
| GMM密度改善、方向稳定，但成分与真假无关 | 数据存在其他模式；不能用“两簇”作真假先验 |
| 有标签LDA接近P，但无标签GMM远离P | 简单均值/协方差判别可行；自然密度分解未必与分类方向一致 |
| 无标签方向接近P且留出真假有区分 | 支持当前数据中可恢复判别结构；还要解决无标签定向与跨来源稳定性 |
| P有效，但两种高斯读出都弱 | 检查类条件分布、共同协方差近似、正则或其他有效线性结构，不否定原始信息 |
| GMM未收敛或初始化方向不稳定 | 当前拟合不能作上述肯定/否定结论；先查拟合质量 |

## 6. 输出及验证边界

`density_summary.csv`：留出密度增益与来源区间；`cdf_checks.csv`：投影拟合误差。
`direction_comparison.csv`、`direction_coefficients.csv`：与监督P/LDA的方向及逐通道系数。
`seed_stability.csv`：方向、分数、成分划分的跨初始化一致性。
`test_metrics.csv`、`within_summary.csv`：全流/起点/续错/同答排序。成分0/1只是无语义编号。
`component_counts.csv`、`component_surface.csv`：是否主要分出了位置或词类。
`test_tokens.csv.gz`：全部词及所有固定分数，保留原标签用于审计。
`model_diagnostics.csv`：收敛、成分权重、分离度及界外质量。
报告包不含完整协方差、逐答NPZ或checkpoint；原数组留在远端。

此模块暂不测图增量。图是否额外区分重叠区域，需要与相同节点信息做独立对照，不能借本次节点混合结果作证。
监督模型和无标签拟合共享过来源划分，既有test也已经反复审计；均为探索性，确认需要独立来源。

参考：scikit-learn官方LDA公式 https://scikit-learn.org/stable/modules/lda_qda.html
及共享协方差GaussianMixture https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html 。
