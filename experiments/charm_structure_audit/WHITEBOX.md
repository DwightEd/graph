# 冻结节点模型的白盒验证：先解释计算，再验证解释

基线提交：c231e20b。研究对象固定为**已经训练好的 node_only**；不改原模型、图、标签、阈值、配对或训练划分。不读 edge_index/edge_attr，不做邻居传播，不新增人工 attention 特征。

## 从线索到问题

已有线索：node_only 已有较高排序能力；完整 CHARM 的 no_reuse 效应较小、最后一轮消息作用较明显。因此先确定**节点内部哪些通道组合制造了分数差**。HHI、prompt 邻居数不能代替这项解释。

本轮不把单层 GNN 当成单层 LLM，不把读入 token 后的检测写成生成前预警，不预设 self-attention 越高越错误，也不从测试集选一个最好 head 报成新检测器。

## 唯一运行入口

在 graph 根目录、research 环境：

```bash
python -u -m experiments.charm_structure_audit.main --mode whitebox
```

默认 root 为 `outputs/charm_structure_audit_qa/QA/seed_0`，prepared 为 `outputs/charm_structure_audit_qa/data`。读取 `node_only/checkpoint.pt` 和 `node_only/test` 预测；复用 `charm_in/test/cluster_audit/pairs.json` 中 cluster 配对。

输出 `audit_whitebox/`。每对完成保存小型 `captures/*.npz`，同配置重跑复用完成配对。只重新统计：

```bash
python -u -m experiments.charm_structure_audit.main --mode whitebox --wb-stage report
```

report 不加载 torch 或 checkpoint。不要在不同输入/模型之间复用输出目录。默认运行所有固定配对，没有按效果筛例。仅回放有配对的回答，报告不声称重算了完整测试集。

## 步骤1：先证明正在解释原模型

读取已保存原始 x 和身份/标签/offset/文本；核对完整回答的原 node_only 分数。最大误差须≤2e-5，否则停止。模型评估模式；无训练、无参数更新。

原 float32 回放通过后，将同一组参数转为 float64 做数值解释，并再次核对配对分数。没有重新拟合修复误差。

## 步骤2：精确展开实际 ReLU 计算路径

每个输入 x 得到原投影、各轮自身更新和最后读出的真实 ReLU 开关。在这组开关不变的区域，网络是：

```
z(x) = beta(x) @ x + intercept(x)
```

`beta` 用实际权重和开关反向连乘；`intercept` 从各层训练偏置正向传播，**不是用 z-beta*x 临时凑出来**。其值与原 logit 逐 token 核对。ReLU 恰好为0时使用PyTorch的零导数约定。

另保存预测头所有单元的 `w_u * activation_u`，加末层偏置须重建 logit。错误减正常时，末层偏置消掉。这是精确数值分解；单元编号没有先验语义。

局部系数不是跨输入通用系数。正常/错误激活不同开关时，直接用一侧梯度乘输入差会产生误差；输出两侧局部预测与实际差，便于检查。

## 步骤3：把正常→错误的完整 logit 差分到每个输入通道

对固定窗口同offset的正常向量 N、错误向量 E：

```
A_c = (E_c-N_c) * integral_0^1 beta_c(N + alpha*(E-N)) d alpha
```

这是正常参照上的积分梯度；沿途使用真实ReLU开关，所有LLM层/head原坐标保留。所有 A_c 的和应接近 z(E)-z(N)。

采用Gauss-Legendre积分，从16/32个点开始逐次加倍，默认上限512（`--wb-max-points`）。**同时**检查：
1. 完备性残差：abs(sum(A)-真实gap) ≤ 1e-4 + .005*abs(gap)。
2. 两次积分的逐通道分配变化L1 ≤ 1e-4 + .005*max(sum(abs(A)),abs(gap))。

第二项避免正负误差相消，让总和正确却分错头。有限数值检查仍不是解析收敛证明。上限处未通过的token明确记录，保留原值，排除其通道汇总/归因选点主分析；**不把残差重新分给头来制造精确**。全部逐层交换仍可独立计算。

直线路径可能经过非自然向量；贡献依赖正常参照，不等同于生成因果解释。没有从a_tt反演Q/K/V或证据语义。

## 步骤4：逐层做2×2输入差异实验

逐个原LLM层L，其余网络完全不变：

```
zNN = f(N_L, N_other)
zEN = f(E_L, N_other)  # 只引入这一层的配对差异
zNE = f(N_L, E_other)  # 从错误侧移除这一层的差异
zEE = f(E_L, E_other)
```

分别记录 removal=zEE-zNE、insertion=zEN-zNN、interaction=removal-insertion。交互说明作用依赖其他层背景；不是模型自动理解了head语义。

每对再算原排序、移除后排序、只引入该层差异后的排序及原阈值两侧信息。对照仍然使用已知正常伙伴，**“only layer contrast”不是独立训练/可部署的单层LLM特征检测器**。单层充分性作为新的分类任务需要单独source-disjoint重训，当前不声称已完成。

## 步骤5：归因选的通道是否真比随机通道更影响模型？

每个配对位置选 |A_c| 最大的固定8个通道（`--wb-budget`），做相同的移除/引入测试。排序使用有符号贡献的绝对值，保留正负；删除负贡献可能提高错误分数，不能统一要求下降。

3次随机对照（`--wb-random`）匹配每个LLM层的选中数量，以及层内 |E-N| 的四分位幅度组。报告有符号和绝对影响差、输入改变L1、与所选集合的重叠，以及可交换数量。候选不足时随机可等于原集合；此时不解释成“归因无效”。粗幅度匹配不保证范数完全相等。

这是局部解释验证，通道由当前输入的归因选择，并非独立发现的新检测规则。默认预算、误差容差和方向固定，不因test效果改变。全部通道贡献导出，不只展示最好的头。

## 怎样阅读结果

| 文件 | 回答的问题 |
|---|---|
| `status.json` | 完成多少对，数值积分通过多少位置，最大误差多少 |
| `tokens.csv.gz` | 同位置原文、高低分/TP-FN、真实logit差、局部预测、开关变化、积分误差 |
| `channel_contributions.csv.gz` | 全L/H的有符号输入贡献；全部、前后半、高低分组分开 |
| `intervention_summary.csv` | 每个原LLM层的2×2干预、配对排序、来源区间 |
| `token_interventions.csv.gz` | 每个位置的干预结果、选点/随机实际改动；不只保留总体均值 |
| `random_comparison.csv` | 归因通道与随机通道的影响差，共同可辨别分母 |
| `readout_contributions.csv.gz` | 每个读出单元的精确错误减正常贡献 |
| `gallery.html`、`figures/*.png` | 全部配对文本及分数解释、逐层逐头贡献与层级干预图 |
| `captures/*.npz` | 每对原始x、局部系数/截距、所有通道贡献、开关、末层贡献、全部干预分数 |

`captures`轴：原始x/attribution=[token,channel]；beta/logits/gates/contributions第0轴=[错误,正常]；layer_removed/added=[LLM层,token]。全部head使用 c=layer*H+head。report读的是冻结node_only，不是完整charm_in或原LLM。

先在pair内平均，再在source内平均，最后source等权和重采样。高低分来自每答20%/80%原分数分位；这些子组是模型条件选择、仅描述，不能当独立验证。多个区间未多重校正。前后半等总体有重叠，不能相加。积分失败数和各组来源分母必须先看。

`whitebox_review.tar.gz`自动收集报告表格和图，不包含逐对大数组或checkpoint。需要逐词完整1024通道时再取特定capture，不打包整个实验目录。

## 结论门槛

贡献图只有在回放、完备性/分配稳定性以及通道干预都核验后才可用于解释这个模型。即便通过，也只说明此检测器使用了这些数值组合。没有独立适用性反事实与QK/OV回放，不给头命名“纠错头”、不宣称self lock-in。

后续原LLM实验需固定词面与前缀，改变已人工核验的证据适用关系；分开测自匹配/其他key竞争与value写入。当前attention对角线不足以恢复这些量，本轮不伪造这部分代码或结果。

参考原文：Sundararajan et al., ICML 2017, https://proceedings.mlr.press/v70/sundararajan17a.html ；Zhang & Nanda, 2023, https://arxiv.org/abs/2309.16042 。原论文方法被用于诊断，不构成本数据机制的证明。
