# 逐 token 来源、软单元约束与允许突变的联合读出

## 要修正的问题

旧 `unified` 把来源作用先压成文本单元均值，再用路由偏差区分内部 token，
并严格要求偏差在每个单元内均值为零。因此，来源作用在同一单元内发生变化时，
这部分逐 token 信息没有进入主评分；路由和图也不能纠正单元的来源均值。
它本来就能输出不同的 token 分数，但模型把细粒度定位几乎全交给路由。

用户返回的 carrier token v2 输入上的旧联合结果：

| 旧读出 | pooled token AUROC | AP |
|---|---:|---:|
| unified | .879176 | .341974 |
| unified_no_graph | .871262 | .345599 |
| unified_random_graph | .880975 | .340796 |
| source_consensus | .888905 | .378089 |
| source_local_unit_mean | .914856 | .451496 |

这些是用户提供的 JSON，不是本轮重跑。真实图没有一致胜过随机图；
不能把该结果描述为图拓扑已经改善了定位。`.914856` 是单元均值广播后按 token 计算的
AUROC，不是片段 AUROC 的平均，更不等于 91.5% 正确率。

## 一个目标函数，四个明确角色

所有原始分数方向均保留：数值越大表示候选风险越高。沿用无标签、来源等权的经验 mid-CDF，
分别拟合 token 与单元的尺度。秩不是错误概率，不能把不同尺度下的数值当概率相加。
这里等权融合是预先固定的评分假设，未用四答标签优化权重。

- `r_t`：原始 route 的 token 秩。
- `l_t, c_t`：原 `source_local`、selected-history source 分数各自的 **token 秩**。
- `s_t=(l_t+c_t)/2`：逐 token 来源观测。
- `y_t=(s_t+r_t)/2`：逐 token 的来源／路由等权观测。
- `a_u`：沿用旧方法，先求两个来源分数的单元均值，各自转单元秩，再等权平均。
  `rank(mean(source))` 与 `mean(rank(source))` 不相同，不能混用。

求完整回答的向量 `z`：

\[
\min_z\ \frac12\|z-y\|_2^2
 +\frac\alpha2\sum_u n_u(\operatorname{mean}_{t\in u}z_t-a_u)^2
 +\frac\gamma2(z-s)^T L(z-s)
 +\tau\sum_{t=1}^{T-1}|z_t-z_{t-1}|.
\]

1. **token 观测**：每个位置自己的来源作用直接进入 `y`，不必经过单元平均。
2. **软单元约束**：片段信息可降低 token 噪声，但不再固定单元均值。
   权重 `n_u` 使单元先验与其覆盖的 token 观测具有可比较的总量。
   无图、无 TV 时，单元均值恰好是 `(mean(y)+alpha*a)/(1+alpha)`。
3. **图修正**：继续保留实际逐头删除的来源交互与历史端点，构造原有 `L`。
   图平滑的是 `z-s`，允许依赖两端已有不同的来源观测，不要求两端绝对风险相同。
   “修正量相关”仍是假设；与随机端点、去图对照一同报告。
4. **连续性**：一阶 TV 惩罚相邻 token 的绝对分数差。小抖动可合并成平台，
   强观测变化仍可保留跳变；它也会压低过短、过弱的真实信号，不能保证边界恢复。
   相邻关系覆盖整个回答，不在金标或现有文本单元边界处强制断开。

默认 `alpha=1, gamma=1, tau=.05`，在本轮自然缓存评价前固定，无参数搜索。
前两者表示单位强度的先验／图正则；`.05` 是 `[0,1]` 秩尺度上的 TV 系数，
不是经验最优值、检测阈值或可迁移的事实常数。分数没有概率校准，也不截断成概率。
没有 exp 位置权重、人工词类过滤、熵真假规则或另训分类器。

TV 与 ADMM 是已有数值方法，参考 Wahlberg et al. (2012)：
https://arxiv.org/abs/1203.1828 。它提供保留变化点的正则形式，
不提供“变化点就是幻觉边界”的结论；本项目新增的是这些观测与约束的具体组合。

## 实现与运行

```bash
bash experiments/native_support/run_unified_token.sh
```

默认输入 `outputs/native_support_ragtruth4/message_carriers_token_v2`，
输出 `outputs/native_support_ragtruth4/unified_token_v2`。也可读取完整 review ZIP：

```bash
python main.py transport-unified --token-readout \
  --input /path/to/message_carriers_token_v2_review.zip \
  --output outputs/unified_token_v2 --resume
```

`--token-readout` 是明确的方法修改；旧命令仍运行硬单元约束 v1。
新入口保留旧 `unified`、原始风险、所有输入基线，主候选单独命名 `unified_token`。
固定消融为去图、去软单元约束、去 TV、去 token 来源观测、随机历史端点，
并保存未正则的 `token_observation`。去 token 来源观测仍保留单元来源先验和来源构造的图，
不是“完全不用来源”。消融不会根据标签自动替换主候选。

`token_solver.py` 建立稀疏二次项，用 ADMM 联合求解整条 token 向量；
每次迭代解一个稀疏线性系统，不逐 token 重跑模型。固定 `rho=1`，
原始／对偶最大残差容差 `1e-9`，最多 10000 次迭代，不收敛就报错。
对确定的平台再解约束二次问题并核验 KKT，避免数值尾差把理论相同的分数排出假次序。
默认全流程只复用缓存，无新增模型前向；原生 carrier **采集** 仍是逐 token 的独立测量。

保存 `source_token`、`local_token`、`carrier_token`、`token_observation` 和
`token_source_correction`，以及各消融收敛误差和单元均值移动量。
全部回答评分冻结后才读取标签；外部参考与目标来源不重叠。
默认仍使用整个未标注 cohort 的尺度，属于 transductive 离线评分；不是在线预警。

v1 carrier 缓存同样可用，但其边来自单元目标筛选／查询束干预，不能冒称 v2 的独立 token 回溯。
完整逐头表示仍保留于输入缓存，本读出没有训练 7190 维表示分类器。

## 应怎样评价

除了 `evaluation.json` 的 pooled token AUROC/AP，要同时看：

- `within_unit.json`：**同一个单元内部**正常／错误 token 的排序；无混合单元时为不可评价。
- `ranking_scope.json`：同回答与跨回答排序，不能把跨回答校准收益当精确定位。
- `annotation_alignment.json`：混合单元数量与覆盖；现有上传 v1 四答 41 个单元中为 0。
- `unit_evaluation.json`：单元等权评价；与 `_unit_mean` 广播后的 token 指标不同。
- `diagnostics.json`：优化是否收敛、软先验被移动多少。

合成平台／跳变测试只检验求解器能表达局部变化，不能作为自然幻觉检测收益。
四答反复用于方法设计，所有成绩为探索性。来源敏感性不是事实充分性，
输出一个 token 分数也不代表已能可靠判断任意推理步骤的逻辑正确性。

## 本轮真实缓存结果

本轮实际运行的是已上传 **carrier v1** 完整缓存，4 答、836 token、81 标错。
用户 token v2 完整缓存未上传，不能把下表当作 v2 输入的结果。

| 方法 | pooled token AUROC | AP |
|---|---:|---:|
| unified_token（主候选） | .871621 | .355191 |
| 去图 | .879854 | .357251 |
| 去软单元约束 | .793361 | .256368 |
| 去 TV | .840307 | .322348 |
| 去 token 来源观测 | .865686 | .438550 |
| 随机端点 | .869577 | .338934 |
| 未正则 token_observation | .761213 | .230514 |
| 原 unified，同输入 | .877475 | .356279 |
| source_local_unit_mean | .914856 | .451496 |

主候选仍未超过旧 unified 或最强单元均值。去图优于完整模型，故图修正没有收益结论；
去 token 来源观测的 AP 更高，说明直接加入来源观测也存在噪声与排序取舍。
软单元先验及 TV 在本组固定消融中有收益，但不能据四答宣称泛化。
主候选答内 AUROC=.908303，旧 unified=.918907；均无混合单元定位成绩。
新主候选再做单元均值后为 .904701/.434463，与原硬约束读出不再必然相等。
没有据上述结果调参、自动换主候选或删除失败分数。

软件测试覆盖闭式软先验解、单元内部跳变、平台精确同分、图只约束修正量、
同单元均值但不同 token 来源的可辨识性，以及两个原生缓存版本的流程和标签隔离。
本次实际求解最多 128 次迭代，最大 KKT 误差约 `3.9e-14`。
这些验证证明代码实现了指定目标函数，不证明目标函数足以识别事实错误。
