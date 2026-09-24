# 监督读出：现有响应表示能否区分幻觉

2026-09-24。用户明确授权自然标签训练。入口 `main.py transport-readout`。
原无监督评分、捕获张量和 annotation 保持不变；本任务明确属于监督诊断。

## 1. 检验目标与边界

问题是已有表示 X 是否含有跨来源可迁移的 token 标签信息，测试 p_theta(y_t=1|X)。
不是优化原回答的语言模型似然，也不是从“有贡献”推出“继承事实”。
分类器采用监督目标学习判别方向，可以利用之前固定正负/比例读出未利用的模式。
但自然幻觉标签没有标出适用证据、事实决策点或作用路径：预测成功不等于回答这些因果问题。
预测失败也不能证明表示完全无信息：样本数、模型容量、表示压缩与来源偏移都可能限制结果。

本次没有重新采用 alpha=1/2 的分配，也没有改写旧归因规则。
原生Fisher响应沿用现有无符号测量；选填choice视图明确是旧分解产生的历史观测。
归因规则稳定性、有限删除/恢复属于另外的机制验证，不以完成这些验证作为训练前提。

## 2. 视图与先验覆盖

| 视图 | 输入 | 检验问题 |
|---|---|---|
| position | 相对/对数位置、答案长、prompt长、块数（5维） | 单靠位置/篇幅是否就能读出标签 |
| routing | raw route、observable route、raw attention（3维） | 有效方向性信号能否被重新校准 |
| instant | 10个现有读取/响应/变化量，加每层5个operator观测（32层时170维） | 当前已有逐层统计是否可读出标签 |
| heads | instant + 每个物理头的31个来源/响应/交互观测 | 平均头前是否还存在可用模式 |
| native_state | instant + 原条件核的完整响应因子A、sqrt(reading)、log1p(scale) | 直接学习原核所持有的结构，避免只测手工汇总 |
| temporal | 当前、前window均值、后window均值、当前减过去均值，加两端可用性 | 延续/变化是否提供增量 |
| choice（选填） | instant + local/lineage source deficit、local/lineage opposition | 历史正向缺口及失败反对量是否能被监督读出利用 |
| choice_temporal（选填） | choice的同一时序展开 | 历史读出在时间维是否提供增量 |

默认 window=16，固定一次，不根据标签搜索。temporal 在summary模式展开instant，
heads模式展开heads；不会对原核大因子再做4倍时序展开。
native_state 固定原probe坐标，不是完整head×head Gram显式展开，也没有训练SVD；
它保留原核所用因子的全部坐标及尺度，但原来合并来源块造成的信息损失仍然存在。
默认逐头和native_state取层索引 [L//4, L-L//4)，32层即8到23；可显式选择all。
这不是根据本四答成绩选层，不能称为已证明的最佳层段。

逐头31个通道：4角色attention质量；3种响应×4角色的log1p范数；
4角色residual/FFN余弦；source与history/self/other总响应余弦；以及8个块状态量。
FFN的余弦可以为负，但没有任何固定“负=反对事实”标签。原响应范数/尺度都保留。
不同头保持固定列身份，训练可以赋予不同权重；没有声称这是因果协同。

### 来源块保持

仅在同一回答内跟踪相同来源块 g。a_t[g] 为某层头对该块的attention质量，m_t=sum_g a_t[g]：

    memory_t[g] = (1-m_t) memory_(t-1)[g] + a_t[g]

初始memory=0。当前读入越多，新块分布占比越高；几乎不读来源时保留此前分布。
更新前计算当前读取和memory、当前读取和前一读取、读取和来源响应强度分布的重叠。
没有标点分段、最大attention阈值、金标首错重置；不要求reanchor必须从local跳到远处。
这个memory是attention加权的统计记忆，不是大模型真实隐状态，不继承任何事实正确率。
它不能区分一个token是否仍然认同此前事实；也未加入逐历史key的实际复用图。

可以表达读取集中、块切换、来源依赖保持等观测；仍不能识别“哪条证据适用”、
“此时是否决定新事实”以及“约束是否被语义采纳”。这些缺口必须如实保留。

## 3. 训练协议与防泄漏

- 默认leave-one-source-out，同一来源的所有回答整体留出。没有随机拆token。
- `--reference` 用明确独立来源作训练；训练/评价来源相交直接报错，模型/probe/列身份须一致。
- imputer、scaler、classifier仅拟合训练有效token。gold boundaries/标签不构造特征或memory。
- 时序窗口仅在回答内部计算；允许未来token，明确离线。不是实时检测。
- 排除transport_context/conditional作为训练输入：其全cohort参考可能包含留出来源。
- 不把response/source ID、原文token ID作为分类特征。
- 位置等混杂单独训练；不能从instant好于position就证明已排除全部位置混杂。
- 标签为二元自然标注。训练来源先等权，再使用类别平衡权重。预测概率未经独立校准。
- 一折训练若只有一类，则明确不可训练，输出NaN；不填造分数。同时报告共同覆盖指标。
- 反复研究过的四答是探索集：来源留出仅隔离这轮拟合，不把它升级成全新确认集。

固定模型，不做超参搜索、不根据留出指标选模型：

1. Logistic：训练中位数填充+缺测指示，StandardScaler，L2、C=1、liblinear，max_iter=2000。
2. Trees：相同缺测处理，HistGradientBoosting，80轮、7叶、深度3、min_leaf=20、L2=1。
   关闭early stopping，避免内部随机token验证划分。seed=37。

保留训练集回代AUROC作为过拟合诊断，主成绩为留出预测。
所有方法报告token AUROC/AP、within-answer、source-balanced、首错、span onset、延续、
前后半段和共同覆盖top decile。首错只有2个，不能作强结论。未选择阈值。
模型按fold保存joblib；原始响应无需再次GPU采集，CPU时间取决于样本数和列数。

实现依据：
- https://scikit-learn.org/stable/modules/cross_validation.html （按group划分）
- https://scikit-learn.org/stable/common_pitfalls.html （预处理只拟合训练集）
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

## 4. 已实际运行：四答轻量包

输入用户 `observable_transport_v1_review_light.zip`；836 token，81标错，4个来源。
2个来源没有标错，另外2个来源分别68、13个标错；一旦留出其中一个含错来源，
训练就只剩一个含错来源。这是非常弱的泛化诊断条件。
没有重跑8B；固定参数运行一次。summary模式四视图×两模型；没有运行heads/native_state/choice。

| 方法 | 留出AUROC | AP | within-answer pair AUROC | top84标错数 |
|---|---:|---:|---:|---:|
| raw_route（无监督基线） | .755425 | .199692 | .747461 | 20 |
| observable_route（无监督基线） | .753070 | .205930 | .744698 | 21 |
| route_offline_mean（无监督基线） | .784580 | .193649 | .788978 | 12 |
| position_logistic | .628011 | .120376 | .752838 | 13 |
| position_trees | .340553 | .101475 | .606593 | 5 |
| routing_logistic | .694825 | .223465 | .748432 | 21 |
| routing_trees | .477892 | .097387 | .665509 | 9 |
| instant_logistic | .414455 | .080818 | .524418 | 6 |
| instant_trees | .326907 | .073117 | .581840 | 1 |
| temporal_logistic | .629662 | .143036 | .606033 | 16 |
| temporal_trees | .587744 | .115258 | .696573 | 8 |

instant_logistic训练回代AUROC .952978–.997465，temporal_logistic全部为1.0，
留出却差：明确出现过拟合/跨来源迁移失败，不是“多加分类器就能补回语义”。
routing_logistic 的within-answer约 .748与raw接近，pooled更低；跨折分数尺度和
不同答案分布也会影响pooled排序。不能只展示AP增益而隐去AUROC退步。
position_logistic 的within-answer也约 .753，提示同答排序仍有明显位置混杂。

当前结论：没有获得整体检测改进。尚不能判断原始逐头/原核因子是否存在更可迁移的信息。
服务器已有完整缓存即可运行对应视图，不需要上传1.66GB或重新做8B前向。
后续应增加独立来源中的错误模式覆盖，而不是在四答上反复调C、窗口或挑指标。

## 5. 文件与运行

`bash experiments/native_support/run_readout.sh` 默认完整缓存、中间层、逻辑回归。
只使用轻量包见 native_support/README 的summary命令。
结果：protocol/settings/schema、folds（训练来源及回代指标）、evaluation、metrics.csv、
逐token留出预测、逐答scores、models；自动轻量包排除models。
要求新输出目录防止改写旧实验。原始无监督风险仍保留，监督输出使用独立方法名。
