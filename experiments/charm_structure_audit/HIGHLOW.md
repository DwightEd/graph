# 高低分 token：先解释节点数值，不把邻居现象当成消息机制

## 这个实验解决什么

输入仍有信号、无标签表示保留信号、异常分数把信号排到前面，是三件不同的事。
固定图 nodes 使用九类路由统计、128维 CountSketch 和无标签近邻距离；不是 CHARM node_only 的逐头 self 对角线。
此外 fixed_graph 用 q=P+t-1 预测位置，CHARM 用读入 t 后的 q=P+t。两者成绩不能用来直接隔离“监督/无监督”或“有图/无图”的贡献。
本模式不训练、不改图、不产生新检测器；用已保存分数分组，再对照原始节点输入。

## 运行

仍在 graph 根目录，原 research 环境。先用节点模型，完整模型另列：

```bash
python -u -m experiments.charm_structure_audit.main --mode highlow \
  --models node_only charm_in --node-values
```

默认读取 `outputs/charm_structure_audit_qa/QA/seed_0` 和原 `.../data` prepared 图。
只读取 x 和身份/标签/文本，不读取 edge_attr，不加载 checkpoint，不运行消息网络。
不加 `--node-values` 时只需要已有 tokens.csv/spans.csv/threshold.json，不能得到原始逐头热图。
各模型分别按自己的分数分组，不拿 charm_in 的 TP/FN 冒充 node_only 的分组。
当前结果写 `audit_highlow/<model>/`；原模型、图、预测、配对完全不变。

## 先分组，再揭示标签

在每个回答内部，以分数20%/80%分位数划分 low/high，其他为 middle。
两个标签共享同一分位数。分数并列不按标签打破，分位数相同则记 tied；因此不是强制精确20%的抽样。
这是事后分数诊断，不是测试集调报警阈值。原报警继续使用各模型独立校准的严格大于阈值。
高分正常不是自动等同FP，低分错误也不被自动重标；predicted/outcome保持原值。
所有回答，包括全正常回答，都保留，因此不能把相对高分组里的正常数量当作原模型误报数。

四组都要看：高分错误、低分错误、高分正常、低分正常。不能只展示第一组和最后一组。
`tokens.csv.gz`保存每个词的组、分数、报警、原标签、span位置、回答位置和词面类别。
`score_groups.csv`给组数、来源、均分；`composition.csv`检查组间位置/词面组成是否不同。
`score_bins.csv`是同答十分位排序的实际错误比例，绝非校准概率。

## 原始逐头对照

没有新增用于训练的手工特征。只使用 x 中每个原 LLM (layer, head) 的 self 对角线。

1. 固定原 cluster 配对，相同 offset 的错误减正常。正常伙伴不因模型分数而改变。
   输出全部配对，以及按错误高/低分限制的组；另报后半段。条件组只是解释分数选择，不证明机制。
2. 分别在错误和正常内部比较高分减低分；要求同回答、同一个标注span/正常连续区间、
   相同相对三分之一区域、相同粗词面类别。没有同时出现 high 和 low 就不贡献观测。
   单元先均值，再按片段均值，最后按 source 均值；不把每个 token/head 当作独立样本。

每头保留原坐标，导出 `node_value_contrasts.csv.gz`。区间仅为探索性 source bootstrap，未校正多重检验。
数值差不是重要性；需要已有 `--mode node` 的冻结节点通道交换来检验模型是否使用它。
不把某头变大/变小自动命名为事实核查或 self lock-in，不自动在test挑最佳头、翻转评分方向。
同编号head在不同层不被假定为同一功能通道。这里也没有把GNN层当作LLM层。

## 无监督高低分是否找错对象（已有 fixed_graph 可选）

```bash
python -u -m experiments.charm_structure_audit.main --mode highlow \
  --models node_only --fixed-output outputs/fixed_graph_v1 \
  --output outputs/charm_structure_audit_qa/QA/seed_0/audit_highlow_fixed
```

只读已经完成的 fixed_graph/predictions；不重建表示、参照库或分数。
用原 prepared token IDs、文本、gold和source核对同一目标词，保留共同覆盖，不给缺首词补分数。
`fixed_score_comparison.csv`统一共同覆盖上的六个无监督分数与监督分数。
`fixed_score_tail_crosses.csv`查看监督高/无监督低、监督低/无监督高等组中真假各多少。
`fixed_score_tokens.csv.gz`保留所有原坐标和原分数，不能用人工调换方向得到更好结果。
这里没有消除两种方法的观测位置、原始属性、压缩和拟合划分差异，不能称为损失函数单因素实验。

## 图和小包

`figures/error_fraction_by_score_rank.png`：按同答分数排名的错误频率，标明分母。
`figures/four_score_groups.png`：四组数量，而非把高分等同错误。
`figures/*high_minus_low.png`：同标签、同区间对照的逐层逐头数值差。
其他逐头热图为固定正常伙伴上的差异；无原 x 时不生成假的热图。
`examples.html`展示四组分数极端词及全部配对词；选例不用于估计性能。
同时生成SVG。顶层 `highlow_bundle.tar.gz`只包括CSV/JSON/PNG，不包括原图、checkpoint、NPZ或embedding。

此前 routes/heads 已完成时，先用已有 `--mode review` 生成 `audit_review/review_bundle.tar.gz`；不要重跑模型。
这份包用于路由关联/完整模型敏感性，本模式的包用于节点高低分分析，不能混称新因果发现。

## 验证与下一步边界

新测试检查并列分数、标签盲分组、固定伙伴、同区间同词面控制、source均衡、无边/无torch实际CLI、
无监督共同覆盖与错误token IDs拒绝。实际上传数据只复算CSV；原始逐头x和新的fixed_graph结果未在本环境提供。
要判断是输入、表示还是评分的问题，需在**相同输入/时刻/来源划分**下比较原x与冻结z的诊断读出，
以及各自的无标签评分。诊断读出用标签时应明确是监督诊断，不能计入最终无监督检测成绩。
本模式不偷偷训练这个读出，也不承诺某种先验一定有效。
