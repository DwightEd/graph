# 先解释节点模型，再讨论图的额外作用

## 本轮问题与边界

1. `compare`：只读已保存分数，区分两个模型都识别、只有节点模型识别、只有完整模型识别、都漏检的位置。使用各自原校准阈值，不能把不同模型间的分数变化视为因果效应。
2. `node`：只读取原 `x=[self_attention(layer, head)]` 和 `node_only` checkpoint。完全不读取边张量、不运行邻居消息。主分析使用固定正常/错误配对中的所有词；按节点模型TP/FN分组的分析只是诊断。
3. `review`：只汇总现有 `routes/heads/locate/ablate` 结果和两个模型的已有分数，生成索引、图表、小压缩包。缺失的实验明确列出，不重跑神经网络，不自动宣布“锁定机制”。

## 怎样判断用了什么

原始逐头对角线的错误减正常差，是输入关联，不是模型重要性。固定同一对、同一offset，将指定层/头的节点属性双向交换，再运行冻结node_only。观察配对排序和分数间隔的变化。网络不读取任何边，因而这项结果不会被误称消息传播。

原模型回放必须通过；全通道交换应恰好交换原分数，是软件对照，不是发现。单通道/层交换可能破坏自然特征组合，掉分只说明当前模型对这些输入的敏感性。层是原LLM层，不是GNN更新轮数。每个头保留原坐标，不自动选test上表现最好的头、不反转单头评分方向。

首词、前/后半、前三分之一/中/后三分之一为重叠诊断划分，不可相加。主分析先配对再按source平均；CI为探索性bootstrap、未校正多重检验。TP/FN条件分组不用于证明自然因果机制。相同密集结构不等于相同逐头值，更不等于相同事实语义。

## 不修改什么

原图、checkpoint、阈值、分数、匹配规则以及原有四种训练/报告入口不变。没有新特征训练，没有再训练代理模型。新结果写独立audit_compare/audit_node/audit_review目录。新node只保存配对节点属性/交换分数的小数组，不导出全体边或embedding。

## 远端运行

所有命令在graph根目录、原research环境执行；默认root仍是 `outputs/charm_structure_audit_qa/QA/seed_0`。

### 已跑完的审计：先汇总，不重跑

```bash
python -u -m experiments.charm_structure_audit.main --mode review
```

读取root下已有的audit_*目录，产生 `audit_review/REPORT_zh.md` 和 `audit_review/review_bundle.tar.gz`。
已有原预测的token CSV是必要输入；它们用于复算node_only与charm_in的评价、共同配对和判对/判错交集。
自定义结果目录可增加 `--audit-inputs /实际/routes /实际/heads`，或者用 `--root`改seed目录。
每份已有结果保留自己的目录和protocol/config，不把多个seed、多个随机干预合并为一个实验。
归档只含报告、必要CSV和PNG，不包含checkpoint、原始图、embedding、逐次推理分数NPZ。

报告中的 `channel_effects.csv` 属于旧heads的完整charm_in，不是node_only。
`layer_head_routes.csv.gz` 是阶段平均，混合已检出与漏检词；不要把它解释为检出TP的逐头统计。
route_coverage_*.csv列有效pair/source数范围。没有有效观察或完整协议的头，不按0效应处理。
已有routes/heads只能分别提供路由关联和完整检测器敏感性，不能替代节点自身归因。

### 只比较已有两个模型

```bash
python -u -m experiments.charm_structure_audit.main --mode compare
```

输出 `audit_compare/`：
- `decision_counts.csv`：错误/正常分别统计两个都对、只有node对、只有full对、两个都错；错误位置再分首词、前后半、三等分。
- `comparison_tokens.csv.gz`：完整逐词两模型分数/报警、标签和位置；不再把邻居数用作节点解释。
- `pair_success.csv`：相同配对两侧的any/80%/all检出、正常完全无误报、两侧同时正确。
- `paired_comparison.csv`：相同pair、相同区域，full相对node的排序差及source区间。不是因果贡献。
- `figures/node_vs_full_position_recall.png`、`decision_overlap.png`、`matched_position_scores.png`（另有SVG）。

### 节点自身的逐层、逐头识别

```bash
python -u -m experiments.charm_structure_audit.main --mode node --channel-unit layer
```

仅运行已训练的 `node_only/checkpoint.pt` 的逐点MLP，先全体原分数回放；不读取边数组，也不执行msg_mlp。
使用原prepared/test图文件中的x和对齐元数据，不重新提取LLM attention。内存保留的是response的x，而非全部图。
`--checkpoint`在本模式指node_only的checkpoint，不要传完整charm_in；回放不一致立即停止归因。
默认每次交换一整个原LLM层的节点通道，交换双方是固定正常/错误窗口相同相对位置的token。
逐头全扫描使用 `--channel-unit head --output .../audit_node_heads`；限定层用 `--llm-layers`，指定坐标用 `--channels L:H ...`。
`--node-operations swap zero`可同时做交换与清零，两者都只影响节点输入。没有自动挑最佳头或拟合新模型。

输出 `audit_node/`：
- `node_attribute_differences.csv.gz`：所有逐L/H的原输入错误减正常差、source区间、固定高值方向的胜率（不取max(AUC,1-AUC)）。
- `cohort_counts.csv`：每个区域/TP-FN组有多少词、pair、source；all为主分析，其余分组由分数定义，仅作诊断。
- `node_channel_effects.csv`：交换/清零指定通道之后，同配对的AUROC、真假分数差和两侧报警数。
- `pair_channel_effects.csv.gz`：每一对、每个相对区域的原值与干预值，不只保存宏平均。
- `figures/`：各阶段self-attention差异热图、node自身TP/FN对照热图、通道交换效果图。
- `paired_tokens.csv.gz`与`paired_node_attributes.npz`严格按行对应；`exchange_scores.npz`各字段前M行为错误侧、后M行为正常侧，M为paired_tokens行数。
- `replay.json`与`units.json`：原分数最大误差、准确L/H坐标、全向量交换实现校验。

首词/前后半/三等分是span位置，句子位置继续用已有locate，不混为一谈。这里不重新分句。
只读取node输入无法认定prompt读取质量下降，更不能认定self lock-in。那是另一个待核实的机制假设。
新node跑完后再执行review即可将其一并导出；重新汇总不会重跑node/heads/routes。
