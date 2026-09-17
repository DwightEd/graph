# 检出在什么位置？哪些层和头负责区分？

## 先把三件事分开

1. **检出的位置**：span里的首词/中段/末段，和句子里的位置，是两个轴。
2. **路由现象**：原LLM的第几层、第几个head，读取什么来源；不是GNN迭代层数。
3. **检测器依赖**：删除或打乱某层/头的输入后，CHARM能否继续区分同一批正常/错误片段。

中后段TP多，并不推出“早期关系断裂，深层锁定”。也可能是正常词和错误词的数值分布、
词面组成、报警阈值或片段长度造成。三件事分别测量，不自动输出“机制已经证实”。

## 一、运行位置与命中计数

在graph根目录、原research环境：

```bash
python -u -m experiments.charm_structure_audit.main --mode locate --models charm_in
```

默认ROOT仍是 `outputs/charm_structure_audit_qa/QA/seed_0`。
读取原tokens.csv、spans.csv和原预测NPZ中的文本/offset/分数；不加载embedding，
不加载torch/LLM/GNN，不改阈值，不更换正常配对。不需要重新构图。
结果保存在ROOT/audit_locate/charm_in/。

### “检测成功”必须有分母和定义

- 原标注错误span：分别计**至少命中1词、覆盖≥80%、整段覆盖、首词命中**。
- 正常片段：报告**全段无误报、至少一词误报、误报token比例**。
- 相同配对：另计**错误有检出且正常完全不误报**；80%和全覆盖版本也分别计数。
- 不把“一点命中”扩展为整段token全部正确，不将span均分过token阈值称为检出。

四个总体分别保存：`gold_error`=合并重叠后的原错误段；`normal_run`=全部连续正常区间；
`matched_error`和`matched_normal`=锁定配对两侧。总体之间会重叠，**不能相加**。
每个总体内部不重复token。`normal_run`不是人工标注的完整正确命题；全正常回答也进入它。
默认仅用原pairs.json的cluster层级；`--pair-tier`可明确切换。零配对报告零分母，不补容易对照。

### 输出

| 文件 | 回答什么 |
|---|---|
| span_counts.csv | 各类span有多少、命中多少、整段正常多少 |
| paired_detection_counts.json | 同一对中两侧都处理对的数量；均分胜负另列 |
| spans.csv | 每段覆盖、首次报警延迟、所在句子、是否跨句、完整原文 |
| position_counts.csv | span三等分/每个offset、句子三等分的TP/FN/FP/TN及各自分母 |
| position_counts_by_length.csv | 按1–4/5–16/17–32/33+长度分层，避免只看到长片段效应 |
| span_sentence_grid.csv | span相对位置×句子相对位置的二维计数与报警率 |
| sentences.csv | 分句的原文与字符边界、每句TP/FN/FP/TN，可人工核查 |
| tokens.csv.gz / interval_tokens.csv.gz | 完整逐词定位/分总体逐词清单；不截断、不抽样 |
| figures/*.png 和 *.svg | 数量图、召回/FPR图、同配对分数曲线、span×句子热图 |
| gallery.html | 全部回答及其句子、配对区间、逐词分数；按ID而非成绩选择 |

位置用 `(offset+0.5)/length` 分三等分。短span可能没有中间区域；分母如实减少。
分句在**原文**上按标点/空行，用原始offset映射。保护小数、常见英文缩写和首字母缩写；
这仍是可检查的启发式规则，不是句法分析。缩写在句末等歧义需查看sentences.csv。
跨句token按最大字符重叠归属，平局取前句并标记；无文本/纯空白token为sentence=-1，
不丢出全体计数，但不冒充已定位句子。HTML按字符区间渲染一次，不重复显示重叠子词。

## 二、原LLM的逐层逐头路由审计（不运行模型）

```bash
python -u -m experiments.charm_structure_audit.main --mode routes
```

复用原prepared图（默认 `outputs/charm_structure_audit_qa/data`），只检查同一批固定配对。
没有原图时不能从均分CSV恢复逐头路由。输出ROOT/audit_routes/。

`x`和`edge_attr`的通道顺序仍为 `c=llm_layer*heads+llm_head`，**所有层、所有头分别保存**。
阶段为前5个共同正常位置pre、onset、early/middle/late thirds、post，以及onset-minus-pre。
post遇到任一侧下一错误或回答末尾停止；pre只用双方都有且都正常的位置。
每个metric/head都只比较双方同样可观测的token，不能让JS缺失导致两侧统计不同位置。

每个(layer, head)只记录九种明确观察量，不作为新模型特征进行训练：

| 名字 | 含义 |
|---|---|
| self_mass | 原对角线权重 |
| prompt_mass | 保存的prompt读取总量；**不是经核验的正确证据** |
| history_mass | 保存的历史读取量 |
| within_mass | 指向待观察片段内过去位置的保存权重 |
| within_excess | 段内读取减去“同距离/同token复现状态”的随机端点期望 |
| noncopy_excess | 上一差值中移除当前token ID复现贡献 |
| history_hhi | 保存历史权重归一化后的平方和；越大越集中 |
| shared_history_js | 相邻query在共同已有历史key上的JS；越小越稳定 |
| retained_mass | 当前所有保留入边权重加self；显示稀疏缓存的保留程度 |

条件期望：每条history边固定query、head、距离组{1},{2},{3,4},{5,...,8},...和
“source token ID是否等于当前输入token ID”；候选包含已输入但保存权重为0的位置。

```
expected_inside = sum_group(saved_weight_sum * eligible_inside_count / eligible_count)
excess_inside   = actual_inside - expected_inside
```

纯lag1续写的excess必须为0；强局部性本身不是异常复用。复现控制只是token ID，不是语义去重。
JS比较query q与q-1时，排除q-1这个新出现的key；两侧仅在相同的老history keys上归一化。
任一侧无保留历史质量就记缺失，不补0或宣称“完全稳定”。JS单位nats。

输出 `layer_head_routes.csv.gz`（逐L/H、阶段、分母、错误均值/正常均值/来源均衡配对差及区间），
`depth_contrasts.csv` 和 `depth_summary.csv`（原LLM浅/中/深层分组），以及逐层×逐头热图。
`pairs/*.npz`仅保存每对的阶段均值、观察数和坐标，**没有原图、逐边特征或embedding**。
所有stage的总体使用同一组pair，缺测单独报告。各层的同编号head不被假设为同一功能head。
按source先做配对均值，再bootstrap；CI为探索性、未做多重检验校正；单source不给伪精确区间。
不从test选择最好层/头或倒转分数方向，热图不是“显著头排行榜”。

## 三、检测器到底用了哪些通道？

先逐层扫描，区分节点通道和边通道：

```bash
python -u -m experiments.charm_structure_audit.main --mode heads \
  --channel-unit layer --channel-sites node edge
```

每层所有head一起清零。要逐头全扫描，显式 `--channel-unit head`，另给新 `--output`。
`--llm-layers`可限定原LLM层号，`--channels L:H ...`可指定准确坐标；均为0起始。
完整逐头扫描每回答会运行L×H×site数次小GNN，**不是**重跑8B；不要把层级扫描冒充逐头扫描。
选择探索中发现的头再测同一个test，仍是探索分析，不是独立确认。

还可以按LLM层定位多头的“同端点组合”敏感性：

```bash
python -u -m experiments.charm_structure_audit.main --mode heads \
  --channel-unit layer --channel-sites edge --channel-operations coupled independent \
  --output outputs/charm_structure_audit_qa/QA/seed_0/audit_channel_pairing
```

只改被指定层的边向量，其余层不动。同query/来源角色/距离组内，coupled把该层多头向量一起换，
independent让各头独立换。每头的权重列表、总量、条件熵不变。单头下两种控制应相同。
输出 `independent_minus_coupled.csv`，直接比较同一批pair而非各自随意选样本。
这仍可能由边向量的非线性数值组合造成，不能单独等同于“语义邻域协作”。

每份图先回放原checkpoint验证分数，再干预；所有控制保留原union边和入度分母。
`channel_effects.csv`：各层/头/site的整体AUROC/AP变化、同配对AUROC和分数差变化。
`role_changes.csv`：首错、后续起点、续错、正常词的有符号分数变化及丢失/新增报警。
`pair_effects.csv.gz`：每对、同相对区域的原模型/干预结果。
`changed_cells.csv`：实际改了多少值、是否产生全零边槽位。零改动不解释成该头无用。
`scores/*.npz`仅存每种控制的逐词分数，用于复核，不导出大表征。

**此处干预的是CHARM读取的输入通道，不是原LLM的attention计算，更不是生成干预。**
清零有分布偏移。邻居依赖、头值依赖与真实幻觉原因分开；旧的no_source/no_relay/独立重训模式仍保留。

## 对“早期断裂→深层锁定”的判据与边界

可以检查：错误相对正常的onset-pre prompt变化是否在浅层更负；中后段深层是否有更多超额段内
读取/更强历史集中/更低共同key JS。必须同时看各自分母、稀疏保留质量和正常控制。
哪怕同时出现，也只是这种路由模式的关联证据，不是自动定位“证据未整合”或“残差readout沉默”。
原缓存没有value向量、功能消息能量或原模型梯度，本轮没有任何原LLM因果干预。
原CHARM将全部L×H通道送进MLP混合，并非沿原LLM层依次传播；GNN层与LLM层不能互换。
当前仍是post-token i状态对label i；不是生成前提前报警。

文献核对：Reasoning Fails Where Step Flow Breaks (arXiv:2604.06695)报告Shallow Lock-in和Deep Decay，
用的是attention-gradient step saliency，不是这份阈值attention图。
Two Pathways to Truthfulness (arXiv:2601.07422)区分question-/answer-anchored信号；不能将answer/history读取一概视为错误。
这两个结论不直接推出本数据的机制；本模块是单独可否定的实验。

## 验证范围

运行：

```bash
python -m pytest tests/test_charm_structure_audit.py tests/test_charm_prediction_memory.py \
  tests/test_charm_localization_routes.py -q
```

仅运行合成数据及真实CLI测试，不运行用户自然数据的新推理或新统计。测试检查旧模型不变、原文件不变、
句子offset/小数缩写/中文标点、正常/错误分母、非point-adjustment、逐头通道、共同可观测位点、纯lag1基线、
copy控制、共享key JS缺失、source均衡及单来源CI、每层局部coupled/independent对照和三个新模式。
