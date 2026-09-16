# 完整回答的离线片段图检测

状态：**已有可运行实现，不再是空接口。** 训练一个独立的关系评分器，不使用幻觉标签；
最后才读标签评价。可以读取未来位置对当前内容的使用，但不称为在线预警。
原来的 SourceFlow、FlowTracer、自编码器、main.py 和缓存均不修改。

## 先检查现有缓存，无需加载大模型

在 graph 根目录执行，环境使用 research：

```bash
python -m experiments.unsupervised_token_graph.offline_span --phase inspect
```

默认使用已有路径：

```
/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
```

检查输出包含样本数、source_id缺失数量、通道数、预测位置覆盖率、hidden维度和offset。
若缺少 source_id，必须用 `--index /实际已有的/inputs.jsonl` 或 records.jsonl/records.json，
也接受其所在目录。不要按文件序号编造source_id。默认复用 cache目录或父目录的现有索引，
没有要求重新生成metadata。完整回答共享source的划分在训练之前检查。

## 一次执行

```bash
python -m experiments.unsupervised_token_graph.offline_span \
  --phase all \
  --tasks all --generators all \
  --output outputs/offline_span_v1 \
  --device cuda:0 \
  --epochs 5 --resume
```

`cuda:0`用于小检测器，不加载8B模型。也可用 `--device cpu`。
进度分为prepare、epoch、calibration、score；不后台、不使用`set -e`。
训练/评分不需要annotation文件；`all`在评分冻结后尝试评价。默认标注路径为
`/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl`。
文件不存在时明确跳过评价，保留已完成分数。用`--annotations`指定原文件。

只选一个已有生成器：`--generators llama-2-7b-chat`，这选择缓存身份，不是新生成。
任务/生成器身份缺失时无法据此过滤，先补指向已有索引，而非重贴标签。
更换位置：`--train-cache ... --test-cache ...`。输入可以是目录或一个NPZ。

## 配套的数据接口

直接复用父模块 `ResponseCache`、`CacheIndex`、`iter_channels`，没有另造第四套attention解码。

| 原格式 | 直接支持的字段 |
|---|---|
| 原canonical CSR | token_ids、response_idx、attention_diagonal、response_row_ptr、response_column_indices、response_values |
| dense/legacy | attention或data [L,H,Q,N]；compact无query_positions时沿用q=P-1+r |
| 单层单头导出 | adjacency [R,N]、layer、head、token_ids、response_idx；q=P+r |
| 现有身份索引 | id/response_id/sample_id、source_id、split/official_split、task、generator/model、offsets |
| 可选原始hidden | hidden或hidden_states [N,D]；[K,N,D]必须显式传--hidden-layer，含全部prompt+response |
| 可选预测熵 | entropy或next_token_entropy [R]，明确对应每个回答词出现前的预测 |
| 可选材料结构 | source_groups [P]，同记录用同组；source_mask [P] 排除指令/非证据 |

同答的不同layer/head文件合并为一个样本，同一物理通道重复则报错。原token和prompt边界
必须一致。不打开NPZ中的labels/hallucination_labels，不反序列化其object数组。
六字段CSR本身能构图；训练还需要source_id及train/test身份。缺失offset不影响评分，
评价沿用已有EvaluationBinding：用**原本地observer tokenizer**验证完全相同token ID后
恢复offset；不会仅按token数估计。必要时 `--tokenizer /原tokenizer目录`，不重跑模型。
PT/pickle、仅local_attention窗口、S10五列values不是此全图接口，不自动伪装成完整图。

### 两种明确的内容输入

默认 `--feature-mode tokens`：现有attention + token IDs。训练词项embedding与图关系参数，
不预设已知语义/正确绑定。**没有隐藏状态时，不伪造hidden、V、W_O或FFN消息。**

已有hidden时：

```bash
python -m experiments.unsupervised_token_graph.offline_span \
  --feature-mode hidden --hidden-layer 16 \
  --feature-root /实际已有的/hidden_npz目录 \
  --output outputs/offline_span_hidden_v1 --device cuda:0 --resume
```

该目录的文件名为 `<response_id>.npz`，必须包含相同完整token_ids及明确hidden字段。
若hidden就在attention NPZ中，不传feature-root。`hidden-layer`是缓存数组下标，不假定
它就是模型物理层号。本版只使用明确指定的一层，不跨hidden层求平均。
response-only / selection-only的机制缓存不能直接当作[N,D]；缺失时明确报错而不是退回tokens。

## 这版具体的模型

1. 构图保留逐物理layer/head的真实读取端点。默认每行分别保留最强的2条prompt边和2条
   history边（--edges-per-partition；0为全保留），保存原权重及missing/pruned质量。
   不扩成[L,H,N,N]。后续读取只生成离线索引，不伪造反向原生边。
2. 检测器按层顺序更新节点，每个head使用独立可学习的门控后汇总消息。同层的消息全部
   使用更新前状态，因此不会在同层把乘两次邻接矩阵说成真实两跳。所有原始head ID保留。
   这是**检测端attention图编码器**，不是原LLM残差/FFN/WV-WO的精确重演。
3. 每段保留当前内容、首尾/强来源读取的候选选择位置，以及后面实际读它的最多8个节点。
   证据侧取该段选择位置读取最强的一个source group。没有source_groups时以32-token
   prompt窗口作上下文近似；没有source_mask时整个prompt都是候选，包含指令，不能叫纯事实。
   一组不是完整命题；本版没有通用命题抽取或多个独立事实因子的显式绑定模型。
4. 单独编码证据与选择/段内/后文，再学习相容logit。训练原始配对对同材料的来源错接、
   选择根错接；不重构节点或边。原始配对允许错误，人工错接也不自动等于幻觉。
5. 区间异常是负相容logit。按source划出约20%的无标签校准集，按长度/任务/生成器校准。
   评分校准并非正常样本筛选，也不承诺正常回答的5%误报保证。
6. 在长度预算内保留全部起点和单token候选，不需熵门槛。用区间动态规划给出MAP边界，
   同一能量的前后向求和产生连续token分数。禁止相邻异常段的多种分解，允许空解与单词错。
   当前单个异常段最大长度为--max-length（默认32），不暗中合并突破该限制；长span会受限。

关键代码：data → graph → regions/controls → model → learning → detection。
`model.py` 的encode_graph、score_encoded与`detection.py`的interval_inference是核心。
词项embedding词表只由fit来源确定；未知词映射到UNK，可能降低泛化能力。仅tokens模式
不能声称已经识别阶段、否定和正确事实；hidden模式也必须用自然评价检验，不保证更好。

## 训练、续跑、输出

原LLM参数不进入此模型。按source拆分并阻止official train/test来源重叠；默认每轮每个
fit source抽一个回答，防止同source多生成器占据训练。每个样本最多128个候选种子配对，
两种可用对照的损失等权。没有合适对照的训练样本报告数量，不删除其测试位置。

图和分数逐样本保存。训练每个完整epoch写checkpoint；中途终止从最近完成epoch重启，
不承诺恢复到某一batch。--resume不允许改变已保存训练/图设置。旧模型完整则复用。
想作不同消融，使用新output。缓存和原回答不会被写入、重编码或改动。

```
outputs/offline_span_v1/
  graphs/settings.json, manifest.json, train/*.npz, test/*.npz
  model/model.json, weights.pt, checkpoint.pt, reference.json, training.json
  predictions/samples/*.npz
  predictions/prediction_freeze.json
  predictions/evaluation.json
```

每答NPZ含offline_span、singleton、history_minus_prompt、entropy、position；全部原token
均保留，缺失预测行是NaN/coverage=false，不补零提高指标。CSR缺少q=P-1时首token未覆盖。
另含最终span_bounds/score及全部candidate_bounds/score/potentials，便于逐样本调试。
评分时一答图只编码一次，再分批读区间；不枚举全部路径、不做JVP、不生成新回答。

## 评价与消融

`--phase evaluate`只读已保存分数，支持延后提供tokenizer，不需重新训练。
报告父项目的七种token范围、每任务/生成器与合并AUROC/AP、固定完整source权重、source
bootstrap；token边界exact/IoU≥.5的一对一span F1、匹配边界偏差、结束后8个正常token
误报、正常回答任意报警。`paired_vs_singleton`使用同覆盖范围做配对source重采样。
这些是离线定位结果，不是预测提前量。自然原标注字符边界与token边界不同，exact明确
按token区间，不冒称原字符严格匹配。

对照入口：--relation none / permuted；--future-budget 0；--max-length 1。各自新目录
重新训练，同数据划分。permuted只保持query内prompt/history分区和权重，不声称完全
保持粗距离；复杂统计匹配的额外对照与“同一单点分数仅平滑”仍应在研究评价中补齐。
真实图/整段/未来上下文有效性不能从对比loss下降推断。

## 本地验证边界

```bash
python -m pytest experiments/unsupervised_token_graph/offline_span/tests -q
```

测试使用实际父模块（五个原文件与读取的Git blob一致）：dense/CSR/逐头格式、索引配套、
labels隔离、hidden契约、梯度、所有3任务的端到端、续跑、区间DP对穷举、来源拆分与评价。
没有访问服务器自然cache、没有训练真实自然数据、没有新的AUROC/AP成果。
此前设计中的post-WO/FFN原生向量分支与显式多事实绑定，不属于本次已实现部分。
