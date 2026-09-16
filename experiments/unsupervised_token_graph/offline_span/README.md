# 完整回答的离线片段图检测

入口：`experiments.unsupervised_token_graph.offline_span`。
原 LLM 不加载；训练的是独立的小型图关系评分器。
先做无标签评分并保存，再用原标注评价；没有新的自然数据成绩。

## 1. 检查现有输入

在 graph 根目录、research 环境中运行：

```bash
python -m experiments.unsupervised_token_graph.offline_span --phase inspect
```

默认读原有两目录：

```text
/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
```

优先使用 NPZ 或现有 inputs/records 索引。身份缺失时，按回答编号关联原始
`response.jsonl` 的来源、划分、生成器，再从 `source_info.json/jsonl` 读取任务。
自动查找 cache 祖先目录及其 dataset 子目录；原数据在别处可指定：

```bash
python -m experiments.unsupervised_token_graph.offline_span \
  --phase inspect --dataset /原RAGTruth目录
```

已有独立索引仍用 `--index /已有的/inputs.jsonl`，不要求重新生成metadata。
元数据可能与标签保存在同一个文件，但只提取身份字段，不使用 labels、标注边界或回答内容。
缺失 source_id 不以文件序号代替；冲突不覆盖。

重点看 `missing_source_ids`、`unknown_tasks`、`unknown_generators` 是否为0，
以及 `ready_to_fit_metadata`。coverage、hidden、offset仅描述first_id，不是全体统计。
`hidden=[N,0]` 表示没有加载hidden；当前NPZ是否保存过它，另看
`hidden_fields_in_selected_archive`。

## 2. 训练、评分、评价

身份齐全后运行：

```bash
python -m experiments.unsupervised_token_graph.offline_span \
  --phase all --tasks all --generators all \
  --output outputs/offline_span_v1 --device cuda:0 --epochs 5 --resume
```

`--device cuda:0` 只用于小检测器，也支持cpu；不重跑8B、不重新生成回答。
原始身份字段已完整时不需要打开原dataset；`fit/score`只读准备好的图。
`--phase evaluate` 在冻结结果之后读取标签，默认用自动找到的原response文件，
也可用 `--annotations` 单独指定。原文件不存在时明确跳过评价，保存的分数不丢。

图和分数逐样本续跑，训练从最近完成epoch继续。改变实验参数需新output。
首次只执行inspect不会创建输出；本次元数据接入改变了图身份设置，旧prepare结果不混用。
不使用 `set -e`，不后台运行。

## 3. 数据格式

复用父模块 ResponseCache、CacheIndex、iter_channels。

| 数据 | 支持的格式 |
|---|---|
| Canonical CSR | token_ids、response_idx、attention_diagonal、response_row_ptr、response_column_indices、response_values |
| 旧稠密缓存 | attention或data [L,H,Q,N]，优先使用已保存query_positions |
| 分头导出 | adjacency、layer、head；同答多个文件按真实通道归并 |
| 可选hidden | hidden/hidden_states [N,D]，或显式选--hidden-layer的[K,N,D]，含全部prompt+response |
| 可选熵 | entropy/next_token_entropy [R]，与预测回答词的位置对齐 |
| 来源分组 | source_groups [P]；source_mask [P]可排除指令等非材料内容 |

默认 `--feature-mode tokens` 训练词项embedding。已有hidden用 `--feature-mode hidden`；
不在attention文件中时用 `--feature-root /已有目录`，按回答编号读取NPZ并核对完整token IDs。
tokens模式没有测得的hidden、Q/K/V或FFN，不会被伪造为已知语义状态。

缺失offset不阻止评分；评价需用原observer tokenizer与保存token IDs精确核对后恢复，
必要时传 `--tokenizer /原tokenizer目录`。编号关联不会被当作内容与offset已核验。
canonical q=P+r预测q+1，因此缺少q=P-1时首个回答词未覆盖，不将相邻分数错移过去。

## 4. 模型实际做什么

默认每行、每head分别保留2条prompt和2条history强边，保留原权重及剪枝质量；
`--edges-per-partition 0`保留全部。图编码按物理层顺序执行，每个head先形成独立门控消息，
然后汇总。这是检测器的图计算，不是原LLM的WV/WO消息重演。

候选片段包含单token及所有覆盖的起点，最长默认32。每段组合以下节点：

| 视图 | 当前选法 |
|---|---|
| 段内 | 区间中全部回答词 |
| 选择候选 | 首尾预测位置和最强来源读取位置，去重；不是已验证的reanchor |
| 证据 | 上述位置读取质量最大的一个source group；无分组时用32-token窗口 |
| 后文 | 后面直接读取过段内词的位置，按时间取最早8个 |

训练配对比较原证据与同材料内长度接近、词项重叠高的另一证据组；另有选择根错接。
只改检测端的配对索引，不改变原生attention。原配对不保证正确，错接也不保证是幻觉。
当前未完整匹配距离/head质量，未实现多事实绑定；证据侧均值汇总也可能丢失关系顺序。
[详细构造、修复与限制](METADATA_AND_VIEWS.md)。

片段异常为负相容logit。无标签校准后，用区间动态规划输出边界及连续token边际。
默认单段最多32词，不暗中合并突破预算；聚集性不是越强越假的标签。

## 5. 输出与比较

```text
graphs/       逐样本端点图、身份、覆盖及剪枝质量
model/        weights.pt、checkpoint.pt、reference.json、training.json
predictions/  samples/*.npz、prediction_freeze.json、evaluation.json
```

评价包含七种token范围、AUROC/AP、source bootstrap、token区间exact/IoU片段F1、
边界误差、片段结束后误报。缺分数为NaN并报告覆盖，不补零。
`--relation none/permuted`、`--future-budget 0`、`--max-length 1`是独立重训对照，
分别用新output。另需检验整段模型是否超过同一单词分数的普通平滑。

## 6. 软件测试

```bash
python -m pytest experiments/unsupervised_token_graph/offline_span/tests -q
```

44项CPU测试通过：原30项回归加14项元数据测试。含真实小模型训练、现有格式、
身份白名单、标签隔离、区间DP对穷举和六字段缓存prepare/fit/score。
尚未读取服务器2497/449个实际文件；测试通过不等于自然检测有效。
