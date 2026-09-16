# 金标片段机制比较

**用标签直接比较错误和正常生成。不训练模型，不枚举检测窗口，不输出幻觉概率。**

先读 [实验要求](../../../iclr/MECHANISM_FIRST.md) 和
[代码分工](../../../iclr/MECHANISM_IMPLEMENTATION.md)。

## 运行

```bash
python -m experiments.unsupervised_token_graph.span_audit \
  --split train \
  --dataset /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset \
  --output outputs/span_mechanism_train_v1 \
  --resume
```

默认根据split读取现有 `RAGTruth/attention/llama31_8b/{train,test}`。
`--cache`可指向另一个目录或单个NPZ。默认覆盖QA/Summary/Data2txt所有任务。
`--tasks QA Summary Data2txt`可显式选任务。来源、任务、生成器从原response/source_info关联。
六字段cache没有source_id也不需要先造索引；已有inputs/records可用`--index`补充。
没有offset时用原observer tokenizer核验完全相同的token IDs后恢复，
可从缓存祖先的settings/manifest找路径，否则明确传 `--tokenizer /原本地tokenizer目录`。
只加载tokenizer，不加载LLM权重；不能只凭token数恢复标注。

`--feature-root`可读现有 `<id>.npz` 的entropy与完整token_ids；没有熵就报告没有匹配熵。
`--layers 15 19 --heads 24`可以先验证读取，正式确认不能根据test的差值挑头。
`--limit 10`是按执行顺序的小试跑，不是随机代表性数据；正式全量使用0或不传。
`--window 8`只是首尾观察半径，金标片段本身没有32token长度上限。
参数改变用新output。标签文件修改后也用新目录，防止将旧配对误当成新结果。

## 输出看什么

- `matched_pairs.json`：实际错误/正常区间、文字、匹配差异、未配对数量。
- `samples/<id>.npz`：逐head的两侧原数值和有效行数。`readings`轴为
  `[channel, pair, error_or_control, metric]`；0=错误、1=正常。
- `paired_effects.csv`：错误减正常，先聚合source，再按source给探索性区间。
- `summary.json`：配对覆盖、缺熵数量和解释边界。

`inside_endpoint_excess`不是一般历史占比，而是实际端点质量减去固定距离/词项复现条件后的
随机端点期望。`boundary_excess_change`是该超额读取从段内尾部到结束后的变化。
正常对照执行相同过程。段后仍指向原片段，不在金标终点人为清零。
`onset_*`是金标首词附近的观测，不是自动发现的reanchor；`prompt`不冒称适用证据。

没有配对/缺行/没有段后正常内容时保留缺测；区间不自动判定机制支持与否。
匹配控制长度、同答身份、位置、首token表面类别、token重复率；熵只有存在时才控制。
它尚未控制命题角色、全部语法/词汇差异，也没有判断某条来源是否真正适用。
所有head一起查看是探索性分析，不能把未经多重校正的区间当成多项确认性发现。

已有逐答结果仅重算统计：

```bash
python -m experiments.unsupervised_token_graph.span_audit \
  --phase summarize --output outputs/span_mechanism_train_v1
```

## 主流程一眼看懂

```python
answer = inputs.load_answer(response_id)
pairs = match_controls(answer)
channels = inputs.channels(answer)
channel_ids, readings, counts = measure_answer(answer, pairs, channels, window=8)
save_answer(output, answer, pairs, channel_ids, readings, counts)
```

科学有效性检查集中在输入和测试。纯计算函数只处理数组，不训练、不访问GPU、不写日志。

```bash
python -m pytest experiments/unsupervised_token_graph/span_audit/tests -q
```

真实服务器的数据未在本地运行。该测试不是新机制或检测效果证据。
