# 重锚节点：计算定义与本轮真实样本审计

本轮实际重新计算了上传存档中的四段原生 attention 前缀，每段 32 层×32 heads。
这四段只有 top-8 路由及一个局部 claim 的人工支持性核验，**没有完整回答的官方幻觉 span 标注**。
因此可以检查具体节点和已核验 claim 的时间关系，不能据此报告全量 RAGTruth 的发生率。

## 定义已修正

同一个 head，过去 3 步的平均读取以局部 10 个普通回答 token 为主，当前转为旧来源为主。
旧来源是普通 prompt token 与更早回答 token；全部特殊 token 排除。
要求旧来源增加和局部减少同时至少 0.10 attention 质量；同一 head 连续满足时只留首个确认位置。
同时保存 0.05 / 0.10 / 0.20 三档，节点计算完全不读幻觉标签。

比较各行时固定当前时刻的来源集合；不因 token 变老造成虚假切换。
不把删除特殊 token 后的剩余质量归一化。稀疏边缺失量作为未知质量，
只有最不利分配下仍满足全部不等式才确认节点，否则区分“能排除”与“未知”。
这确认的是结构切换，尚未确认读取的是下一条事实所需的证据，更未确认因果整合。

完整公式及服务器运行方法：`experiments/unsupervised_token_graph/span_audit/ONSET_AUDIT.md`。

## 实际结果

默认 0.10 阈值共确认 **79 个不同 token 位置**，涉及 226 个 head 事件。
“supported/unsupported”只指指定局部 claim，不代表整个历史完全正常／错误。

| 前缀 | 全前缀确认节点 | 指定 claim 前 8 步 | 指定 claim 决策步 |
|---|---:|---|---|
| 14315_headwear_scope / supported | 9 | 有：`full`、`,`、`and` | 未知 |
| 14315_headwear_scope / unsupported | 10 | 有：`metal` | 未知 |
| 14375_onion_stage / supported | 29 | 未知，未确认节点 | 未知 |
| 14375_onion_stage / unsupported | 31 | 未知，未确认节点 | 未知 |

在本存档的稀疏程度下，其余位置都至少有某些 head 不能排除潜在切换，因此总体没有可确认的“无节点”位置。
这不是“所有 token 都重锚”，而是 **79 个确认有，247 个未知**（含四个首词预测位置）。
洋葱两侧的“未知”不能解释为“未发生”。

| 阈值 | 头饰 supported | 头饰 unsupported | 洋葱 supported | 洋葱 unsupported |
|---|---:|---:|---:|---:|
| 0.05 | 10 | 15 | 34 | 38 |
| 0.10，主结果 | 9 | 10 | 29 | 31 |
| 0.20 | 6 | 5 | 17 | 19 |

三档下，头饰两侧在 claim 前 8 步均有候选，洋葱两侧均未知。
同一 head 连续段的首次确认位置可能随阈值改变，所以不同阈值的节点位置不必互为子集。

## 具体读回了什么？

以下 node_token 和 target 都是回答内 0-based 下标。node_token 是进行读取的词，target 是它预测的下一词。

| 前缀 | node_token / 节点词 | 预测词 | claim 决策位置距离 | 实际来源举例 |
|---|---|---|---:|---|
| 头饰 supported | 27 / `full` | `-length` | 7 | L6H29 → prompt 的 `length`，位于裙长描述 |
| 头饰 supported | 29 / `,` | `and` | 5 | L17H30 → 历史的 `wore` |
| 头饰 supported | 30 / `and` | `added` | 4 | L23H21、L12H17 → 历史的 `wore` |
| 头饰 unsupported | 31 / `metal` | `pins` | 3 | L20H3 → prompt 的 `metal`，位于金属别针描述 |

最后一行的实测量：

| 量 | 数值 |
|---|---:|
| 过去旧来源 / 局部已保存质量 | 0.193125 / 0.342224 |
| 当前旧来源 / 局部已保存质量 | 0.625458 / 0.094543 |
| 过去 / 当前未知质量 | 0.137828 / 0.026093 |
| 双向切换幅度保守下界 | 0.221588 |
| prompt `metal` 单端点增量下界 | 0.390004 |

即使把未保存质量作最不利分配，它仍符合本轮定义。
但它属于 `metal→pins` 这部分生成，且是同 token ID 回读；已列出的近 claim 来源均在核验的头饰证据范围之外。
**它能证明错误 claim 前有结构回看，不能证明模型在这个节点重新读取了头饰所需的证据。**
正确局部 claim 前同样存在节点，所以“存在节点”本身不区分这对 claim 的支持性。

## 正向和反向问题目前各回答了多少？

- 节点→后续幻觉：头饰 unsupported 的 `metal` 节点后 3 步开始已核验的不支持 claim。
  其余位置的完整后续标注未提供；不能把其他节点算成“后续没有幻觉”。
- 幻觉→前置节点：两个已核验不支持 claim 中，头饰前确认有节点，洋葱前未知。
  不能写成 1/2 命中率，更不能写成所有幻觉前都有节点。
- supported 对照：头饰前也确认有节点；洋葱前未知。仅有两对，且自然前缀及 claim 措辞不同。

全量入口已经实现两方向统计、正常位置对照、回答结束删失、全部阴性与未知。
当前工作区没有全量原 attention；完整 RAGTruth 结果仍需在服务器缓存上运行。

## 文件与复现

- `review.html`：显示整个回答前缀并标出节点；展开每个节点可看具体来源词和上下文。
- `nodes.csv`：全部 79 个节点及已核验 claim 关系，不只保留靠近错误的节点。
- `coverage.csv`：三档阈值的确认／排除／未知数与实际 head 数。
- `claim_association.csv`：四个已核验 claim 的反向查询。
- `samples/*/head_events.csv`、`source_reads.csv`：逐 head 的原始质量、缺失量界和具体来源。
- `samples/*/detection.npz`：全部 head 的逐位置状态、连续段入口、token ID；在标签关联前写出。

输入来自 `reanchor_review.tar.gz` 的原生 routes.npz/context.json。
特殊控制 token 从存档保存的逐 token 解码核验，实际出现的 IDs 为 128000、128006、128007、128009；
没有用词表数值截断来猜特殊 token。全量入口用原 observer tokenizer 的 all_special_ids。

复现本报告：

```bash
python -m experiments.unsupervised_token_graph.span_audit.onset_run \
  --review-archive 已解压的reanchor_review目录 --output outputs/reanchor_prefix_nodes_v2
```

软件验证：53 项测试通过，包括旧来源内部换位、恒定全局 head、特殊 token 质量变化、
移动距离边界、缺失质量界、无未来读取、节点/预测词对齐、标签不影响节点、双向关联、
右删失、全正常且无事件的回答、完整 HTML/表格输出。自然样本结果来自上述四段真实缓存，并非这些测试构造。
