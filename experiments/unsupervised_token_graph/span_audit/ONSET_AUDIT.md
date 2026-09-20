# 重锚节点计算与双向标注审计，v2

这次先独立扫描整个回答，包括全正常回答，再读取标签检查节点和幻觉 span 的关系。
不从幻觉起点向前挑最大变化，不用正常／错误标签设定节点阈值。
`v1` 的“旧来源正增量之和”已退出该入口；它会把旧来源之间的换位算成事件，不能作为本定义。

## 1. 本轮明确检验的定义

**结构重锚候选：同一 head 从局部读取占优，切换到旧来源读取占优，并且两方向质量变化都足够大。**
这不是“正确证据已被整合”的定义。始终全局读取的 head、全局来源之间换位，都不属于这个切换。
若真实机制没有占优切换，本定义会漏掉它；所以阴性只否定这一明确模式，不能否定所有可能的重锚机制。

设 prompt 长 P，query 位置 q。该节点是回答 token `q−P`，预测回答 token `t=q+1−P`。
token 下标均从 0 开始。节点词和它预测的词同时输出，避免把接收信息的位置错写成下一词。

排除全部特殊 token 后，令局部集合为：

    L(q) = 已生成回答中 q−10 < j ≤ q 的普通 token
    R(q) = 普通 prompt token ∪ 回答中 j ≤ q−10 的普通 token

比较当前行与过去 3 行的平均值。**这 4 行使用当前 q 定义的相同 L/R 集合**，
防止一个 token 仅因距离变大被划到旧来源，就制造虚假的重锚。

    L_now = Σ[j∈L(q)] A[q,j]             R_now = Σ[j∈R(q)] A[q,j]
    L_before = mean(Σ[j∈L(q)] A[q−b,j], b=1..3)
    R_before = mean(Σ[j∈R(q)] A[q−b,j], b=1..3)

    shift = min(R_now−R_before, L_before−L_now)

必须同时满足：

    L_before > R_before
    R_now > L_now
    shift ≥ 0.10

0.10 指原 attention 质量，不是概率置信度或 nats。这个操作阈值在看结果前固定；
同时保存 0.05 / 0.10 / 0.20 三档，不挑结果最好的一档。
不重新归一化剩余普通 token，避免特殊 token 释放质量造成比例上的虚假局部下降。
同一 head 连续满足条件只保留首个确认位置；原始逐行成立状态也保存在 NPZ 中。
未知行会中断“连续确认”段，后一节点是新的确认段入口，不保证是完整真实轨迹的第一次切换。

## 2. 稀疏 attention 的未知部分不能补零

设保存质量为 m，缺失质量 u=max(0,1−m)，特殊 token 已保存的质量仍计入 m，之后才从读取集合排除。
旧／局部实际质量分别在 `[observed, observed+u]` 内。
例如，旧来源增长的下界是 `R_now_observed−R_before_observed−u_before`；
局部下降的下界是 `L_before_observed−L_now_observed−u_now`。
占优关系也用最不利缺失质量检验。

- 所有下界仍满足定义：`state=1`，确认有结构候选。
- 至少一个必要条件的上界也不可能满足：`state=0`，排除该位置符合定义。
- 其余：`state=-1`，未知。缺基准行、特殊 query 行同样记未知。

这是保守边界，不恢复已丢失 attention。小于 0.005 的行质量超 1 数值舍入统一缩回 1。
总体节点只需某个 head 确认成立；判定没有节点则需要所有输入 head 排除。
这里的“所有”指实际读取的缓存 head；`detection.npz/head_ids` 保存完整观察范围，不推断缺失 head。

## 3. 节点具体读取了哪些 token

每个确认节点保留全部触发的物理 layer/head，不平均头。
对该 head 找到累计覆盖 80% **已观测旧来源正增量**的来源位置，输出：
来源 token ID、token 文本、上下文、prompt/history、当前质量、过去质量、增量及其保守下界。
总体切换可以确认，但某个具体端点的增长仍可能不能确认，所以 `source_gain_low` 单列。
同时标出是否只是回读同 token ID，以及来源是否曾是一个更早的候选节点。
“曾是候选”不自动叫作功能性 hub；原生消息利用和因果作用是另一个待检验问题。

## 4. 双向关联，不能只报成功例子

默认 H=8。所有节点先冻结，再使用金标：

| 方向 | 具体输出 |
|---|---|
| 节点→幻觉 | 预测位置 t 后 `(t,t+H]` 是否有新 span 起点；最近起点距离；当前是否已在幻觉内 |
| 幻觉→节点 | onset 前 `[onset−H,onset)` 是否有节点；具体节点 token；首错决策步 `t=onset` 单列 |
| 正常比较 | 每个 span 找同回答、相似位置、同表面词类、此前是否有错相同的正常起点，比较相同历史窗口 |

反向统计包含全部标注 span，区分有、无、未知。相邻标注和真正连续正标签的入口分别记录。
正向统计保留所有非事件位置作为对照；回答结束导致观察不足 H 步时记右删失，不能算无幻觉。
forward_summary 只用完整后续窗口，按当前标签分组，避免“已经处在错误里”混入“将出现新错误”。
这些是观察关联，不能因节点位于错误之前就称它导致错误。
head 越多越容易出现某个节点，因此正常对照也用完全相同的 head 集合与节点定义。
相邻窗口和匹配样本相关；配对差按 source 聚合、重采样，不把 head 当独立样本。

## 5. 运行与查看

服务器上一次运行全部已有缓存：

```bash
git pull --ff-only origin main
python -m experiments.unsupervised_token_graph.span_audit.onset_run
```

默认原 RAGTruth 路径、train/test、三任务、全部缓存 head。只读取 tokenizer 和 attention，不加载 LLM。
可用 `--tasks QA` 缩小任务；路径不同加 `--cache ... --dataset ... --tokenizer ...`。
恢复相同设置加 `--resume`，输出 `outputs/reanchor_nodes_v2`。

| 输出 | 怎么看 |
|---|---|
| review.html | 回答中标出具体节点，列出来源 token、span 对应；全量输出较大，逐答原表也保留 |
| nodes.csv | 每个节点 token、预测 token、触发 head、之后的 span 信息 |
| node_span_links.csv | 节点与标注起点的具体对应，区分提前和首错决策步 |
| spans.csv / reverse_summary.csv | 每个幻觉 span 前有／无／未知，以及匹配正常位置比较 |
| positions.csv.gz / forward_summary.csv | 所有事件／非事件位置的后续 span 发生率与删失标记 |
| samples/ID/head_events.csv | 同一节点的各 head 切换质量和未知质量上下界 |
| samples/ID/source_reads.csv | 实际读取哪些来源 token，是否同 token 回读 |
| samples/ID/detection.npz | 标签介入前的逐 head 状态、连续段入口状态、三档阈值、原始 token ID |
| population_coverage.csv | 官方目标回答数与缓存缺失，不把缓存子集冒充全量 |

现有四个原生前缀也能复核：

```bash
python -m experiments.unsupervised_token_graph.span_audit.onset_run \
  --review-archive 已解压的reanchor_review目录 --output outputs/reanchor_prefix_nodes_v2
```

该存档只有 top-8 边和局部人工核验的 claim，没有完整 RAGTruth span 标注。
它输出 claim_association，其他历史／后续 token 保持未核验，不能用它估计总体幻觉率。
这里按存档的逐 token 原生解码明确列出特殊控制 token ID；全量模式必须使用原 tokenizer 的 all_special_ids。
