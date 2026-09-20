# 标注起点之前是否普遍有明显回看变化？

本入口回答发生率问题。旧 `experiments.reanchor_audit.run` 的两个事实案例和路径干预不回答这个问题。
不训练检测器，不以金标首词定义重锚，也不把 attention 最大来源称为所需证据。

```bash
git pull origin main
python -m experiments.unsupervised_token_graph.span_audit.onset_run
```

默认扫描既有 `RAGTruth/attention/llama31_8b` 下 train/test 全部缓存、三任务、全部物理 head。
tokenizer 自动使用既定服务器目录；目录不同请加 `--tokenizer 原observer模型目录`。
只需要 tokenizer 与既有 attention NPZ，不加载 LLM、不重新生成。不限制回答数。
中断后同一命令加 `--resume`。输出默认 `outputs/pre_onset_audit`。

## 一次只检验这个明确的结构假设

对于预测回答 token t 的 query q=P+t−1，比较相邻两行在**同一个来源集合**上的 attention。
来源包括普通 prompt token，以及距离 q 至少 10 个位置的历史 token。
用 tokenizer 的 **all_special_ids** 同时排除来源特殊 token；比较行或目标本身为特殊 token 时记缺测。
不把去掉的特殊 token 或稀疏丢弃质量重新分摊给普通 token。

逐 head 记录普通旧来源 attention 正增量之和：

    gain(q,h) = Σ普通旧来源j max(A[q,h,j] − A[q−1,h,j], 0)

两行使用当前 q 确定的相同来源集合，所以仅由 token 距离增加不会制造正增量。
这个量也能发现“prompt 总质量不变，但从某些旧来源转向另一些旧来源”。
它是结构上的回看变化候选，并不证明重新获得了正确证据。

两个时间范围严格分开：

| phase | 使用的预测 token 位置 | 含义 |
|---|---|---|
| strict_before | onset−8,...,onset−1 | 严格早于首个标注 token 的预测步骤 |
| onset_decision | onset | 即将生成首个标注 token 的步骤 |

不使用 onset 后的行选择“提前事件”。缺任一必需行时记缺测，不补零。
原缓存往往不含 q=P−1；因此回答开头的部分起点不可观测，会列入分母。

## 正常参考和“明显”的固定定义

正常参考取同回答、同首 token 表面类别、相对位置差≤0.25、此前是否已有错误相同的位置。
正常位置前后 8 个 token 以及变化计算的前一基准行不得包含金标错误。
每个起点固定取最近的合格正常位置作对照，不依据其 attention 选对照。
剩余至少 20 个有效正常窗口用于描述性分位数比较。

“明显”要求变化量同时严格超过普通正常窗口的 95% 分位数和原 attention 单位的 0.05。
两个数是公开固定的操作定义；同时输出连续值和 percentile，不能把这个定义当成自然定律。
参考窗口可能重叠，所以 percentile **不是显著性检验 p 值**，也不是独立校准的 5% 检测 FPR。
这是起点比较，未声称匹配事实角色、句法或整段重复度。

每个 head 单独保存；总体起点判定采用所有请求 head、所有窗口步骤的最大值。
**正常参考也取同样的多头、多步最大值**，不把每个 head 的 5% 尾部直接做并集。
不同 head 的量级可能不同，因此同时保留逐 head 文件，不能只凭总体最大值否定小幅单 head 效应。

## 看哪些文件

| 文件 | 内容 |
|---|---|
| population_coverage.csv | 官方目标回答全集；哪些回答有缓存，哪些完全缺缓存 |
| inventory.csv | 实际缓存回答及其标注片段数、特殊 token 数 |
| onsets.csv | 每一个 token 映射标注起点 × 两种时间范围，包括阴性和缺测；最大变化的 layer/head/来源位置 |
| incidence.csv | 按 split、任务、生成器分别给出发生数、阴性数、缺测数、正常发生率；同 source 配对差和 bootstrap 区间 |
| head_incidence.csv | 每个物理 head 的发生数、正常发生数与各自可观测分母；不据此逐头宣称统计显著 |
| samples/ID/heads.csv.gz | 每个物理 head 的变化、峰值来源位置、参考分位数、普通 token 保留质量 |
| summary.json | 总体覆盖，明确这不是检测器评价 |

主看 `answer_first` 和 `clean_before`；`all` 保留所有标注起点，包含紧邻前段错误的起点。
`run_onsets` 把连续正标签的真正入口单列，避免把相邻两个标注之间的边界当成新错误启动。
阴性和不可观测必须分开；不能只展示成功起点或先取 top-k 案例。

置信区间按 source 重采样，不把 head 或重叠窗口当独立样本。
已有 test 被反复研究，仍是探索性审计。特殊 token 排除、对齐和多头正常比较由单元测试验证；
**完整自然 attention 发生率必须运行本入口后才能报告**。
