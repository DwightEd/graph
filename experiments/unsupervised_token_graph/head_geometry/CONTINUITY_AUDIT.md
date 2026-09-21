# 冻结分数的连续性审计

本入口检验“检测收益主要落在连续错误片段，以及平滑依赖这种连续性”的解释。
读取现有 predictions，不加载 LLM、不重新拟合参考库、不修改检测分数与阈值。
标签用于审计分层和匹配，不参与原检测器训练。所有结果按 task/generator 分开。

## 一键运行

```bash
git pull --ff-only origin main
bash experiments/unsupervised_token_graph/head_geometry/run_continuity.sh --bootstrap 1000
```

默认读取 `outputs/head_cross_terms_v1`。如果目录不同：

```bash
OUTPUT=outputs/my_cross_terms \
  bash experiments/unsupervised_token_graph/head_geometry/run_continuity.sh --bootstrap 1000
```

`dataset`、`source_info` 和 `tokenizer` 继承已有 settings，也可显式传
`--dataset /path/to/dataset --tokenizer /path/to/original_observer_tokenizer`。
有已核验 offsets 时不加载 tokenizer；没有时仅加载原 tokenizer 做精确对齐，不读取模型权重。
旧 geometry 冻结结果也可读取，但缺少的 cross-term / smooth 对照会明确报告 unavailable，
不从已有校准分数补造“原方法”的平滑告警。

默认审计全部 `all__` 方法，使用共同有限覆盖和有效文本 offsets；
`--methods all__pair_state all__pair_state_smooth` 可以明确缩小比较集合。
不同方法集合的共同覆盖可能不同，不能直接混用其分层指标。

输出在输入目录的 `continuity/`，自动打包为同级的
`head_cross_terms_v1_continuity_review.tar.gz`。旧 predictions 和旧 review 包不改动。
进度条分别显示正常 span 配对、方法统计和来源级配对比较。

## 直接回答五个问题

### 1. 数据是否主要由后续幻觉 token 组成

依据官方字符区间和原 tokenizer 的 offsets，合并重叠到同一 token 的标注，
保留相邻但不重叠的标注。每个映射片段第一个 token 为 onset，其余为 continuation。
报告完整标注与共同可评分子集各自的比例、长度分布、阳性连续 run 数。
字符标注不是自动按整句扩展，不能把“一个标注片段”直接当作一句话。

这只是当前冻结回答集合的统计，不能以它代替 RAGTruth 全库比例。
正常片段长度和结构控制之外，本入口没有声称完成词性、语义或事实角色匹配。

### 2. 总体排序增益有多少落在后续位置

所有方法用同一正常负例集合、同一正例覆盖，将正例拆成四个互斥组：
onset、offset 1–3、4–7、8+。对于两个冻结方法，精确满足：

\[
\Delta\mathrm{AUROC}_{all}
=\sum_g \frac{N_g^+}{N^+}\Delta\mathrm{AUROC}_{g\;vs\;normal}.
\]

保存每组人数、权重、两侧 AUROC、差值、加权项、和及残差。
continuation 另按真实正例数正确汇总，不简单平均阶段 AUROC。
AP 没有这个加法分解。增益小于零或接近零时，也不硬报一个“解释百分比”。

区间通过 source 配对重采样计算，各次重新计算正例权重。
抽样后整体仍有正负类但某阶段无人时，该阶段的加权贡献是数学零；
不伪造该阶段的 AUROC。整体缺类的抽样不进入有效区间。
此分解是**排序增益落在哪里**，不能称作因果机制的贡献比例。

预设比较包括 smooth−state、full−diagonal、full−persistence、
diagonal−state、full−w1、moment−moment_diagonal、moment−raw。
除总体外报告起点、后续、匹配子集的 pooled 和严格同答差值及 source 区间。

### 3. 是否只是容易区分整条回答，或长错误片段占了多数

同时报告 pooled AUROC、严格同答正负配对 AUROC、同答 macro AUROC/AP、
仅跨答案配对 AUROC。单类答案不能计算答内 AUROC，单列其数量和 token 覆盖。
另报告每个可观测错误 span 总正例权重为 1 的 AUROC，正常 token 权重仍为 1，
用于检查长标注片段主导总体的程度；不把改变类权重的 AP 混入普通 AP。

### 4. 正常的连续片段是否也会被检出

每个错误 span 只匹配同答的一个等长正常区间；首 token 表面类别相同，
相对起点差 ≤0.25，内部 token 重复率差 ≤0.15，此前是否已有错误的状态相同。
有保存熵且两侧值有限时控制首词熵差 ≤0.5 nats；缺熵明确未控制，不补 0。
匹配不读取分数大小，正常区间不重叠复用，双方都要求共同覆盖完整。

默认 `--neighborhood 0` 比较完整片段内部；传正数要求正常片段前后相应范围也没有错误。
规则不自动放宽，未配对原因逐项保存。相同答案与可用正常内容限制了覆盖范围，
结果不代表全错答案或全部 gold spans。
有 token IDs 时用 ID 重复率；没有时使用原 offset 文本片段，明确记录这一较弱代理。
这控制的是词项重复，不等于已经匹配神经状态的重复。

报告错误/正常片段的任意告警率、token 告警率、首词告警、延迟和文本。
延迟统计同时保留完整未检出、缺起点、未完全观测片段数量，避免只报告检出者造成乐观偏差。
阈值始终沿用原混合无标签校准，不声称控制了正常 FPR=5%。

### 5. 平滑是否依赖相邻排列

在同一回答的连续共同可观测区间内，联合重排 `(单点分数, 标签)`，
保持单点 AUROC/AP 和答案身份不变，再对分数计算相同长度的因果均值。
默认 20 次，可用 `--permutations` 修改；保存每次增益与阳性相邻率。
缺失位置保持为缺口，窗口在缺口重置，真实零分数仍是有效观测。

若原序平滑有明显增益而重排后消失，说明该平滑收益依赖相邻排列。
这也改变了精确位置效应，不能单独证明幻觉特有的神经重复。
此处是离线空模型，不是生成新回答、模型内部干预或新的在线检测器。
它对已校准单点标量重新平滑，不能当作原冻结 smooth 分数的数值复现；不产生新告警阈值。
空模型的 95% 范围是重排分布范围，不是来源抽样置信区间。

## 结果文件

|文件|用途|
|summary.md|先读：分任务总表、连续标签比例、平滑增益分解、顺序空模型|
|decomposition.csv|每个预设对照的阶段权重和 ΔAUROC 加权项|
|comparisons.csv|共同覆盖下 pooled / 同答配对差值和 source 区间|
|metrics.csv|总体、首错、起点、后续、前后半段、窗口阶段、匹配子集|
|annotations.csv|标注片段长度分布和 continuation 比例|
|matching.json|配对文本、匹配差异、缺熵及未匹配原因|
|spans.csv / matched_spans.csv|逐片段报警、覆盖、漏检与延迟|
|profiles.csv|全部错误、匹配错误及匹配正常片段的相对位置曲线，各位置分母|
|answers.csv|逐回答的分层排序，用于定位 pooled 与同答差异|
|audit.json|完整统计，包括逐次顺序空模型和 span-balanced AUROC|
|tokens.npz / answers.json|共同覆盖的评分、既有告警、标签、回答身份及 offsets，供复核|

## 代码分工及解释边界

`continuity.py` 只编排读取→共同覆盖→匹配→统计→保存；
matching / metrics / decomposition / order 是纯计算，reporting 负责文件。
读取复用 EvaluationBinding，排序复用 Ranking，匹配复用 span_audit 的跨度与词项工具，
时间均值复用 cross_terms，不另建缓存格式或训练流程。

本次测试检验数学恒等式、来源 bootstrap、缺测、相邻标注、匹配独立于分数、
无未来平滑、三任务入口和归档、原预测文件不变。没有新的自然数据检测结论。
当前测试集反复用于探索，所有区间未做多重比较修正。

只有“增益主要在 continuation + 同答/正常匹配控制后仍存在 + 顺序空模型显著减弱增益”
共同出现，才更支持利用错误的持续模式。仍需逐头状态数据才能进一步声称重复的是哪些头状态。
