# 离线片段图：当前实现与下一步

更新：2026-09-16。空骨架已替换为可执行代码。
目录：`experiments/unsupervised_token_graph/offline_span/`。
代码规范：`iclr/CODING_GUIDELINES.md`。

## 当前数据流

```
已有 attention NPZ + 原 token/样本索引
    ↓ ResponseCache / CacheIndex / iter_channels
逐层逐头的端点图 + 材料窗口 + 后续读取索引
    ↓
候选片段及来源—选择—段内—后文视图
    ↓
同材料来源错接、选择根错接
    ↓
EvidenceSpanScorer 对比训练（不读幻觉标签）
    ↓
无标签校准 → 区间动态规划 → 逐token分数和片段边界
    ↓ 保存冻结
原 RAGTruth 身份/offset校验 → token与span评价
```

## 已实现模块

| 文件 | 职责 |
|---|---|
| data.py | 现有三类NPZ、索引、可选hidden的读取，逐样本图保存 |
| graph.py | 分层分头保留实际端点，记录剪枝与缺失质量 |
| regions.py | 全起点区间、候选选择位置、材料上下文、离线后文 |
| controls.py | 仅改变检测侧配对，保留原生attention |
| model.py | 层序图编码及证据/整段关系评分，不是原LLM消息重演 |
| learning.py | source划分、对比训练、完整epoch检查点、混合无标签校准 |
| detection.py | 区间异常、非相邻区间DP及连续token边际 |
| evaluation.py | 冻结后调用原项目七种token口径，增加span/边界/结束误报 |
| run.py | inspect / prepare / fit / score / evaluate / all |

核心数据结构：Sample、TokenGraph、SpanView、MatchedPair、DetectionResult。
原生读取边保存channel/source/query坐标；模型按物理层顺序执行检测端聚合。
layer/head身份不在读入时平均。标签不属于任何训练数据结构。
原LLM完全不加载；tokens模式学习词项embedding，可选hidden模式使用明确缓存的[N,D]。

## 与最初设计的边界

已经完成：可运行的训练—校准—评分—评价链路、真实端点关系模型和片段推断。
尚未完成：原LLM的post-WO/FFN向量分支、通用自然命题解析、显式多个事实因子绑定。
不能从旧attention标量恢复这些信息，也不把学习出来的消息叫作原模型真实消息。

当前证据视图取最强的一个source group；没有分组字段时用32-token窗口近似。
源选择与错接规则还可能被词项重叠、来源遗漏或常见错误规律混淆。能训练不代表已发现机制。

默认最长候选32个token、最多8个后续读者，均为显式资源预算，不是金标长度先验。
无高熵门槛；缺失预测行仍未覆盖。长span、正常复制、纠正与否定必须单独研究。

## 验证情况

30项CPU测试通过，无跳过。包括：dense/CSR/单head合并、既有索引、标签隔离、可选hidden、
真实梯度、三个任务的端到端、阶段续跑、epoch恢复与连续训练一致、DP对穷举、source拆分、
固定完整source权重与token-offset评价。
五个被复用的父模块文件与当前Git blob逐字节一致，没有用模拟reader代替真实接口。

没有在服务器真实数据上训练；没有GPU资源实测；没有新增自然AUROC/AP或机制结论。
软件测试不是自然有效性证明。

## 下一步执行顺序

1. `--phase inspect` 核对原缓存字段，尤其source_id、split、offsets和可选hidden。
2. 小批验证实际吞吐、剪枝质量与配对数量，再运行完整任务；不新增大模型前向。
3. 冻结后比较token/span表现，优先看严格首错后和错误结束后的正常词误报。
4. 对比真实图/无边/重接图、前缀/后文、单词/片段；各自用新output和同一source拆分。
5. 再补完全匹配的距离对照、同一单词分数仅平滑，以及原生消息向量分支；不根据test选方向。

运行命令与字段详情见 [模块README](../experiments/unsupervised_token_graph/offline_span/README.md)。
原有main.py、SourceFlow、FlowTracer、旧自编码器和缓存不修改。
