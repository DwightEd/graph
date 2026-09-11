# 候选条件的路径残差：初始研究方案（2026-09-11）

## Problem Anchor

目标是在不训练新神经检测器、不使用幻觉标签构图或定向评分、不把模型消融作为前提的条件下，从自然生成轨迹中读取带连接关系的来源／路径 × 候选 × 层 × 通道差异，用于无监督幻觉检测。必须超越角色总量和全局谱熵的汇总；最终成功标准是来源隔离的自然数据上，相同输入信息、相同无监督读出下，连接结构带来可重复增益。软件通过测试不等于达到此研究标准。

## 已核对的依据与负结果

- [CHARM](https://arxiv.org/html/2509.24770v2)：有监督消息传递利用 attention 边及节点激活；说明图有用，但没有证明去掉监督仍可保持辨别力。
- [TOHA](https://arxiv.org/html/2504.10063)：attention 拓扑读出；当前版本也讨论无标注选头。不能把“无监督拓扑”当成我们的新贡献。
- [Geometric Scattering](https://proceedings.mlr.press/v97/gao19e.html)：固定多尺度图滤波早已存在。直接换掉 AE、加入模或多跳不是充分创新。
- [Attention rollout](https://aclanthology.org/2020.acl-main.385/)：残差混合及跨层 attention 乘积已存在；只是近似观测路由，并非真实 value/MLP 计算贡献。
- [事实召回阶段](https://aclanthology.org/2023.emnlp-main.751/)与[幻觉的 enrichment/extraction 分析](https://aclanthology.org/2024.findings-emnlp.466/)：支持把候选可读状态与后续读取分开研究，不能证明 logit-lens 值等于知识是否存在。
- 本库 `docs/EXPERIMENT_HISTORY.md`：高阶路径及 lock-in 未稳定优于一阶；部分 rupture 分数与位置高度相关；重构误差下降没有证明检测有效。

## 路线取舍与核心 insight

A：通用 directed scattering + 无监督 kNN。可实现，但贡献主要是已有组件组合，且扩散阶数容易再次变成长度代理。

B（选择）：**在固定角色分配量和自注意力的条件下，测量候选相关状态与具体边端点的对齐是否异常。** 两段生成可能具有同样的 attention 峰、source/history 比例，甚至同样的原始 attention 特征值，但读取的是支持不同候选的状态。差异是“边与信号的关系”，不只是图或信号各自的统计。

特别地，因果 attention 矩阵为下三角，其原始特征值仅由对角线决定。此陈述不适用于对称化 Laplacian，也不否定已有谱熵工作；它解释了为何本方案直接读取算子作用。先验进入角色条件的零模型和跨层顺序，而不是人为指定“某个熵越大一定越错”。

## 可实现的算子

预测响应第 t 个 token 的位置是 q = response_start + t - 1，包括 t=0。只读取 0..q 的节点与因果边。候选固定为该位置最终 logits 的 top-K（默认 K=2），不用实际响应 token 或未来 token 选候选。

节点角色：source、instruction、history、special。对每个层输入状态，使用冻结模型 final norm 和输出 embedding 方向构造 top-1 对其余候选的余弦差 X；这是候选可读性代理，不是真值或因果贡献。保留正负号和候选排名，不提前合并层／终点 head。

令 A[l,h,i,j] 表示接收节点 i 从节点 j 读取的 attention。定义 N：对每个接收行 i、每个角色 r，把严格过去 j<i 且 role(j)=r 上的权重均匀分配；保留 A[i,i]，其余位置仍为零。因此 N 精确保留每行的角色质量、自注意力、因果掩码。N 是这些过去边权在同角色端点内独立置换的期望；它是分析零模型，不调用模型干预。

P=(I+A)/2，Q=(I+N)/2。选少量相邻层深度 d（默认 1,2），保持真实先后顺序；前面的层以固定 head 均值传播，最后一层保留各 head。d=1 是端点对齐残差；d=2 才引入 source→history→query 的中继结构。

对每个终点层 l、终点 head h、路径与候选：

    observed = e_q^T P[l,h] ... P[l-d+1] M_role X[l-d+1]
    null     = e_q^T Q[l,h] ... Q[l-d+1] M_role X[l-d+1]
    residual = observed - null

路径包括 source 起点、history 起点；d>=2 时另读 source 起点且最后一步经 history 的路径（在最后乘法前加 M_history）。这个中继定义必须排除 q 自身，避免把 self residual 算成中继。

不求全前缀累积、不构造同层 attention 的任意幂、不把高 residual 固定解释成幻觉。保留同维 observed、null、residual 表征及无边的角色内信号均值作对照。相同无监督读出比较它们，才能回答拓扑是否增加信息。

## 无监督读出与评估

按 task、log2(已生成长度+1)、log2(prompt 长度) 匹配参考层；参考与测试按 source_id 完全隔离。每个来源在每个层等量、确定性抽样，以防长响应支配。用参考中位数／MAD 标准化，对不同参考来源取最近距离，再平均 k 个来源距离；方向预先固定为距离大更异常。无足够参考时明确失败，不跨任务暗中回退。

这依赖参考分布具有代表性且大部分为常见行为；不保证异常等于幻觉。candidate omission（正确答案不在 top-K）、logit lens 不忠实、位置混杂残留、source 本身错误都是明确限制。

冻结分数后才读取 RAGTruth 标签，用 token 字符跨度对齐，报告全流 token、N→H（含首 token）、H→H，来源平衡 AUROC/AP 和来源 bootstrap；不靠标签挑检测位置、调正负号或挑层。

## 三个验证块

1. 软件及构造性证据：保持角色质量、自注意力、原始谱不变，只改变同角色端点，残差可改变；常数角色信号一阶残差为零；多层先后次序有效；未来内容和目标 token 不改变过去特征；首 token 覆盖。
2. 真实冻结模型数据：原始 RAGTruth → 无标签准备 → full attention + layer-input states → 固定表征 → source-disjoint reference → 独立标签评估。旧 v3 compact trace 只有角色聚合量，不足以恢复连接，不伪造转接器。局部 tiny Llama 仅验证执行，不报告为有效性结果。
3. 相同信息比较 residual/observed/null/signal 及 entropy/top1-margin/position；源内配对置信区间，多个模型／任务／来源拆分。只有 residual 明确超出 null/signal 才支持新增连接价值；若 d=2 不优于 d=1，撤回中继贡献。正确候选不可用时单列机制解释的边界。

## 工程范围

重构 graph 的默认 main 入口；旧四边 factorial 和 retrospective onset 审计保留为有名字的历史基线命令。新代码采用少量职责清楚的模块：数据准备与标签边界、原生模型提取、固定路径算子、参考检测、评价。使用本地 HF Llama 权重，冻结参数；显式上下文及 attention 内存上限，不下载模型、不静默截断样本。

研究状态：有可证伪的方法假设，尚无新方法自然数据性能。不得宣称已解决 enrichment/extraction 分类、发现因果 anchor，或已证实论文新颖性。
