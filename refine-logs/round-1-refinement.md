# 候选状态与路径端点的条件对齐：完整修订方案

## Problem Anchor

目标是在不训练新神经检测器、不使用幻觉标签构图或定向评分、不把模型消融作为前提的条件下，从自然生成轨迹中读取带连接关系的来源／路径 × 候选 × 层 × 通道差异，用于无监督幻觉检测。必须超越角色总量和全局谱熵的汇总；最终成功标准是来源隔离的自然数据上，相同输入信息、相同无监督读出下，连接结构带来可重复增益。软件通过测试不等于达到此研究标准。

## 问题、依据与路线选择

核心假设：**候选状态与真实边端点的对齐，在扣除角色、来源单元、距离区间和自注意力的分配后，仍保留可用于幻觉检测的异常模式。** 这是待验证假设，尚未获得自然数据性能支持。

高 attention 峰和全局谱熵无法单独回答“端点状态支持哪个候选”。下三角 attention 的原始特征值只取决于对角线，是选择 operator action 的数学动机；不针对对称 Laplacian 的谱／TOHA 作否定性推断。

相关工作的已知贡献与我们的边界：

| 一手文献 | 已知能力 | 本方案不能声称的新意 |
|---|---|---|
| [CHARM](https://arxiv.org/html/2509.24770v2) | 用节点激活和 attention 边训练图检测器 | 图和激活的组合本身 |
| [TOHA](https://arxiv.org/html/2504.10063) | 拓扑差异检测，也讨论无标注选头 | 无监督 attention 拓扑本身 |
| [Geometric Scattering](https://proceedings.mlr.press/v97/gao19e.html) | 固定图滤波与多尺度表征 | 无训练图算子替换 AE 本身 |
| [Attention Rollout](https://aclanthology.org/2020.acl-main.385/) | 残差近似与跨层 attention 传播 | 多层矩阵连乘本身 |
| [事实召回阶段](https://aclanthology.org/2023.emnlp-main.751/)、[幻觉阶段分析](https://aclanthology.org/2024.findings-emnlp.466/) | 候选富集与提取的机制区分 | 把状态可读性直接宣称为知识存在或因果支持 |

比较通用 scattering 与条件路径残差后选择后者：它把可检验贡献收紧到“边与候选信号的条件关系”，而不是加入更多滤波器。本库历史中高阶路径没有稳定增益，rupture 曾退化为位置指标，AE 重构变好也未证明检测变好，因此不使用累计前缀分数或预设熵的正确方向。

## 输入与精确定义

RAGTruth 原始 prompt 和响应经本地冻结 Llama teacher-forced replay。它是 probe model，不保证就是产生数据的原模型，也不声称还原原 chat template。prompt 保留原文，加 BOS；response 单独分词后追加。只支持当前可验证的 QA、Summary 原始 source 字符范围。

预测 y_t 时 q=prompt_tokens+t-1；包括 t=0。所有候选、信号、边仅使用节点 0..q。C_q 是该位置最终 logits 的 top-K，默认 K=2。不拿真实响应或正确答案补全候选。

令 h_l(i) 为第 l 层的输入（l 从 0 开始），F 为冻结 final norm，u_c 为冻结输出 embedding 的第 c 行。固定信号：

    x_l(i,k) = cos(F(h_l(i)), u_Cq[0]) - cos(F(h_l(i)), u_Cq[k]), k=1..K-1.

保留符号与候选排名。这是 cosine logit-lens 对比，与最终 logit margin 不等价；它也不是经过 value/O/MLP 的真实消息贡献。labels 可验证生成 token 是否在候选中；正确答案 alias 的覆盖目前没有标注，不能用生成 token coverage 替代。该覆盖统计只在冻结分数后计算。

角色为 source、instruction、history、special。source 单元是原文中以空行分隔的 passage/paragraph；边界判定是结构规则，并不声称得到语义证据块。没有段落边界就只有一个 source 单元。

## 强条件零模型

A_lh[i,j] 是 i 从 j 读取的权重。每行严格过去 j<i 的端点按以下条件分组：

    (role(j), source_unit(j) if source else fixed_role_unit, floor(log2(i-j))).

在每个非空组内均匀重分配其原有 attention 质量，得到 N_lh；A[i,i] 完全不变。单元素组 N=A，不合并、不猜补；空组质量为零。该操作保留因果 mask、节点数、每行 role/source-unit/log-lag 的质量，以及自注意力。它不保留每个 lag 的精确距离或每列入度，所以仍需位置对照，不能声称消除所有 confound。

N 是“给定当前 X 后，各层各行各条件组独立置换原 attention 权重”的解析期望；这是冻结图上的分析参照，**不是**重新运行网络干预后 attention 的期望。对跨层乘积的解释也仅限这个独立置换模型。没有使用 p 值或宣称一般随机图显著性。

第一轮把 role 内 cardinality 也列为未保留量：原方案已经保留节点集合／数量，故这一点不接受；lag 与 passage 混杂的批评接受并已强化零模型。为避免实现分支，当前主实现只保留强零模型，不同时堆三种可选零模型。

## 有序路径算子

P_lh=(I+A_lh)/2，Q_lh=(I+N_lh)/2。残差混合 1/2 是固定分析 convention，不是对真实 residual stream 的能量估计。前序层对 head 等权平均，最终 head 保留为显式通道；不声称追踪特定跨层 head 到 head 的真实组合。

默认深度 D={1,2}、最终层 L-1、全部终点 heads、K=2。层选择按结构预声明，不用标签挑层；CLI 支持显式层索引，改变时必须记录并另立实验配置。

    R1(l,h,r,k) = e_q^T (P_lh-Q_lh) M_r x_l(:,k).

    R2(l,h,r,k) = e_q^T P_lh P_(l-1,mean) M_r x_(l-1)(:,k)
                  -e_q^T Q_lh Q_(l-1,mean) M_r x_(l-1)(:,k).

    R2relay(l,h,k) = e_q^T P_lh M_Hminus P_(l-1,mean) M_S x_(l-1)(:,k)
                      -e_q^T Q_lh M_Hminus Q_(l-1,mean) M_S x_(l-1)(:,k).

M_Hminus 排除 q 本身；因此 prompt 最后一个位置及首个生成 token 都不能被冒充为之前的 history relay。r 分别为 source/history。默认特征数是 5 × heads × (K-1)，例如 32 heads 时 160 维。保留对应 observed、null 和角色信号均值向量，不提前把四者加成任意手工总分。

更高 depth 可用于显式实验，但不构成默认贡献；每一步沿真实层顺序，绝不在同层矩阵上任意取幂。若 d=2 无收益，撤回中继贡献。

## 无监督读出

参考与测试 source_id 完全隔离，task/generator/log2(prompt_tokens)/log2(t+1) 为固定匹配条件，绝不使用响应总长度匹配。各条件内每个来源取相同数量确定性样本（数量是来源最小可用数与 cap=8 的较小者）。不挑正常标签；参考需要代表测试分布，其异常不必然是幻觉。

主分数：参考中位数与 MAD 标准化后 mean(z²)，尺度为 max(1.4826 MAD,1e-6)，方向固定为越大越异常。四种表征使用完全相同规则。补充 kNN 分数：对每个不同参考来源取最近样本距离，再平均最近 k=3 个来源距离。条件中不足 k 个参考来源时明确失败，不跨任务／位置条件回退。

参考分布稀疏、同组大量 singleton、候选不可用、参考污染均须报告为适用性边界。默认只读最终层的选择牺牲部分早层诊断能力，避免全层全头超高维读出的解释自由度。

## 评价与三个验证块

1. 构造性／软件验证：相同 role/source/lag 质量与原始谱而候选对齐不同；singleton 组不制造 residual；角色常数信号一阶 residual=0；显式 d=2relay 方向正确；q 自环排除；真实 tiny Llama 首 token 与 target/future invariance；数据标签不进入准备选择、提取、参考拟合或评分。
2. 固定自然数据主比较：按 source 隔离。冻结全流评分后，用字符 span 连接 RAGTruth 标签，评价全 token、所有 span onset（含首 token）、response first error、continuations；source-balanced AUROC/AP、来源 bootstrap，并报告 residual−null 与 residual−signal 的配对差。entropy/negative-margin/position 是强对照，不允许按测试结果翻转分数。
3. 最小机制验证：相同特征与来源条件下 d=1 vs d={1,2}、K=2/4/8、不同预声明层/模型/任务；连到候选状态的增量必须超出 null/signal 和不确定性基线。若强零模型下没有稳健增量，不发表“捕捉到提取失败”的 claim；若候选遗漏严重，不以此区分 enrichment/extraction。

不把无标签 kNN 当成真值判定器；不宣称已分类 enrichment/extraction。这两类机制是设计问题的来源，当前直接建模对象是局部候选状态与路由的关系。

## 工程方案与限制

main.py 的默认流程为 prepare → extract → detect → evaluate。旧 factorial/onset 审计移到有名字的 control_graph 基线入口；真实 consumer 保留，避免复制成假兼容层。

route_graph/data.py 负责无标注准备及 JSONL 边界；capture.py 负责本地冻结模型；operator.py 负责所有路径数学；detector.py 负责无监督参考与评分；evaluation.py 负责独立标签连接与统计。参数从 main 显式传入。

完整 attention 当前仍是稠密提取，设置 max_tokens 与 attention 存储估计上限，不静默截断或漏样本。估计不含模型、logits 与暂存，不是 OOM 保证。保存层/路径/候选配置、权重文件 digest、特征 digest 和完整性 manifest。旧 compact v3 不足以恢复 token 连接，不能转换成新的自然图证据。

当前状态：局部算子与 tiny Llama 软件执行已经验证；真实预训练权重上的检测增益尚未验证。研究 READY 与实现可执行是不同结论。
