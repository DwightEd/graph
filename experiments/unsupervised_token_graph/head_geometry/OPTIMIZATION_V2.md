# 从零值修复到可检验的先验

2026-09-21。v2 的实际实现与待研究方向分开记录。

## 已有证据

v1 QA 测试 30,619 个 token 中只有 5,037 个被评分：删除特殊 key 后任一 head 的普通质量为零，
就丢弃整个 token。L0H31 单独涉及 25,382 行；实际连续窗口仅为 1 或 2，没有完整 16-token 窗口。
共同覆盖子集上 raw/contrast/log_moment AUROC 为 .617557/.711791/.712177。
保留头身份的相对构型已有依据，矩阵对数尚无独立增量证据。
head-role 删除 positional 有小幅 AP 收益，但总体 AUROC 未明确超过同数量随机去头。
v2 纠正观测定义和覆盖，分母会改变，不能将旧数值作为新方法预期或纯算法增量对照。

## 1. 注意力不需要严格大于零

softmax 后 a≥0；有限精度、稀疏保存及 mask 都可能出现零。
这与可为负的 attention logits、中心化特征或有符号消息贡献不同。

旧流程在普通 key 集合 O 上计算条件分布：

    m = Σ(j∈O) a_j
    p_j = a_j / m

只有第二行要求 m>0。self=0 不妨碍归一化；m=0 时不能伪造这个条件分布。
v2 删除特殊 key 后保留原子质量，路由直接求和。已保存 query 的普通质量为零时，
self/prompt/history 均合法为零；极小质量不会被除法放大。真实行缺失仍记 NaN。
稀疏缓存中的零仅表示已保存图无正质量，不代表完整原生 attention 严格为零。
特殊 ID 合并 tokenizer 命名特殊词与 AddedToken.special 标记，避免漏掉未列作 BOS/EOS 的聊天控制符。
这也可能使 v2 的排除集合比旧输出更完整，必须记录实际 ID，不能将全部变化只归因于评分公式。

head-role 将条件熵改成质量加权熵：

    entropy_mass = -Σ(j∈O,a_j>0) a_j log(a_j/m) = m H(a/m), m>0
    entropy_mass = 0, m=0

这是明确定义的扩展量，不把“条件熵未知”当作“确定性为零熵”。普通质量 m 同时进入特征，
conditional_defined 单独保存。最大 key 也保留原质量，空普通行取零。

## 2. 已实现：当前 token 的条件协同偏离

近邻假设“罕见构型更可能错误”，可分性并不保证这个方向成立。
新假设是：相对单头异常，多头联合状态对既有条件依赖的偏离有额外检测信息。
正相关、负相关都可以拟合，不要求所有 heads 同向，也不将统计依赖叫作因果协同。

每层保留 raw head 坐标，使用参考集逐坐标 median/IQR 缩放为 x，只用混合无标签参考来源拟合：

    μ = mean_reference(x)
    Σ = mean_reference[(x-μ)(x-μ)^T] + ridge I
    P = Σ^(-1)

每个 head 的全部信号组成向量 x_h。条件高斯给出：

    E[x_h | x_-h] = μ_h - P_hh^(-1) P_h,-h (x_-h-μ_-h)
    Cov[x_h | x_-h] = P_hh^(-1)
    e_h = P_hh^(-1/2) [P(x-μ)]_h

e_h 是预测误差按条件方差标准化后的有符号向量。独立头对照：

    b_h = Σ_hh^(-1/2) (x_h-μ_h)

先保存每层每头 mean(e_h²)、mean(b_h²)，之后才平均为最终分数。没有在建模前抹掉 head 身份。
原始坐标、精度矩阵和逐 head 能量可用于审计；这里的均值中心是训练参照，不是把所有 heads 混为一个。
多信号时整个 head 一起留出，不能让自身 prompt 分量预测自身 self，再冒称头间协同。

不能在 contrast 上做留一回归：减层均值后头间和恒为零，某个 head 会机械地被其他头决定。
条件模型用 raw 坐标；contrast 保留为独立对照。

这两个 energy 分数只使用当前 token，window=1/16 不改变它们；新增项不来自显式时间平滑。
原 moment/log_moment 保留，供完整覆盖后检验时间关系。二阶矩对窗口内行重排不变，
不能拿简单的窗口内时间重排当它的有效消融。

### 解释边界

- conditional_energy 超过 independent_energy，才支持同层条件关系优于单头边际异常。
- 只在 continuation 改善，仍需正常持续片段匹配，不能说解决了起错机制。
- 未改善则保留阴性结果，不用测试标签翻转方向或挑层。
- 混合 TRAIN 也含错误。稳定、常见且协同一致的错误仍可能低分，少见正确状态可能高分。
- 一层高斯是线性近似，跨主题和阶段可能多模态；条件分数不是证据语义或 readout 支持。

## 3. 其他先验及对应实验

| 先验 | 需要测的关系 | 必须有的对照 | 状态 |
|---|---|---|---|
| head 条件依赖 | 留出整头后的标准化残差 | 单步独立 head 能量 | 本轮已实现 |
| 位置/内容行为不同 | 内容交换后跟随位置还是内容 | 两类各自同层同数量随机删头；完整输入迁移 | 新增 symbolic 匹配随机组，保留原 profile/binding |
| 距离、词项复现是正常结构 | 同距离组、同词项复现下的超额端点选择 | 正常等长持续片段；保持边质量的端点对照 | 本轮未接入检测 |
| 读取应符合当前需要 | 独立估计的对象/阶段/范围适用性 vs 实际贡献 | 同值而适用性相反；只换阶段/对象；不能拿大 attention 定义适用证据 | 需要独立语义角色输入 |
| 运输与最终采用不同 | 逐 head 消息在残差/读出方向上的作用 | Q/K/V 分开及联合干预、下游恢复 | 需要 value/residual/原生干预数据 |

多回看、集中、持续都不是固定正确性规则。正常生成会这样，错误也可能高度确定且协同一致。
当前 attention 摘要缺少 value 内容、当前关系适用性和下游抑制等变量。如果这些变量不同，
而保存的特征相同，任何只读取这些特征的分类器都无法区分它们。

### 独立的证据适用性参照

[Information Flow Reveals When to Trust Language Models](https://github.com/rxu0112/RAG-information-flow/blob/main/Information_Flow_Reveals_When_to_Trust_Language_Models.pdf)
将贡献/主要路径排序与 ranker、SHAP 估计的 query relevance 排序比较（simulatability），
并以集中度等训练校准器。可借鉴“实际贡献 vs 独立相关性”的比较；不能把论文完整方法说成无监督。

我们的阶段/对象案例需要比 query relevance 更细的适用性：两条数值证据都可能与问题相关，
只有阶段、对象、范围匹配的那条适用。未来可以利用程序构造的关系对照或预训练语义模型得到独立参照，
但须先验证参照本身。“不使用自然幻觉标签”不等于“没有外部语义先验”。
若要预测 token t，参照只能来自问题、来源和 t 之前的回答前缀；生成后识别则明确检测时刻与延迟。

[Decoupling Positional and Symbolic Attention Behavior in Transformers](https://arxiv.org/html/2511.11579v1)
支持用位置/内容置换定义路由行为；没有证明所有位置型 head 都是噪声。v2 不根据测试 AUROC 挑选固定头或中层。

[Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads](https://arxiv.org/abs/2505.20045)
结合特定 head 行为和 token 置信度的递归估计，可作低成本对照，但不代替适用性审计。
缺 logit_entropy 时不补零，也不能用 attention entropy 冒充 logit entropy。

## 4. 验收顺序

1. 先检查缺行和观测零质量分开、覆盖恢复、完整窗口数及阳性覆盖；特殊 key 仍删除。
2. 相同 source 划分、原坐标、参考行、校准和覆盖，比较全部七种方法。
3. 同时报 onset、first error、continuation、front/back、error→normal 恢复，保存逐 head 分数。
4. 匹配正常/错误持续片段的长度、位置、标点/句界、重复程度，检验条件分数是否仍有增量。
5. 在独立来源确认规则；当前反复分析的 TEST 属探索性，不调方向、不按分数拼接规则。

本轮仅实现观测修复、单步条件残差、symbolic 同数量随机去头和可用性报告。
语义适用性、跨层消息冲突和残差抑制仍是待验证问题；没有新的自然数据 AUROC。
