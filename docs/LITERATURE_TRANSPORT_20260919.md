# 消息运输、多头交互与适用性：文献核查

核查日期：2026-09-19。服务于 graph / reanchor 的实际问题：证据仍在帮助有依据候选，
为什么最终生成可能不采纳它？本次核查下列 18 项相关工作，区分原作者结果、采用的实验原则、
不能直接迁移的结论。方法设计见 [TRANSPORT_METHOD_20260919.md](TRANSPORT_METHOD_20260919.md)。

“原文方法”表示读取了对应方法公式或实验段落，不声称重现作者实验；
“摘要定位”只用于确定研究范围。没有用二手综述替代原论文的计算定义。

| 工作与原始来源 | 本轮核对范围 | 对当前问题有用的内容 | 采用或排除的理由 |
|---|---|---|---|
| [Information Flow Reveals When to Trust Language Models](https://github.com/rxu0112/RAG-information-flow/blob/main/Information_Flow_Reveals_When_to_Trust_Language_Models.pdf) | 作者仓库 PDF，§3–4.5、公式 1–6；目视核对第 3 页 | 从真实 attention/value/output 投影构造来源向量，逆向提取 emergence order，逐层矩阵连乘得到 contribution layout；相关性来自独立 ranker，另训练 calibrator | 采用 target→source 和区分“贡献/相关性”；其来源向量先合并 head，再做非负距离度量，不适合作为有符号 head 竞争的完整量；答案级校准不能直接当无标签 token 检测 |
| [How Does Reasoning Flow?](https://arxiv.org/html/2606.10646v1) | 原文 §3–4 | attention DAG 上的答案可达性重加权、守恒流和 hub，用于 RL token credit | 可借鉴目标条件化；人工构造的非负守恒流不是原生 Transformer 的有符号消息或精确因果分解 |
| [Attention Illuminates LLM Reasoning](https://arxiv.org/html/2510.13554v1) | 原文 §4.1–4.3 | local/global heads、WAAD 和未来影响 FAI 揭示 preplan/anchor 节律 | 保留 head 差异；FAI 读取未来，不直接用于在线首错评分；节律本身没有提供真假方向 |
| [ALTI: Measuring the Mixing of Contextual Information](https://aclanthology.org/2022.emnlp-main.595/) | 原文背景和方法 | 将 value、output projection、残差和归一化纳入局部贡献，而非只读 attention 权重 | 保留原生变换；来源贡献矩阵和最终候选支持是不同对象 |
| [DecompX](https://aclanthology.org/2023.acl-long.149/) | 原文引言、方法动机 | 指出标量 rollout 丢失向量信息，并将 FFN 与分类头纳入向量分解 | 不在聚合前丢弃方向；不能把面向编码器的分解直接声称为 Llama 全后缀的精确分解 |
| [The Hydra Effect](https://arxiv.org/pdf/2307.15771) | 原文 §2–4 | 删除后其他 attention 层补偿，晚层 MLP 可起抵消作用；直接读出与总效应有区别 | 增加下游消息/MLP 恢复；不能把一次删除效应小直接解释成该组件不重要 |
| [AtP*](https://arxiv.org/html/2403.00745v1) | 原文 §3.1、§6–7 | 梯度近似会因 softmax 饱和、直接/间接作用抵消而漏检 | 梯度只用于候选定位，有限干预和未选候选控制仍保留；本实现不是 AtP* 复现，也未给出漏检概率上界 |
| [Have Faith in Faithfulness / EAP-IG](https://arxiv.org/html/2403.17806v2) | 原文 §2–4、附录 D 的定义 | 单点梯度与积分梯度不同；节点重叠不能替代干预后行为保真 | 检验真实 margin，不以“选回同一批 head”验收；两个删除剂量不是 EAP-IG，不冒称积分归因 |
| [Towards Best Practices of Activation Patching](https://arxiv.org/pdf/2309.16042) | 原文方法与结论 | 替换方式和评估指标会改变机制定位结果 | 保留同世界恢复、等范数随机方向、绝对候选概率与完整候选指标；随机方向只是幅度控制，不能充当语义反事实 |
| [How do Language Models Bind Entities in Context?](https://arxiv.org/html/2310.17191v2) | 原文方法和 binding ID 干预 | 内容与实体—属性绑定可以具有不同的表示结构；绑定需要因果检验 | 值被传入不能证明它对当前对象适用；不能由某个来源位置的删除量定义“绑定完整度” |
| [Monitoring Latent World States with Propositional Probes](https://arxiv.org/html/2406.19501v1) | 原文 §3–6 | Hessian 提取低秩绑定子空间，组合 lexical/domain probe；受控场景中可读出的世界状态与错误输出分离 | 为关系读出提供具体范式；它有程序构造的语义监督和有限关系域，不是自然 RAG 无监督判错的现成解决方案 |
| [(How) Do Language Models Track State?](https://arxiv.org/html/2503.02854v2) | 原文 §3–4 | 不同内部算法可以实现同一状态追踪任务，probe/patching 的联合图样用于区分算法 | 应先列出不同机制的反事实预测；不能从一次中间状态解码成功反推唯一算法 |
| [Grounding Latent Algorithm Routing](https://arxiv.org/html/2607.24471v1) | 原文受控任务、路由干预及范围声明 | 区分 nuisance 不变性、regime 敏感性和定向干预 | 同词面改变适用关系、同关系改变词面必须分开；受控训练模型的路由结果不能直接推广到自然预训练 LLM |
| [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574) | 原文 retrieval score、复制检测与删除实验 | 少数及动态激活的 retrieval heads 对复制检索重要 | 保留物理 head 身份；复制到正确值不等于复制了它的适用范围 |
| [Two Pathways to Truthfulness](https://arxiv.org/html/2601.07422v1) | 原文问题定义、knockout/patching 概述 | 区分 question-anchored 和 answer-anchored 的真实性线索 | response 路径不能预设为错误来源；该划分不等同于当前直接 self 与过去 history 的划分 |
| [CHARM](https://arxiv.org/abs/2509.24770v2) | 原文 §3–4 与训练说明 | 保留逐 head attention 的属性图，可用 GNN 学习检测 | 监督检测是信息可读出的证据；本项目已有 node-only/连续性结果，关系增量必须单独比较 |
| [CausalGaze](https://arxiv.org/html/2604.11087v1) | 原文 §3.2 和 A.4 | 用 detector loss 对 A 的梯度引导可学习边 gate，再训练 GAT 分类器 | 不把 detector 的梯度边称为 generator 的实际消息因果效应；不迁移其监督标签训练为“无监督” |
| [Do LLMs Latently Perform Multi-Hop Reasoning?](https://aclanthology.org/2024.acl-long.550/) | 原作者摘要定位 | 区分中间实体被召回和它被后续有效使用，结果依赖关系类型 | 为“进入/被使用”分开验证提供旁证；本轮没有复现该文，也不由此声称服饰/洋葱共享同一机制 |

另核查了 [Flow of Spans](https://arxiv.org/abs/2602.10583) 的摘要。
它研究动态 span vocabulary 的 GFlowNet 生成，不能作为原生 attention head 协同的直接依据；
本轮没有据此加入随机游走或 GFlowNet。

## 文献实际改变了哪些决定

首先，来源“影响大”与来源“对当前陈述适用”必须分开。Information Flow 的外部相关性参照、
绑定工作中的关系变量都体现了这个区别。我们不再用消息范数、注意力集中度或 head 交互强度
替代适用性语义，也不把外部 ranker 的相关性视为精确事实合法集合。

其次，需要给归因明确指定计算终点。新实现计算的是**完整候选 log 概率差对原生消息 gate 的导数**，
通过真实后缀网络回传；过去的 local logit lens 只在当前残差上读最终 unembedding。
两者可以不同，但差异不是自动成立的“信息反转”。完整候选仍需长度和措辞控制。

再次，多头关系包含两个不同问题：在同一世界中，两条消息的联合效应是否非加和；
删掉上游消息后，下游消息是否改变并补偿/扩大损失。四世界交互与恢复实验分别回答它们。
非加和可能来自共享下游的归一化、MLP 或 attention，不能只凭 J 命名“语义协同”。

最后，当前候选定位使用单点 VJP，而非 EAP-IG、AtP*、完整因果电路发现或无监督语义学习。
这是一项可验证的测量重构。文献没有证明“强冲突”“异常簇”“少数状态”必然对准自然幻觉。
真正的检测创新若成立，应来自**被表示的适用关系与实际采用该关系的运输过程之间的失配**，
并在独立自然来源上超过只读节点、只读值和纯过去分数；这是本项目提出的待验证方向。
