# 图方法与冻结 LLM 信息采纳检测：有界文献提取

核查日期：2026-09-12。范围：四篇论文的一手全文、作者/论文链接的官方代码；没有运行实验或 GPU。表中“可复用”是接口层面的技术关联，“不能直接证明”是论文原评测范围之外的主张，不构成方法优劣或研究录取裁决。

**验证状态分层**：四篇均已打开 arXiv 一手页面及全文，核对标题、作者和方法；官方仓库均可访问。父流程确认项目与技能工具目录均未解析到 `verify_papers.py`，将按技能降级流程写总候选与状态文件。下表程序状态统一为 **⚠️ UNVERIFIED（verification unavailable: helper unresolved）**；一手内容核查已经完成，但不冒充 arXiv/CrossRef/S2 程序交叉验证。

| 论文、出处与程序状态 | 原任务及监督来源 | 是否修改模型 | 可复用的技术关联；原结果不直接证明的内容 |
|---|---|---|---|
| **GLEM** — Zhao et al., *Learning on Large-scale Text-attributed Graphs via Variational Inference*, ICLR 2023；arXiv 2210.14709。⚠️ UNVERIFIED（见上） | 文本属性图的节点分类；三个 OGB 图。使用观测节点标签，以及 LM/GNN 互相给出的伪标签，交替执行 EM 两步。 | 两个模块均会训练；实验 LM 为 DeBERTa，属于分类编码器，不能称为“冻结自回归生成器的双向增强”。 | 可借用文本与结构两路预测、交替蒸馏的组织方式；其分类提升没有验证生成答案中的证据采纳、事实真伪或连续幻觉边界。[全文 §4–5](https://arxiv.org/html/2210.14709)，[官方代码](https://github.com/AndyJZhao/GLEM)。 |
| **GraphAdapter** — Huang et al., *Can GNN be Good Adapter for LLMs?*, WWW 2024；arXiv 2402.12984；DOI 10.1145/3589334.3645627。⚠️ UNVERIFIED（见上） | TAG 节点分类。先用节点原文中的下一个 token 作自监督目标，再用下游节点标签训练分类任务。 | 冻结 LM 主干和预训练输出头，训练 GNN/融合模块；预训练的最终词分布会改变，下游另加分类头。 | 可借用冻结表示缓存、低维图分支及输出融合；原实验没有将差分预测概率标定为幻觉概率，也未定位连续幻觉范围。[全文 §4](https://arxiv.org/html/2402.12984)，[作者版会议论文](https://yangy.org/works/gnn/WWW24_GraphAdapter.pdf)，[官方代码](https://github.com/zjunet/GraphAdapter)。 |
| **GraphPrompt** — Liu, Yu, Fang & Zhang, *GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks*, WWW 2023；arXiv 2302.08043；DOI 10.1145/3543507.3583386。⚠️ UNVERIFIED（见上） | 链接预测自监督预训练；节点/图分类采用每类 k 个标签，学习读出提示向量并构建类别原型；在五个图数据集评测。 | 下游冻结预训练 GNN，仅调提示向量；原方法没有 LLM 生成器。 | 可借用局部子图表示与任务读出；原方法是少样本，不能把“Graph Prompt”名称当作零标签分类或零样本幻觉检测的依据。[全文 §3.1、4.3](https://arxiv.org/html/2302.08043v2)，[官方代码](https://github.com/Starlien95/GraphPrompt)。 |
| **GraphCLIP** — Zhu et al., *GraphCLIP: Enhancing Transferability in Graph Foundation Models for Text-Attributed Graphs*, WWW 2025；arXiv 2410.10329v4；DOI 10.1145/3696410.3714801。⚠️ UNVERIFIED（见上） | TAG 跨图零/少样本迁移。使用图与 Qwen2-72B 生成摘要配对，做对比预训练；零样本用类别描述匹配，少样本提示调优使用目标标签。 | 训练图编码器和投影器，冻结文本编码器；摘要 LLM 用于产出训练文本，没有训练成幻觉判别器。 | 可借用图—文本对齐、类别文本接口及跨域评测；摘要配对是语义监督来源，不是独立事实真值。相似度本身不证明某条证据被生成器采纳。[最新版全文 §3](https://arxiv.org/html/2410.10329v4)，[官方代码](https://github.com/ZhuYun97/GraphCLIP)。 |

## 对“瓶颈残差 Graph Adapter”的实现核对

GraphAdapter 官方 `FusionBlock` 将 LM 表示投影到低维空间，与图表示拼接后经过 MLP；预训练时再映射回 LM 隐藏维度。它的 residual learning 对应论文式 (11)–(12) 的两路 **softmax 概率平均**，不是 LLM 各 Transformer 层内部的 `h + Adapter(h)`。代码位于 [graphadapter.py 的 FusionBlock](https://github.com/zjunet/GraphAdapter/blob/main/graphadapter.py#L149) 和 [pretrain_utils.py 的概率融合](https://github.com/zjunet/GraphAdapter/blob/main/pretrain_utils.py#L273)。

因此，“保留原生成器、只在外部训练检测头”可与其冻结缓存方式建立工程联系；若实际把图分支接入生成词分布，组合模型的输出机制已发生变化。两种接口需要分别表述。这里没有从结构相关性推出原生成器采纳证据的因果关系。

## 资源与宣传口径核对

GraphAdapter 官方说明预处理缓存 token 表示，Arxiv 数据处理约需 **300 GB 存储**；这个数字不是显存。[官方 README](https://github.com/zjunet/GraphAdapter#requirements)

GraphCLIP 官方完整预训练配置写明 **8 张 A100 40GB、约 7 小时**，同时提供 PubMed 的单 GPU 示例及下载检查点入口；没有给出本任务在单张 3090 上复现完整结果的保证。[官方预训练说明](https://github.com/ZhuYun97/GraphCLIP#3-pretraining-on-source-data)

四篇原评测集中于图节点/图分类、链接预测与迁移；本次没有复现实验，因而不把作者报告的优于当时基线改写成本任务的 SOTA 承诺。

## 补充：CHARM 的内部计算属性图

Frasca, Bar-Shalom, Ziser & Maron，*Neural Message-Passing on Attention Graphs for Hallucination Detection*，ICLR 2026；核查版本 **arXiv:2509.24770v2，2026-07-27**。程序状态：**⚠️ UNVERIFIED（verification unavailable: helper unresolved）**；已人工读取一手全文 §3–5、附录 B/C/D 及官方代码。以下是原方法提取，不是新颖性或采纳判定。[论文元数据](https://arxiv.org/abs/2509.24770v2)

| 核查项 | 原文定义与边界 |
|---|---|
| 一个样本是什么 | 一个 prompt+response 对应一张图；每个 token 是节点。 |
| 高维属性 | 节点含全部层/头的自注意力对角向量 `α_ii ∈ R^(LH)`，可加残差激活 `a_i^l ∈ R^d`；非对角边含完整 `α_ij ∈ R^(LH)`，不是跨头均值。主实验激活选第 24 层，附录另测多层拼接。 |
| 拓扑与稀疏化 | 关系 `i→j (i>j)` 表示 i 关注 j；逐分量阈值置零，任一层/头保留正值才留下该边，默认 `τ=0.05`。原实验删除 prompt 内部边。 |
| 图读出 | MLP 消息与更新、邻居均值聚合；另给 prompt/response 关系类型标记。token 检测直接读出节点，response 检测平均池化。 |
| no-graph 对照 | `CHARM (no g.)` 去连接/消息传播，仅对节点属性施加稠密层；同样调参。CNN 的 AUPR 为 19.2，对应 CHARM 为 22.7；这不是保留全部边属性、只打乱拓扑的实验。 |
| zero-shot 的含义 | 已用源域标签训练，在同一 LLaMA-2-7B-chat 的 NQ→CNN 或 CNN→NQ 上迁移；目标域不再训练。没有展示“从未使用标签就自动分类”，也不是跨生成模型迁移实验。 |

以上定义、no-graph 对照与迁移条件来自 [CHARM v2 §3–5、附录 C/D](https://arxiv.org/html/2509.24770v2)。该论文的标签任务是幻觉检测，未提供“哪条约束拥有当前生成决定”的标签任务。

**监督来源与分割。** NQ/CNN 复用 Lookback Lens 的既有生成与 token 标注；官方溯源文档注明这些标注来自 **GPT-4o**。通过 teacher forcing 重放响应抽取 traces，而非重新采样后沿用旧标签。[溯源记录](https://github.com/Noired/charm/blob/7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61/PROVENANCE.md#L15)，[数据流程](https://github.com/Noired/charm#pair-up-with-precomputed-annotations)。论文改为完整 prompt-response 级 60/20/20 分割，避免同一响应的片段跨集合。Movies/WinoBias/Math 采用 LLMsKnow 的题目和参考答案，代码以答案字符串规则产生 correctness。[官方判定实现](https://github.com/Noired/charm/blob/7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61/data_prep_2/data_collection_utils.py#L129)。训练 GNN/读出用标签的 BCE；原 LLM 只提供计算记录，不随此损失更新。[训练实现](https://github.com/Noired/charm/blob/7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61/training.py#L299)

**影响复用的代码约定。** 所核官方提交为 `7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61`：

- `graphs.py` 用最后一个 prompt 位置的状态预测首个响应 token，因此保存 `prompt_len−1`；它把注意关系边翻转成 PyG 的 **j→i 消息方向**。默认去掉 prompt 内部边，代码阈值是 `<τ`，而论文公式写 `≤τ`。这些索引/方向/阈值约定应与原始记录分开保存。[构图代码](https://github.com/Noired/charm/blob/7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61/dataset/graphs.py#L13)
- 论文定义 `y=1` 表示幻觉；NQ/CNN 标注代码则默认 **1=正确、0=幻觉**。评估函数显式用 `1−y` 和负预测分数计算 hallu AUPR。复用时须核对极性，不能只按字段名猜测。[标注代码](https://github.com/Noired/charm/blob/7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61/data_prep_1/data_annotation.py#L217)，[评估代码](https://github.com/Noired/charm/blob/7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61/training.py#L262)

与当前仅观察真实正确/错误窗口的技术关联：可借用“完整内部属性 + 有向关系 + token 对齐”的数据表示；构图本身不需要幻觉标签，但 CHARM 的分类结果依赖监督训练。当前两窗口的属性区分观察不能引用为其训练性能的复现。
