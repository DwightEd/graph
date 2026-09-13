# 约束适用性：MiniCheck / GraphCheck 方法精读与结构建议

日期：2026-09-13。任务：寻找能实现“当前陈述应采用哪些证据”的模型结构，而非重复证明关系影响输出。

文献核验状态：**UNVERIFIED（deterministic helper unavailable）**。已人工打开作者论文全文、方法及附录，读取官方固定提交代码；这不替代自动书目核验或跨模型科学审查。本文未运行 GPU，未改动现有实验。

## 1. 已读方法事实

| 文献 | 阅读范围 | 核心事实与限制 |
|---|---|---|
| Tang, Laban, Durrett，MiniCheck，EMNLP 2024；UNVERIFIED | §3.1–3.3，Appendix A.3/G/H；官方推理和 C2D/D2C 代码 | 用结构化合成监督学习文档是否支持一个句子，不能定位生成模型内部回看点。 |
| Chen et al.，GraphCheck，ACL 2025；UNVERIFIED | §3–4.1，Appendix G/H；官方 graphcheck/gnn/config/graph_build 代码 | 用两个文本知识图辅助冻结 LLM 判别支持性，未显式学习原生成器的证据采用路径。 |

**MiniCheck 论文事实。** C2D 将 claim 分成原子事实，每个事实扩展为需联合推理的句对，再合成文档；删掉必要句并检查剩余材料不能支持对应事实，才得到负例。重组事实时，全部得到支持才标正。D2C 从真实文档三个 chunk 的摘要出发，逐句删除、跨 chunk 配对后逐事实检查。训练用交叉熵；FT5/DeBERTa 使用约 14K 合成加 21K ANLI，后者 neutral/contradiction 合并为 unsupported；先训 ANLI+C2D 两轮，再训 D2C 一轮。默认阈值 0.5。作者小规模人工审计合成标签仅约 80%/78% 准确，不能把生成标签称作真值。无独立“缺失证据”状态。[全文 §3、Appendix A.3/G/H](https://arxiv.org/html/2404.10774v2)

**MiniCheck 推理代码事实。** 固定提交 `b58b9fa69acbd1015ec970fa65dd752413a053d2`：FT5 输入为文档和 claim 的拼接，decoder 首步两个标签 token 的 logits 经 softmax 得支持概率；RoBERTa/DeBERTa 使用二分类 head。长文档分 chunk，最终取支持概率的最大值；多句 claim 推荐用户预先分句，没有内部逐事实定位输出。因此“多个 chunk 合在一起才支持、任何单个均不支持”的情况不能依靠这个 max 聚合表达。[inference.py:143–237](https://github.com/Liyan06/MiniCheck/blob/b58b9fa69acbd1015ec970fa65dd752413a053d2/minicheck/inference.py#L143)

**GraphCheck 论文事实。** Claude-3.5-Sonnet 分别抽取文档/claim 三元组；实体与关系由 all-roberta-large-v1 编成 1024 维特征。GNN 分别编码两个图，全图 readout 后 projector 映到冻结 LLM 的输入维度，与原始文本拼接，生成 support/unsupport。监督来自 MiniCheck 14K 合成数据。附录为 GAT/Graph Transformer，2–4 层、4 heads；主模型 Llama3.3-70B/Qwen2.5-72B，4×A100-80GB，20 epochs、验证早停。没有 claim→source 对齐矩阵、明确的 null evidence 或回看/采用监督，不能直接作为当前内部机制模型。[全文 §3–4.1、Appendix G/H](https://aclanthology.org/2025.acl-long.729.pdf)

**GraphCheck 固定代码事实。** 提交 `9d1456df09752df3930aa728f94cc7091ded6013` 的实现是共享 GNN 分别处理两个图，**mean** pooling，各得一个向量；projector 为 `1024→2048→d_LLM`，中间 Sigmoid；两个向量变成两个前缀 token，损失只作用于标签和 EOS，LLM 冻结而 GNN/projector 可训。论文描述 sum readout，附录 virtual_tokens=4，但此代码采用 mean/2，不能混写。空图向量置零只是工程兜底，不是学到“无适用证据”。原文本直接截至 max_txt_len=1024。[graphcheck.py:51–167](https://github.com/Yingjian-Chen/GraphCheck/blob/9d1456df09752df3930aa728f94cc7091ded6013/model/graphcheck.py#L51)

**复用前必须检查边特征。** 此提交默认 GAT，构造 GATConv 时未传 edge_dim；Graph Transformer 则明确设置 edge_dim。按 PyG 当前实现，edge_dim=None 不建立 lin_edge，传入 edge_attr 也不会进入注意力的边特征项。此判断是对固定作者代码及当前依赖源码的静态推断，未复现其训练时依赖版本。[作者 gnn.py](https://github.com/Yingjian-Chen/GraphCheck/blob/9d1456df09752df3930aa728f94cc7091ded6013/model/gnn.py)，[PyG GATConv 源码](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html)

## 2. 对当前方法的判断（以下为我们的设计推论）

两篇工作都不能补上“当前应该采用哪个约束”的全部缺口。MiniCheck 提供了较好的训练任务构造原则，GraphCheck 提供轻量图编码接口；直接复制“两个图池化为两个 token”会丢掉我们最需要观察的证据对应关系。更不能拿原生成器的注意力作为正确归属监督：原生成器恰好可能注意了错误阶段的数字。

应当把两个对象显式分开：**规范归属**回答哪些来源事实足以支持当前命题；**实际采用**回答原生成器沿哪些边将何种信息写入当前预测。这两个分支必须使用不同监督。对前者，高 attention 不等于适用；对后者，语义匹配不等于已被输出采用。

建议采用“保留局部对齐的有类型因子图”，而不是先选一个回看 token 再检测。即使回看定位不确定，也应能完成当前命题的支持性判别；定位作为可检查的概率分布与集合输出，不能成为整个检测器不可恢复的单点瓶颈。

## 3. 可实现的结构约定

1. **节点不是只有实体。** 保留 source、response 的全部 token/子句节点，并增加命题节点。命题连接谓词、论元角色和限定条件 span：对象、动作、阶段/时间、位置、否定、数量/单位、条件作用域。条件未知必须保留 unknown，不能由模型脑补。相同数字、相同实体名称的不同提及保持不同节点，避免跨事件融合。每条抽取边携带原文字节/token 区间与抽取置信度。
2. **高维特征保留原始来源。** native 4096 维残差与按层 Q/K/V 是原始特征；图编码器通过各类型线性投影到 256 维，不先把它们变成几个手工标量。角色/限定关系通过独立 edge embedding 参与消息计算。两层有类型消息传递作为初始结构，残差连接；源图还可保留文本语义特征，但必须记录所用编码模型。全图始终可访问，span 划分仅改变聚合权重，不删除跨 span 边。
3. **显式适用矩阵。** 对每个待核实命题/属性槽 `c_i`，产生其对 source 命题或证据集合 `B_j` 的分布 `P_app(j|c_i,D)`，另含 `null`。关系一致性必须条件化到事件和限定词，不能仅点积数字节点。多句证据用显式集合/因子节点组合；不能要求每个 source 单句独立蕴含整个 claim。输出同时保留 entailed、contradicted、unverifiable 三类；“所有槽均被覆盖”才构成全命题支持。unknown 抽取与真正 null 证据分别输出。
4. **采用分支保留输出方向。** 将 native 路径上的 attention×value 消息和下游写入方向输入另一个 head，估计各证据对当前命题/后续 token 的正向、负向贡献。用实际干预得到的 log-prob 或命题对比变化做蒸馏目标；attention 强弱不能作采用标签，残差 cosine 不能替代输出方向。该分支的结果再与 `P_app` 对比，从而区分“路由到不适用事件”“路由到了适用证据但后续写错”“没有适用约束仍具体化”。
5. **定位不先硬选点。** 对一个 claim 的各预测位置输出归属状态和采用状态，定位它们的变化及上游因果支持集合。完整 response 场景可用后续 claim 定义回看目标，但必须标成离线定位；在线版本不得偷看未来。标点只提供边界候选，事件/归属变化提供软分段，历史边保留。不能将单点翻转与唯一必要节点混同。

## 4. 不使用 RAGTruth 幻觉标签的训练目标

`L = L_app + L_support + L_route + L_consistency` 是结构接口，不是已证实的最佳权重。

- `L_app`：从训练 sources 的可核对命题生成归属问题，学习“命题→证据集合/null”。正例保留具体支持区间；删除一个区间后重新判断其余全文是否仍提供支持，不能直接标 null。借用 MiniCheck 的局部、逐事实验证，避免用另一个模型对长文一次判定后当真值。
- `L_support`：覆盖全部原子约束的三类判别。构造同实体、同数值、不同作用域的 hard negatives；替换可能碰巧仍正确，只有验证明确的才入训。自然来源缺失某项事实是 unverifiable，不应伪称 contradiction。无法验证的合成样本进入弃权/未标注池。
- `L_route`：对已采集内部图的小批量干预轨迹，蒸馏每条证据路径对实际输出的有符号影响。它只监督采用分支，不把模型本身的采用行为当正确语义归属。
- `L_consistency`：source 顺序置换但事实不变时，归属预测应按节点置换等变；合法同义改写保持语义支持；仅替换已验证作用域时，改变相应归属。不能把所有扰动都当负样本。

这里是**无人工幻觉标签的合成/弱监督训练**，不是严格无监督。抽取模型、验证模型、原生成器若相同，错误可能相关；需要显式置信度、弃权与来源隔离，不得声称一个 8B 自己检查自己就消除了标签误差。按原始 source 分训练/验证/测试，再生成变体，防止同源改写泄漏。

当前应实现的最小完整系统是：命题与限定条件表示 → 可适用证据集合/null → 实际采用路径 → 命题风险与影响区间。图编码的价值取决于它是否在相同节点、相同监督、相同算力下超过无图对齐模型，不能从 GraphCheck 的结果移植“图必需”结论。
