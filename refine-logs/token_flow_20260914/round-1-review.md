## 独立方法评审：无监督 Token 图条件创新

### 问题锚点

- **底线问题**：在一个完整 prompt＋response 样本图上，无幻觉标签地给 response token 排异常分数，并检验其是否能定位幻觉。
- **必须解决**：直接自重构的复制捷径、未来拓扑泄漏、图统计与事实正确性混淆；错误起点和顺畅延续须分开检验。
- **非目标**：不把 attention 当真实因果消息，不以图罕见性证明事实错误，不复用 S10 监督权重。
- **约束**：沿用冻结 observer 缓存，来源隔离；标签仅在全部分数冻结后加入；CPU 优先。
- **成功条件**：工程无泄漏且数值正确；科学上真实连接在固定自然来源评价中胜过容量匹配的无图和置换对照，并改善回答内定位。

核心判断：当前文档是一份严谨、可证伪的工程基线规格，但还不是研究级方法。主要问题并非缺少模块，而是“上游条件创新”这一统计名称尚不由当前变量的时间关系支持；attention diagonal 版本还存在近似代数自重构。若不解决这两点，后续自然指标即使上升，也难以归因于上游信息流。

### 七维评分

| 维度 | 权重 | 评分 | 加权 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9 | 1.35 |
| Method Specificity | 25% | 7 | 1.75 |
| Contribution Quality | 25% | 4 | 1.00 |
| Frontier Leverage | 15% | 6 | 0.90 |
| Feasibility | 10% | 7 | 0.70 |
| Validation Focus | 5% | 7 | 0.35 |
| Venue Readiness | 5% | 3 | 0.15 |
| **加权总分** | **100%** |  | **6.20 / 10** |

**CALIBRATION: none**

**GAP：**没有提供人工选定的优劣候选方案，因此不能相对具体 exemplar 校准。本方案在问题忠实度、边界声明和删除判据上明显强于常见“attention 即解释”的方案；离 READY 的主要差距集中在 Contribution Quality 与 Venue Readiness：中心统计量仍是经典线性图聚合、岭残差和块最大值校准的组合，而且“创新”解释被目标条件化边权和 softmax 闭合关系破坏。它目前更接近一个高质量 falsification baseline，而非已闭合的机制贡献。

### 阻塞性问题

1. **边权仍携带目标信息，残差不是历史创新。**

   令 \(\mathcal F_{i-1}\) 表示 token \(i\) 出现前可用的信息。当前
   \[
   u_i=\phi(A_{i,\cdot},z_{<i},c_i),
   \]
   但 query \(i\) 的 attention 行 \(A_{i,\cdot}\) 由包含 token \(i\) 的 query state 产生，因此通常不属于 \(\mathcal F_{i-1}\)。即使没有显式输入 \(z_i\)，模型仍通过 \(A_{i,\cdot}\) 条件于目标。于是 \(e_i\) 可以称为“目标条件化的图残差”，不能称为相对于历史滤过的 innovation，更不能据此解释“上游解释不了当前 token”。

   必须二选一：

   - 若保留现有缓存与 query \(i\)，把估计对象改名为**当前 token 条件化的路由状态异常**，删除历史创新、提前预测和信息流增量解释。
   - 若坚持“历史条件创新”，所有 predictor 必须对 \(\mathcal F_{i-1}\) 可测；query \(P+t-1\) 的矩形缓存可以作为 token \(t\) 的前缀预测变量，但不能冒充 token \(t\) 的节点属性。当前 CPU 实现可继续作为后验基线，研究主张则需等待这一时序接口成立。

2. **attention diagonal 与行质量存在近似代数闭合。**

   若第 \(r\) 个 layer/head 的目标属性是
   \[
   z_i^{(r)}=A_{ii}^{(r)},
   \]
   则完整因果行满足
   \[
   A_{ii}^{(r)}=1-\sum_{j<i}A_{ij}^{(r)}.
   \]
   阈值截断后，只是多出 omitted/unknown mass。当前 \(s_i,h_i\) 和消息又来自同一 attention 行，因此低重构误差可能只是 softmax 单纯形恒等式，而非历史节点对当前状态的解释。层头平均只能削弱这一问题，不能消除它。

   研究主分数必须使用与边权没有这种行和恒等关系的目标，例如独立缓存的 frozen hidden 投影；若当前只有 diagonal，则将它明确降级为拓扑管线 sanity baseline。不能用它支撑“语义创新”或“信息流”主张。

3. **异常校准与幻觉识别是两个独立问题。**

   source-block maximum conformal 校准在定义正确、来源可交换、每个块采样协议一致时，确实可以控制新来源出现任一报警的概率；但它只控制混合无标签总体的报警预算。它不产生
   \(P(\text{hallucination}\mid G)\)，不控制正确 token 的 FPR/FDR，也不提供真假可分性。文中的不可识别反例是正确的，而且正是主理论边界。

   应明确写出所需但不能由无标签数据证明的区分假设，例如在固定任务、位置和不确定度协变量 \(C\) 后：
   \[
   P(S>s\mid Y=1,C)>P(S>s\mid Y=0,C)
   \]
   至少在某一区间成立。自然金标实验只负责检验该假设，不能被表述成无监督理论保证。

4. **“同容量 no-neighbors”目前不成立。**

   图模型比 \(c_i\)-only 模型多 \(2d\) 个输入和相应自由度。“相同预测器/损失”不等于相同容量。权重置换控制只检验固定端点集合内的权重分配；均匀边时还会退化成恒等。

   保留一个真正关键的容量匹配零模型即可：使用同维度的 \(m_i^P,m_i^R\)，但在 source、segment、位置或 lag 粗分层内置换端点特征，或使用同质量的均匀历史聚合。它与真实图使用相同正则化和拟合流程。`no_neighbors` 可保留为删除对照，但不能单独支撑“图连接有增量”。

5. **来源最大值校准的有限样本条件必须落实到 roster。**

   \(\alpha=0.05\) 时至少需要 \(B\ge19\) 个校准来源，否则阈值必为 \(\infty\)。本地 RAGTruth 来源数量使这一点原则上可行，但需在规格中冻结：

   - source 的精确定义；
   - 每来源固定的回答生成器集合和回答数；
   - 长度/位置抽样规则；
   - 是否按 task×generator 分层校准；
   - 每层实际 \(B\) 和由此可达到的最小 \(\alpha\)。

   若命名来源并非从共同超总体随机抽取，则“可交换”只是分析假设，不能写成无条件保证。

6. **首错定位评价需与后验时间语义一致。**

   当前分数看到 token 后才生成，可以做 post-hoc localization，不能做 token 发出前的预警。主指标应是每回答中首错在“回答开始至首错”候选位置里的 rank percentile、MRR/Recall@k，以及首错前提前报警率；source-balanced pooled AUROC/AUPRC作为辅助。严格首错后的表现只作诊断，因为当前主分数没有延续机制。

   还需冻结字符 span 到 observer token 的映射、部分重叠 token 的处理和未覆盖率，否则“精确首错”可能由标注投影规则决定。

### 低分维度的必要修正

- **Contribution Quality — 4/10，CRITICAL**

  弱点：岭残差、attention 聚合、Gaussian 工作模型和块 conformal 都是现成工具；更严重的是中心术语与实际时序对象不一致。

  修正：先修复前缀可测性和 diagonal 闭合，只保留一个主命题：

  > 严格前缀可测、无标签拟合的历史条件残差，是否在新来源中为首次幻觉 token 提供超越位置、路由质量和 next-token uncertainty 的回答内增量排序信号。

  自然结果出来前应写成待检验命题。只有真实图稳定胜过容量匹配零模型，才能转成论文主张。

- **Frontier Leverage — 6/10，IMPORTANT**

  弱点：技术路线并不落后，但 attention 图检测已经拥挤。Lookback Lens 已表明简单 head-wise context/generation 比例具有强竞争力；LapEigvals 已使用 attention 图谱特征；ACL 2026 的 TOHA 已直接使用 attention graph topological divergence，另有工作使用 attention KL divergence 做轻量检测。[Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/)、[LapEigvals](https://aclanthology.org/2025.emnlp-main.1239/)、[TOHA](https://aclanthology.org/2026.acl-long.704/)、[Attention Divergence](https://aclanthology.org/2026.acl-srw.6/)

  修正：不要加 LLM critic、RL 或扩散模块。应把现代性放在“严格时序审计＋标签冻结＋来源外验证＋容量匹配图零模型”上，并与这些近期 attention 方法的特征或冻结版本公平比较。仅凭“转成图”已经不能构成新颖性。

- **Venue Readiness — 3/10，CRITICAL**

  弱点：尚无自然实验结果，且当前主分数的解释有阻塞漏洞。近期研究还显示首个幻觉 token 虽总体更可分，但回答间方差很大，因此已有观察不能替代本方法的实证验证。[First Hallucination Tokens Are Different from Conditional Ones](https://arxiv.org/abs/2507.20836)

  修正：完成上述统计对象修订后，只做一次冻结的自然确认。若真实图未胜容量匹配零模型，保留负结果并删除“图改善检测”的贡献；若只提升 pooled 指标、没有回答内首错 rank 提升，也不能宣称定位成功。

### Simplification Opportunities

1. 主算法只保留残差能量 \(a_i\) 与 source-block 校准；Gaussian NLL 与“编码代价”不增加排序信息，可移到附录或删除。
2. diagonal 和 hidden 不应作为两个并列主版本。diagonal 仅作工程 sanity，时序合法且无代数闭合的表示才可成为研究主版本。
3. 路由熵、JS、Dirichlet 能量和 CMI 讨论保留为边界或基线，不并入主分数。一个主分数、一个容量匹配零模型、一个 no-neighbor 删除对照足够。

### Modernization Opportunities

1. 不增加新可训练模块；优先把 predictor 改为严格前缀可测的 observer 内部状态或 attention 行。
2. 将贡献定位为 foundation-model 内部信号的**过滤时序正确性与增量识别**，而非再次发明一种 attention 图统计量。
3. 基线更新到 2026 年 attention topology/divergence 工作，并明确监督 probe、少样本 probe 与本方案 label-free fitting 的设定差异。

### Drift Warning

**NONE。** 文档本身仍忠实保留“无标签打分、事后检验能否定位幻觉”的问题锚点，并主动声明不可识别性。潜在 drift 只会发生在后续把“混合参考中的图异常”直接改写成“幻觉概率”或“事实信息流失败”时。

### Verdict

**RETHINK**

需要重想的是中心估计对象和它的名称，不是放弃整条工程路线。当前 CPU 实现仍值得完成，用于验证索引、稀疏计算、来源冻结、校准边界和负对照；但在 query 目标条件化与 diagonal 闭合消除前，它只能是工程基线。修复后若新来源自然实验显示真实历史连接对回答内首错定位有稳定增量，这条路线可以发展成研究；在此之前不能称为已有效的方法。
