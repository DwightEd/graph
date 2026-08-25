# 文献与实验审计：我们真正知道了什么，哪些说法还站不住

> 本文写给后续方法设计和 ICLR 论文使用。它的目的不是把过去的尝试包装成一条“逐步成功”的故事，而是把每个假设、每个特征、每个结果和失败原因说清楚。只有这样，下一版方法才不会换个名字后重复同一个错误。

## 0. 先说结论

过去几轮实验并非“attention 完全没有信号”，但也没有支持“随便构一张图，再做重构或异常检测就能识别幻觉”。目前最可靠的观察只有四条。

第一，**层和 head 不能过早平均**。旧的 channel-mean 谱方法几乎随机，而保留 32 层 × 32 heads 后，response-history（下文简称 RR）联合谱残差达到约 0.660 AUROC / 0.134 AUPRC。

第二，**直接看 prompt attention 不够**。在幻觉起点附近，prompt mass share 甚至上升；但把历史 response token 追溯回 prompt 后，一跳 prompt provenance 的位置重心和离散度出现了当前最强的单结构信号。这说明真正有用的对象可能不是“当前 token 看 prompt 还是看 response”，而是“当前 token 依赖的 response source 是否仍然继承了 prompt 证据”。

第三，**幻觉不一定是难恢复的异常点**。来源重连判别、图重构、动态预测和流形 kNN 多次失败。一个错误回答一旦形成内部自洽的续写模式，反而可以很稳定、很容易预测。把“低概率、难重构、局部结构破裂”直接等同于幻觉，假设本身就不稳。

第四，图关系确实有可预测信息，但目前只证明了“图有结构”，没有证明“该结构就是幻觉机制”。无标签 held-out 审计中，depth transport gain 为 `+0.05663`，query-set gain 为 `+0.02642`；relay 和 exact-path rewire 只有约 `+0.006`。这支持继续使用图，但不支持把 holonomy、长路径或复杂消息传递直接写成论文结论。

因此，当前 HoloRoute 应当被视为一个**可运行的 attention 图表征基线**。下一步不该继续往它上面堆模块，而应该改变研究问题：从“这张图是否异常”改成“这个 token 的路由是否已经可以绕开 prompt 证据继续运转”。

---

## 1. 读结果前必须分清三种指标

过去的结果来自不同数据范围、不同任务和不同评估方式，不能放在一张排行榜里直接比较。

### 1.1 标签后验的 separability

部分逐层、逐 head 筛查报告：

\[
\operatorname{separability}=\max(\operatorname{AUC},1-\operatorname{AUC}).
\]

它先用标签判断“高值更像幻觉”还是“低值更像幻觉”，再把方向翻到大于 0.5。因此它适合回答“这个坐标里有没有可分信息”，不等于一个提前冻结方向的无监督 detector。

### 1.2 冻结方向的无标签分数

RR 谱残差、CaSH 分数和部分图方法是在 train 上拟合 reference、在 test 上先冻结分数，最后才打开标签。这些 AUROC/AUPRC 更接近真正的无监督检测结果。

### 1.3 幻觉 onset 的配对效应

onset 实验比较同一回答中幻觉起点附近与匹配位置的正常 token，报告均值差、配对效应量和显著性。它说明局部变化是否稳定，不是 detector AUROC。

还要注意：有的实验只跑了 30 个样本，有的跑了 751 个 token，有的覆盖 73,994 个 token。本文会保留各自的范围，不把它们混成同一组结论。

---

## 2. 文献真正提出了什么

下面不是按论文标题罗列，而是按“它相信幻觉为什么会发生”来整理。

| 工作 | 核心假设 | 方法怎样检验 | 监督信息 | 对我们的启发 | 仍然没有回答的问题 |
|---|---|---|---|---|---|
| [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/) | 上下文幻觉与“看给定上下文”相对“看自己已生成内容”的比例有关 | 每个 layer/head 计算 context-to-generation attention ratio，再训练线性分类器 | 需要幻觉标签训练分类器 | attention 的 prompt/history 分配确实有信号，而且 head 之间差异很大 | 看 response 不代表脱离证据；response token 可能在转运 prompt 证据 |
| [CHARM](https://arxiv.org/abs/2509.24770) | 单个统计量丢失 token 间关系，监督 GNN 能从 attention flow 和 activation 图中学习更复杂模式 | token 为节点，attention 为边，attention/activation 为属性，训练监督图分类器 | 监督 | 图学习可能优于固定启发式；图不是只能做手工统计 | 性能来自图、activation 还是标签驱动的判别边界并未被无监督地拆开 |
| [LapEigvals](https://arxiv.org/abs/2502.17598) | attention 图的谱结构在事实与幻觉之间不同 | 从 attention graph 提取 Laplacian 特征，再训练检测器 | 以监督 probe 为主 | layer/head 联合结构不应过早压成少数标量 | 谱异常只说明结构不同，不说明证据从哪里来、是否被回答真正依赖 |
| [TOHA](https://aclanthology.org/2026.acl-long.704/) | prompt 子图与 response 子图的拓扑差异在特定 heads 上与不忠实回答相关 | 比较 prompt/response attention subgraph 的 topological divergence | 少量标签用于选择/校准 | prompt 与 response 结构不能只看总质量 | response 子图中的路径可能是合法证据中继，也可能是自我确认；TOHA 没有区分 |
| [HalluZig](https://aclanthology.org/2026.eacl-long.159/) | 正确与幻觉生成在 layer-wise attention topology 的演化上有不同持久签名 | 把层序列建成 zigzag filtration，提取持久同调签名 | 检测阶段仍需学习/校准 | 跨层动态值得保留，单层快照不够 | topological signature 是全局摘要，难解释某个 token 依赖了哪条证据路径 |
| [RFS-Guard](https://aclanthology.org/2026.acl-long.885/) | 跨阶段 routing 过度对齐语义相近的历史步骤，会形成 self-confirmation | attention routing 与 hidden-state cosine similarity 联合成 Routing Focus Score | 需要正确性监督来定方向/建检测器 | “看历史”本身不是问题，问题可能是历史语义与路由共同收缩成自证回路 | 依赖 hidden state、step segmentation 和 reasoning/answer phase；不是 token-level attention-only 无监督方法 |
| [Reasoning Fails Where Step Flow Breaks](https://aclanthology.org/2026.acl-long.1212/) | 错误推理常见两种流故障：浅层只盯当前步骤，深层逐渐忘掉前文 | attention-gradient Step-Saliency；再用干预修复 shallow lock-in 和 deep decay | 诊断使用输出相关梯度，不是纯无监督 | 机制要靠干预验证，不能只看相关性；跨层变化应有明确的“桥接信息”对象 | 健康地传播错误前提仍然会错；信息流好不等于事实正确 |
| [CoDA](https://aclanthology.org/2026.findings-acl.576/) | 有充分证据时仍幻觉，是因为中后层 context-selective routing 变弱，参数知识压过外部证据 | 分析 residual stream 中 context/parametric dominance，并增强 evidence-aligned value states | 主要是分析和干预 | raw attention weight 不等于真正运输的内容，value state 很重要 | 我们目前只有 attention cache，只能研究 routing footprint，不能声称功能贡献 |
| [The Phenomenology of Hallucinations](https://arxiv.org/abs/2603.13911) | 模型未必没检测到不确定性，而是没把它接入输出；不确定性进入低敏感子空间 | hidden-state geometry、topology、gradient/Fisher probe 和干预 | 机制分析 | “内部有信号”与“输出采用该信号”可以分离；幻觉不一定是低密度异常 | attention-only 只能观察这种内部变化在后续 Q/K 路由上的投影 |
| [CausalGaze](https://arxiv.org/abs/2604.11087) | 幻觉依赖的图关系比事实关系更脆弱，监督梯度能找出关键边 | hidden states 为节点、attention 为边；用检测损失对边的梯度做 refinement，再训练图模型 | 明确监督 | 只看静态边不够，结构敏感性和反事实视图值得研究 | 它的敏感性来自幻觉标签；不能直接搬到严格无监督设置 |
| [GraphMAE](https://arxiv.org/abs/2205.10803) | 遮蔽节点属性并恢复，可以学到通用图表示 | masked feature reconstruction | 无监督 | 当前 HoloRoute 的直接工程基线 | “能恢复”只代表图中有冗余，不代表恢复误差对应幻觉 |
| [CoLA](https://arxiv.org/abs/2103.00113) | 异常节点与其局部子图不一致 | target-subgraph 对比学习 | 无监督 | 可在节点 embedding 上做无监督异常检测 | 幻觉可能与历史子图高度一致，甚至比正常 token 更自洽 |
| DOMINANT / SL-GAD / BOURNE | 异常表现为属性/结构重构差、不同视图不一致或 bootstrap 表示异常 | 图重构、对比、双视图学习 | 无监督 | 提供标准图异常基线 | 都依赖“异常与正常图规律不一致”的一般假设，而我们的多轮结果反复挑战了这个假设 |
| [DBGNN](https://proceedings.mlr.press/v198/qarkaxhija22a.html) | 动态图中的影响沿有时间顺序的 causal walks 传播；高阶路径可能包含非 Markov 信息 | 把有序 walk 提升为高阶 De Bruijn 图，再做 message passing | 任务监督 | 路径必须保留顺序，不能只看静态邻接 | 我们的 order-2/3 path gain 很弱，说明“高阶越好”在当前数据上并不成立 |
| [Neural Sheaf Diffusion](https://proceedings.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html) | 不同关系上的表示未必在同一坐标系，消息应先经关系映射再聚合 | 学习 edge-specific linear maps 和 diffusion | 下游任务监督 | 可作为 relation-specific transport 的实现参考 | 它解决 GNN 表达问题，不提供幻觉机制；加进去不会自动产生 insight |

从这些工作中可以抽出一条共同线索：**attention 中真正有价值的不是“权重高不高”，而是某段信息通过哪些路径被持续读取、转运、放弃或替代。** 但现有方法大多停在三处之一：直接 prompt/history 比例、监督图分类、或全局拓扑摘要。它们都没有在严格无标签、逐 token 的条件下回答：一个 response-origin 路径到底是在转运 prompt 证据，还是已经形成不需要 prompt 的闭合回路。

---

## 3. 仓库与方法演化：我们实际上试过什么

以下提交节点反映了研究主线，而不是单纯的工程修改。

| 阶段 | 代表提交 | 研究问题 | 结论 |
|---|---|---|---|
| 简单图统计与属性图 | 早期 `attention_graph` 系列 | 幻觉是否更少看 prompt、更集中、更局部 | 部分有弱信号，但方向并不统一 |
| multiplex recovery | `0513c3c` | 正确图是否更容易从邻居恢复 | 只有极小效应，且 diagonal 方向相反 |
| 非神经 lineage / routing audit | `558fbb5`、`1ec3906` | response token 是否能追溯到 prompt origin | 一跳 provenance 有信号，但压成少数状态后很弱 |
| routing dynamics prototype | `ed55cfd`，后由 `6f73e8f` 退役 | 跨层动态神经模型能否学到 fracture/lock-in | 存在未来层泄漏、标签对齐和显存问题，不能作为证据 |
| causal walk / De Bruijn | `278bf74`、`825cbec`、`c577ae2` | 高阶有序路径是否比一阶关系更有信息 | order-1 有信号，高阶 gain 接近随机；最终总分被位置累积主导 |
| holonomy audit | `f5eec43` | depth、relay、query、diamond 是否真的提供 held-out 预测增量 | depth/query 明确有用；relay/exact path 很弱；只证明结构可预测 |
| HoloRoute baseline | `6572e0d` | 用 event graph + masked reconstruction 学无标签表示 | 已实现，但尚未证明图检测优于 Flat-1024 |
| Flat-1024 对照 | `e0457a0` | 图增益是否只是来自更多层头数据 | 已实现，正式结果待跑 |
| 可读性与运行修复 | `29f0d5a` 之后 | 形成稳定、可消融的工程基线 | 这是工程底座，不是论文机制 |

需要特别说明：仓库后来清理了旧实验代码，很多旧特征只能从保存的结果、上传文件和 Git 历史恢复定义。本文对能确认公式的特征给出完整计算；对只能确认概念但缺少最终旧版 schema 的特征，会明确标为“历史定义”，不会补造细节。

---

## 4. 数据到底是什么

当前 formal sparse attention cache 对每条样本保存：response query rows 的 layer、head、source、target、attention weight，self diagonal 单独存储。每个 response token 在每个 layer/head 上构成一行因果 attention。

三个限制贯穿所有实验。

1. **未保存的边不是零。** 它只表示权重低于 `attention_floor`。把它直接当零会低估边数、质量和熵。
2. **attention weight 是路由系数，不是完整功能贡献。** 真正的 residual update 还取决于 value、output projection、residual 和 MLP。
3. **teacher forcing 与自由生成不同。** 一旦真实生成产生错误，后续上下文会改变；teacher-forced cache 只能观察给定 response 下的内部路由。

这意味着本文中的“prompt-rooted”“response-closed”“provenance”都只是 **attention-derived routing 描述**，不是因果归因到最终 logit 的证明。

---

## 5. 所有已审计特征：怎么计算、结果怎样、为什么可能失效

### 5.1 基础 token 图统计

早期代码先把同一 `source -> target` 在所有 layer/head 中的 retained weight 相加，再除以 channel 数，得到一个合并后的 pair weight。随后对每个 response token 计算下表。

| 特征 | 具体计算 | 它试图表达什么 | 已观察结果或问题 |
|---|---|---|---|
| `incoming_mass` | 当前 token 所有合并后 incoming pair weight 之和 | retained attention 总量 | 受 sparse floor 和对角线处理影响，单独不稳定 |
| `prompt_mass_share` | prompt-source mass / 全部 retained incoming mass | 当前 token 直接看 prompt 的比例 | onset 实验中幻觉反而增加 `+0.02353`，不能把“少看 prompt”当统一机制 |
| `normalized_entropy` | 令 `p_i=w_i/Σw`，计算 `-Σp_i log p_i / log(degree)` | incoming attention 是否集中到少数 source | onset 平均效应 `-0.03153`，配对 `d_z=-0.8179`；局部强，但全局方向在不同实验中会变 |
| `history_lag` | RR 边按权重加权的 `(target-source)/(response_count-1)` 均值 | 更依赖近邻还是远距离历史 | onset 无显著信号；全局层筛查也只有约 0.630 separability |
| `in_degree` | retained distinct source 数 | 路由支持的宽度 | 早期 AUROC 约 0.463，方向不可靠 |
| `prompt_degree` | retained prompt source 数 | 直接证据覆盖范围 | 强受 prompt 长度和 floor 影响 |
| `history_degree` | retained prior-response source 数 | 回答内依赖范围 | 随 token 位置自然增长，必须做位置条件化 |
| `in_density` | `in_degree / absolute_target_index` | 除去可用历史长度后的密度 | onset 无显著差异 |
| `prompt_density` | `prompt_degree / prompt_length` | prompt 覆盖率 | 长 prompt、稀疏阈值会显著改变值 |
| `history_density` | `history_degree / response_position` | 回答历史覆盖率 | 早期 token 分母小，位置效应强 |
| `history_edge_share` | `history_degree / in_degree` | retained 边中 RR 边的比例 | onset 无显著差异；另一次全局审计 `history_edge_fraction` AUROC 约 0.642 |
| `channel_edge_density` | retained channel-level entries / `(L×H×可用 source 数)` | 层头层面的稀疏程度 | 主要反映 cache floor 与序列长度，难直接解释正确性 |
| `history_mass_fraction` | RR retained mass / 全部 retained off-diagonal mass | 回答内质量占比 | AUROC 约 0.5865，弱信号 |
| `mean_edge_strength` | 当前 token retained incoming weight 的均值 | 保留下来的边是否更强 | AUROC 约 0.5337 |
| `top1_share` | 最大 incoming retained weight / 总 retained mass | 是否由单一 source 主导 | AUROC 约 0.5842 |
| `retained_concentration` | 对每个 `(target,layer,head)` 计算 `Σw_i²/(Σw_i)²`，再跨 channel 平均 | source 权重的 HHI 集中度 | AUROC 约 0.5850；和 entropy 类似但不完全等价 |

这些特征没有完全失败；它们说明 attention 的集中度和 RR 依赖包含弱信号。但它们有一个共同问题：把“看了谁”压成一个数后，丢失了 source 的身份、prompt 位置、跨层变化和 response source 自己的来源。

### 5.2 Lookback ratio：1024 个坐标，不是一个万能分数

对 response token `t` 和 channel `c=(layer,head)`：

\[
M_P(t,c)=\sum_{s\in prompt}A_c[t,s],
\qquad
M_R(t,c)=\sum_{s\in response,s<t}A_c[t,s].
\]

读取单独保存的 diagonal `d(t,c)`，再按可用 source 数做均值：

\[
\bar M_P=M_P/P,
\qquad
\bar M_G=(M_R+d)/(t+1).
\]

Lookback ratio 为：

\[
LB(t,c)=\frac{\bar M_P}{\bar M_P+\bar M_G}.
\]

Llama-3.1-8B 每个 token 有 `32×32=1024` 个坐标。历史筛查中最强坐标约为 layer 31 / head 12，separability `0.6892`；top 10% 坐标约 `0.6464`。不同 heads 的方向不一致。

为什么它能工作一部分：某些 heads 可能专门承担 context retrieval，prompt/history 比例确实反映是否读取外部证据。为什么它不能成为我们的最终机制：它把所有 response source 一概视为“自生成内容”，没有区分这些 source 是否从 prompt 继承了证据；最佳 head 还是标签后验选出的。

### 5.3 Prompt 位置几何

这类特征不只问“看了多少 prompt”，还问“看了 prompt 的哪一部分”。先在每层合并同一 source-target 的 heads，设 prompt source 的归一化位置为 `x_s=s/(P-1)`。

| 特征 | 计算 | 结果 |
|---|---|---:|
| `prompt_centroid` | `Σ w_s x_s / Σw_s` | best separability `0.6905`，幻觉在最强层更低 |
| `prompt_spread` | `sqrt(Σw_s x_s²/Σw_s - centroid²)` | 单独结果较弱，主要在 provenance 版本中变强 |
| `prompt_centroid_shift` | 当前 token centroid 与前一 token/参考轨迹的变化 | `0.6494` |
| `retained_prompt_span` | retained prompt source 的最大位置减最小位置 | `0.6474` |
| `retained_prompt_coverage` | distinct retained prompt sources / prompt length | `0.6332` |
| `retained_prompt_mass` | 当前层所有 prompt retained edge weight 之和 | `0.6873`，幻觉在最强层更低 |

这些结果说明“证据位置”比单纯 prompt mass 更有信息，但仍是直接连接。它无法处理一个常见情况：当前 token 主要看历史 response，而那个 response token 之前已经正确读取了 prompt。

### 5.4 Prompt provenance：目前最值得保留的结构线索

先给每个历史 response token 保存它直接从 prompt 得到的三项状态：prompt mass、prompt 位置的一阶矩和二阶矩。当前 token 再通过 RR attention，把这些状态从历史 response source 传播过来。

一跳传播可写成：

\[
S_t^{(1)}=\sum_{j<t}w(j\rightarrow t)S_j^{(0)}.
\]

其中 `S` 同时包含 mass、first moment 和 second moment。由此得到：

\[
centroid_t^{(1)}=\frac{m_{1,t}^{(1)}}{m_{0,t}^{(1)}},
\qquad
spread_t^{(1)}=\sqrt{\frac{m_{2,t}^{(1)}}{m_{0,t}^{(1)}}-(centroid_t^{(1)})^2}.
\]

二跳版本再沿 RR 图传播一次：

\[
S_t^{(2)}=\sum_{j<t}w(j\rightarrow t)S_j^{(1)}.
\]

| 特征 | best separability | q90 | 最强方向 |
|---|---:|---:|---|
| `prompt_provenance_centroid_hop1` | `0.7158` | `0.6857` | 幻觉更低 |
| `prompt_provenance_spread_hop1` | `0.7051` | `0.6776` | 幻觉更低 |
| `prompt_provenance_centroid_hop2` | `0.6964` | `0.6629` | 幻觉更低 |
| `prompt_provenance_spread_hop2` | `0.6879` | `0.6548` | 幻觉更低 |

这是目前最强的单结构筛查结果，但要正确解读。

- 它说明 response source 的 prompt 来源位置和传播范围有信息。
- 一跳强于两跳，提示最直接的 response relay 更可靠；路径越长，attention 乘积衰减、多个 prompt source 混合和 sparse censoring 都会积累。
- “更低”不等于“prompt grounding 更弱”。centroid 低可能只是来源更靠 prompt 前部；spread 低可能是证据变得集中，也可能是只剩少数 source。
- 这些方向仍是标签后验筛查，下一步必须在 train 上冻结层、方向和校准。

### 5.5 RR 谱子空间：最强的已冻结无标签分数

该方法只使用 response-to-response attention，并保留全部 layer/head channels。

对 prefix 结束于 token `t`、channel `c=(l,h)`、历史 response source `j`，计算它在后续 response 中收到的平均支持：

\[
d_{c,t,j}=\frac{\sum_{u=j}^{t}A_c[u,j]}{t-j+1}.
\]

再减去该 source 的 self diagonal：

\[
\lambda_{c,t,j}=d_{c,t,j}-A_c[j,j].
\]

因为 causal adjacency 和相应 Laplacian 是下三角矩阵，代码直接使用对角元素，不做完整特征分解。每个 layer/head 取绝对值最大的 5 个带符号模式，得到：

\[
32\times32\times5=5120
\]

维 token 表征。train 上按相对位置 bin 做 median/MAD 标准化，再用 PCA 学习常见联合子空间。异常分数是 PCA 正交残差均方：

\[
E_{RR}(t)=\frac1K\|z_t-\hat z_t\|_2^2.
\]

全量 test：73,994 token，4,594 positives，正类率 `0.06209`。

| 分量 | AUROC | AUPRC | 解释 |
|---|---:|---:|---|
| `rr_raw_residual_energy` | `0.6601` | `0.1344` | 当前最强冻结方向谱分数 |
| `rr_untrimmed_pca_ablation` | `0.6466` | `0.1345` | trimming 不是主要来源 |
| `rr_localized_channel_tail` | `0.6436` | `0.1285` | 多个异常 channel 有用，但弱于全局联合残差 |
| `rr_localized_raw` | `0.6285` | `0.1172` | 局部化损失一部分联合信息 |
| `rr_peak_channel_score` | `0.5632` | `0.0864` | 单一最异常 head 不够 |
| `rp_residual_energy` | `0.4822` | `0.0628` | 直接 prompt block 在该表示下没有正向异常信号 |
| `dynamic_prediction_residual` | `0.4630` | `0.0559` | “下一步更难预测”假设失败 |
| `innovation_norm` | `0.4752` | `0.0582` | 动态创新量失败 |
| `manifold_knn` | `0.4869` | `0.0604` | 一般低密度假设失败 |
| 四项融合分数 | `0.5883` | `0.0815` | 弱分量把最强 RR residual 拖低 |

旧版先平均 layer/head、再用 `AA^T` 对称化、提取 22D HKS/SVD 并做 Mahalanobis，结果只有 `0.5107/0.0623`。失败原因很明确：平均掉 channel identity、丢失方向、删除绝对质量，并把弱 RP 与 RR 混在一起。

这一组结果告诉我们：**RR 联合结构确实包含信息，但“异常”并不等于时间预测失败，也不集中在一个 head。**

### 5.6 Dirichlet 与 logistic-normal：分布假设为什么不成立

实验把 attention composition 按 `role` 或 `provenance` 聚合到概率单纯形，分别拟合 Dirichlet 和 logistic-normal，并在 held-out 数据上比较 log likelihood、协方差和 PIT。

| 表示 | Dirichlet - logistic-normal（nats） | Dirichlet 更优的 group 比例 |
|---|---:|---:|
| provenance | `-5.28` 到 `-4.94` | `0` |
| role | `-0.69` 到 `-0.91` | 很低 |

约 24%–25% 的 off-diagonal covariance 为正，而标准 Dirichlet 天然要求组分间以负相关为主。因此问题不是“参数没调好”，而是分布族太受限。Logistic-normal 更适合做单纯形表示，但它本身仍然没有解释为什么某种 composition 对应幻觉。

### 5.7 Source-reuse / CaSH 判别：为什么训练看起来很好，检测却随机

这条线把真实 source history 当正样本，把重连或 shuffle 后的 history 当负样本，训练二分类器。

烟测中：

- 732 / 751 个 contrast score 精确为 `-1.0`；
- 只有 14 个不同 float32 分数；
- 正样本概率几乎为 1，负样本约 `1e-10`。

但 token 幻觉检测只有：

| 分数 | AUROC | AUPRC |
|---|---:|---:|
| current negative margin | `0.5083` | `0.0335` |
| birth negative margin | `0.5080` | `0.0335` |
| dynamic negative margin | `0.4919` | `0.0321` |
| dynamic shuffled NLL | `0.5376` | `0.0377` |

这里既有实现问题，也有更根本的目标错配。

- 负样本没有充分匹配位置、memory norm、历史深度和候选数量，分类器能走捷径。
- sigmoid 饱和后失去排序信息，训练损失继续下降也没有意义。
- 即使负样本做得更难，“真实图还是人工图”仍不等价于“正确还是幻觉”。模型学到 exact source identity，并不自动获得事实判断能力。

因此这条线不应通过更复杂的 InfoNCE 继续硬救；最多作为“exact topology 是否可识别”的结构控制。

### 5.8 Multiplex graph recovery：为什么“正确更容易恢复”只有极小效应

30 个样本、6,866 token、415 positives 的结果：

| 量 | hallucination - correct | 方向 |
|---|---:|---|
| recovery | `+0.00280` | 正确略易恢复 |
| edge recovery | `+0.00312` | 正确略易恢复 |
| diagonal recovery | `-0.000212` | 相反 |

结构控制增益也很小：message aggregation `2.75e-05`，layer order `2.31e-04`，head identity `1.68e-05`，exact endpoint `4.33e-06`，而 full channel / layer-head joint 甚至为负。

不能简单归因于“代码没写好”。核心问题是 target 仍然是 attention 数值。正确和幻觉都来自同一个 Transformer，attention 中存在大量局部冗余；幻觉一旦进入稳定续写，完全可能同样好恢复。全通道任务在 30 个样本上又过难，导致结构增益被噪声淹没。

### 5.9 Typed lineage：把 provenance 压成状态后丢了什么

历史实现把 source 分成直接 prompt、继承 prompt 的 response、response-base 或不确定状态，再计算：

| 特征 | 历史定义 | AUROC / AUPRC |
|---|---|---:|
| `direct_role` | 当前 token 直接 incoming mass 中 prompt/response role 组成 | `0.5290 / 0.0914`（另一版本约 `0.5841 / 0.1031`） |
| `inherited_response_base` | 通过一跳 RR source 继承的 response-base 质量 | `0.5326 / 0.0950` |
| `multihop_response_base` | 多跳传播后的 response-base 质量 | `0.5263 / 0.0958` |
| `origin_transition_gap` | 直接 role 与继承 provenance 类型之间的差距 | `0.5450 / 0.0962` |
| `endpoint_concentration` | 路由质量在 source endpoints 上的集中度 | `0.4813 / 0.0772` |
| `lineage_margin` | prompt-rooted lineage 与 response-base/不确定 lineage 的差 | `0.4515 / 0.0702` |

旧代码已清理，部分阈值和状态 schema 需要从历史提交恢复，因此这里只给出能确认的运算含义。

结果弱的主要原因不是 provenance 完全无用，而是**把连续的 layer/head/source 结构过早离散成少数类别**。一跳 centroid/spread 强，而 lineage margin 弱，正好说明“来源的具体位置与分布”比“归为 grounded / ungrounded 状态”更有信息。

### 5.10 Causal walk / De Bruijn：高阶路径为什么没带来预期增益

这类方法先把每个 token 的路由压成若干 typed states，再拟合一阶、二阶或三阶状态转移。

| 特征 | 如何得到 | AUROC / AUPRC |
|---|---|---:|
| `order1_error` | 用当前/前一状态预测下一 typed route，取负对数似然或预测误差 | `0.6256 / 0.1133` |
| `order2_error` | 用长度 2 历史预测下一状态 | `0.5419 / 0.0927` |
| `order2_path_gain` | order-2 相对 order-1 的误差下降 | `0.5224 / 0.0753` |
| `order3_path_gain` | order-3 相对低阶模型的误差下降 | `0.5203 / 0.0782` |
| `response_persistence` | response-local 状态连续维持的程度 | `0.5147 / 0.0782` |
| `lock_in` | 高 response persistence 与低 prompt anchoring 的人工组合 | `0.5106 / 0.0756` |
| `recoupling_failure` | 进入 response-local 状态后未返回 prompt/evidence 状态 | `0.4997 / 0.0757` |
| `evidence_escape` | 从 prompt-anchored 状态向外移动 | `0.4793 / 0.0729` |
| `anchor_js_mean` | 当前 source distribution 与 prompt anchor 的 JS divergence | `0.2475 / 0.0469`，方向强烈相反 |

高阶方法失败有三层原因。

1. typed state 已经损失大量连续信息，高阶路径是在低信息符号上继续建模。
2. 阶数越高，可观察路径组合越稀疏，估计方差迅速增加。
3. 正常生成本来就会越来越依赖 response；“未回看 prompt”不是普适错误。

随后 full typed-path score 看似达到 `0.6603/0.1328`，rupture 单项 `0.6629/0.1402`，但与绝对位置 Spearman 分别为 `0.928` 和 `0.974`。CUSUM 和 prefix accumulation 会随生成自然累积，所谓 rupture 主要是位置计数器。这是明确的混杂，不是“控制条件太严”。

### 5.11 Holonomy 审计：它证明了图结构，但没有证明幻觉机制

无标签 held-out 审计比较真实结构预测与匹配基线：

| 结构控制 | gain | 具体含义 |
|---|---:|---|
| depth transport | `+0.05663` | 同一 `(source,target)` 上一层的完整 head profile，比 layer mean 更能预测下一层 |
| query-set | `+0.02642` | 同一 `(target,layer)` 的其他 incoming events，比 metadata-only 更能预测 held-out event |
| relay transport | `+0.00590` | 真实 `(u->s,l-1)` predecessor 比 typed mean 略好地预测 `(s->t,l)` |
| exact-path rewire | `+0.00569` | role、lag、观测 head 数匹配后，真实 middle-token path 仍略优于 rewire |
| diamond coverage | `99.56%` | depth/relay diamond 几乎都能构造 |

它支持保留 depth 和 query-set；relay/exact path 只能作为弱辅助。coverage 高只代表 holonomy 可计算，不代表 holonomy 与幻觉有关。

### 5.12 当前 HoloRoute：它现在是什么

每条 prompt-response 样本构造一张 event graph。节点是 `(source,target,layer)`，属性是该层完整 head profile；边/集合包括 depth、relay、query group 和 diamond。训练随机遮蔽 event head profile、删除部分 relay，再从邻居恢复 attention。

模型内部确实得到每个 event 的 embedding：

\[
Z\in\mathbb R^{E\times d}.
\]

但当前 detector 没有直接在 embedding 上做异常检测，而是手工指定六个 residual：event、depth、relay、query、depth-relay disagreement、holonomy，再做条件化联合评分。

所以 HoloRoute 当前最准确的定位是：

> **一个多层 attention-event graph masked autoencoder 基线。**

它的价值是提供统一构图、节点表示、结构消融和 Flat-1024 对照。它目前没有建立新的幻觉机制，也没有证明复杂图模型优于高维 attention 本身。

---

## 6. 效果不显著，到底是哪一类问题

### 6.1 假设从一开始就不对

最主要的问题在这里。

- “幻觉是低密度异常”没有得到支持。RR manifold-kNN 接近随机。
- “幻觉更难恢复”只出现极小效应，且 diagonal 相反。
- “幻觉一定更少看 prompt”被 onset 的 prompt share 上升直接反驳。
- “高阶路径一定比一阶关系更有用”被 order-2/3 path gain 接近随机反驳。
- “不回到 prompt 就是 recoupling failure”忽略了正常语言生成也必须依赖已生成上下文。

### 6.2 表征把信号压没了

这是第二大原因。

- 平均 1024 个 layer/head 后，旧 HKS/SVD 基本随机。
- `AA^T` 对称化丢掉因果方向。
- 把 provenance 离散成几个状态，弱于保留 prompt 位置矩。
- 把 RR 和无信号的 RP、动态预测、kNN 强行融合，反而降低结果。
- 固定 top-K event 是显存措施，不是理论选择；若 K 太小，会剪掉分散但重要的证据路径。

### 6.3 自监督目标与幻觉不一致

- GraphMAE 式恢复学的是数据冗余，不是事实支持。
- 人工 rewire 二分类学的是“像不像真实图”，不是“是否正确”。
- next-state prediction 学的是常见语言路由；一个稳定错误同样可预测。

### 6.4 混杂造成了假信号

- token 位置是最严重的混杂，typed-path rupture 就是例子。
- response 长度、可用 source 数和 retained edge 数都会改变 degree、lag 和 coverage。
- 用标签选最佳 layer/head 和方向，会高估可部署性能。
- 不按 source document 切分，可能把同一证据来源泄漏到 fit 与 test。

### 6.5 数据本身有边界

- sparse floor 使弱边只知道“低于阈值”，不知道精确值。
- prompt query rows 不完整，图主要描述 response 生成阶段。
- attention 没有 value、hidden state 和 MLP，因此不能判断信息是否被下游真正采纳。
- RAGTruth 的 word/span 标签要映射到 model tokens，边界 token 会有噪声。
- teacher forcing 看不到错误真正改变未来上下文后的动态。

### 6.6 确实存在过代码问题

这部分不能回避，但也不能把所有 null result 都怪到 bug 上。

- 退役 routing dynamics prototype 有未来层拓扑泄漏、标签对齐和多 GiB 内存问题。
- 早期版本有把 censored head 当零、分数 artifact 与标签边界不够严格的问题。
- CaSH 负样本不匹配，判别器 sigmoid 饱和。
- 后期 HoloRoute 还连续修过 OOM、AMP dtype、CUDA efficient-attention batch limit 和 API 参数名。

这些 bug 会影响相应版本的可信度；但它们无法解释为什么多种独立实现都显示“稳定错误可以很好预测”。后者更像目标假设的问题。

### 6.7 控制条件并不“太严”

位置残差化、source-group split、matched rewire、Flat-1024 对照和标签冻结不是额外刁难，而是判断图是否真的有价值的最低条件。一个分数若在这些控制后消失，说明之前的效果多半来自位置、长度、高维数据量或标签后验选择，而不是应当保留的机制。

---

## 7. 目前可以放心保留的证据

1. **layer/head 联合结构重要。** 平均 channel 会损失明显信号。
2. **RR 比直接 RP 更稳定。** 当前最强冻结方向分数来自 RR 联合谱残差。
3. **response source 的 prompt provenance 比直接 prompt 比例更有信息。** 一跳 provenance centroid/spread 是最强单结构筛查。
4. **幻觉起点常伴随注意力集中变化，但不一定伴随 prompt share 下降。**
5. **depth 和 query-set 有明确 held-out 预测增量。** 图结构不是完全的装饰。
6. **exact relay path 的增量存在但很小。** 不能把复杂多跳路径写成主结论。
7. **稳定性不是正确性的同义词。** reconstruction、prediction 和 local density 都不能单独作为机制。

把这些结论放在一起，最自然的问题不是“怎样造一个更强的图 autoencoder”，而是：

> 当前 token 依赖的 response 路径，究竟还在转运 prompt 证据，还是已经可以脱离 prompt 自己闭环？

这就是下一份方法文档选择的 gap。

---

## 8. 明确的研究 gap

现有方法各自覆盖了问题的一部分：

- Lookback 区分直接 prompt 与 response；
- CHARM 证明监督图模型可以学习复杂关系；
- TOHA/HalluZig 描述拓扑和跨层演化；
- RFS-Guard 描述语义相近历史上的自我确认；
- CausalGaze 用监督梯度做结构敏感性；
- 通用图异常方法提供无标签表示学习。

但仍缺少一个方法，同时满足：

1. 不使用 hallucination label 训练；
2. 输出逐 token 分数；
3. 保留 layer/head 与 exact source；
4. 允许 response token 合法地中继 prompt 证据；
5. 不把“稀有、难恢复”预设为错误；
6. 直接检验当前 token 对 prompt-rooted 路径和 response-closed 路径的相对依赖。

更简洁地说，缺的不是另一个“attention 异常量”，而是一个**证据依赖检验**。

下一份文档给出一个只围绕这一点展开的研究范式：P-Cut（Prompt-Provenance Cut）。

---

## 9. 后续方法必须遵守的边界

- 当前 HoloRoute 只作为图编码与 masked reconstruction baseline。
- 不再把六个弱 residual 手工加权成“机制分数”。
- 不再用人工真假图二分类作为主训练目标。
- 不再用 CUSUM、prefix sum 或任何随位置机械增长的量。
- 不用 test label 选择 layer、head、方向或权重。
- 不声称 attention edge 是真实功能贡献；加入 value/hidden state 前，只能说 routing provenance。
- 任何图创新都必须同时胜过 Flat-1024 和 matched topology destruction。
- 任何“因果”说法都必须有模型内实际干预；仅重跑图编码器只能称为 counterfactual graph view 或 predictive sufficiency。

---

## 10. 主要参考文献

- Chuang et al. [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/), EMNLP 2024.
- Frasca et al. [Neural Message-Passing on Attention Graphs for Hallucination Detection](https://arxiv.org/abs/2509.24770), 2025.
- Binkowski et al. [Hallucination Detection in LLMs Using Spectral Features of Attention Maps](https://arxiv.org/abs/2502.17598), 2025.
- Bazarova et al. [TOHA](https://aclanthology.org/2026.acl-long.704/), ACL 2026.
- Samaga et al. [HalluZig](https://aclanthology.org/2026.eacl-long.159/), EACL 2026.
- Liu et al. [RFS-Guard](https://aclanthology.org/2026.acl-long.885/), ACL 2026.
- Xu et al. [Reasoning Fails Where Step Flow Breaks](https://aclanthology.org/2026.acl-long.1212/), ACL 2026.
- Shi et al. [CoDA](https://aclanthology.org/2026.findings-acl.576/), Findings ACL 2026.
- Ruscio and Thompson. [The Phenomenology of Hallucinations](https://arxiv.org/abs/2603.13911), 2026.
- Kong et al. [CausalGaze](https://arxiv.org/abs/2604.11087), 2026.
- Hou et al. [GraphMAE](https://arxiv.org/abs/2205.10803), KDD 2022.
- Liu et al. [CoLA](https://arxiv.org/abs/2103.00113), TNNLS 2021.
- Qarkaxhija et al. [De Bruijn Goes Neural](https://proceedings.mlr.press/v198/qarkaxhija22a.html), LoG 2022.
- Bodnar et al. [Neural Sheaf Diffusion](https://proceedings.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html), NeurIPS 2022.
