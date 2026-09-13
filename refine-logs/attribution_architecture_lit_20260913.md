# 面向当前陈述的有符号路由归因：方法细读与结构建议

日期：2026-09-13。范围：ALTI、自回归 ALTI、DecompX、自回归 token 分解、EAP-IG、Circuit Tracing 与 QK 扩展。本文是方法提取与设计推导，不是实验通过报告；没有使用 GPU，没有修改冻结的全量实验。按 `research-lit` 执行，来源为论文全文与作者代码。引用验证辅助程序不可用，以下各篇均保留 **UNVERIFIED（helper unavailable）**；已人工核对所链接的一手正文、标题和方法位置，不能把人工阅读升级为辅助程序验证成功。

## 1. 各方法究竟计算什么

### ALTI：混合图，不是输出采纳图

Ferrando、Gállego、Costa-jussà，*Measuring the Mixing of Contextual Information in the Transformer*，EMNLP 2022。**UNVERIFIED（helper unavailable）**。

论文 §2.1 将 attention、残差和归一化分解为向量；§3 Eq.9–11 定义

\[
d_{ij}=\|y_i-T_i(x_j)\|_1,\quad
C_{ij}=\frac{[\|y_i\|_1-d_{ij}]_+}{\sum_k[\|y_i\|_1-d_{ik}]_+},\quad
R^L=C^L\cdots C^1.
\]

它保留向量方向对局部相似度的影响，但随后变成非负、行归一化的标量，并不穿过输出分类头，原始分析未传播 FFN 非线性。论文在分类和主谓一致上的忠实性评价优于若干基线；这不是约束归属或真假判定。因负项被截断，它不适合作为“反对当前陈述”的量。[全文 §2–3](https://aclanthology.org/2022.emnlp-main.595.pdf)

### 自回归 ALTI：可分 source/history，仍不能决定适用性

Zaranis、Guerreiro、Martins，*Analyzing Context Contributions in LLM-based Machine Translation*，Findings of EMNLP 2024。**UNVERIFIED（helper unavailable）**。

§2.2 在完整 prompt+translation 上建立因果掩码的 \((S+T)^2\) 贡献矩阵，逐层 rollout；§2.3 对来源部分的 token 求和、对生成 token 求平均。实验采用 Llama-2/Tower，发现低来源贡献可用于分析异常翻译。它说明 source/history 可被分开统计，但没有辨别同一来源中的错误实体、阶段和条件，也没有给出回看节点检测器。[全文 §2.2–2.3](https://aclanthology.org/2024.findings-emnlp.876.pdf)

### DecompX：保留高维向量到输出，但“可重构”不等于干预忠实

Modarressi、Fayyaz、Aghazadeh、Yaghoobzadeh、Pilehvar，*DecompX: Explaining Transformers Decisions by Propagating Token Decomposition*，ACL 2023。**UNVERIFIED（helper unavailable）**。

§3 保存 \(x_i^\ell=\sum_kx_{i\Leftarrow k}^\ell\)，在实际 attention 下传播各来源向量，并最终穿过分类头。§3.2 保留残差、用完整输入的 LN 分母；§3.3 将激活替换为固定斜率 \(\theta(a)=f(a)/a\)，在观测点精确重构。bias 采用 AbsDot 分摊。输出分量有正负，无需局部范数 rollout。原实验覆盖 BERT/RoBERTa 分类。固定 attention/斜率的分解并不等于删除来源后重新计算全网络；原 FFN 公式也不是 Llama 的双支路 SwiGLU。[全文 §3、Eq.1、4–11](https://aclanthology.org/2023.acl-long.149.pdf)

### 自回归 token 分解：输出相关的多 token 效应，但保持了原路由

Oh、Schuler，*Token-wise Decomposition of Autoregressive Language Model Hidden States for Analyzing Model Predictions*，ACL 2023。**UNVERIFIED（helper unavailable）**。

§3 将 logits 写为 \(z_i=\sum_kz'_{i,k}+b_i\)，单独保存 bias-like 项。MLP 使用观测点切线斜率与截距，LN 分母及 attention 取实际前向值。Eq.22–24 定义

\[
\Delta LP_k=\log p(y)-\log\operatorname{softmax}(z-z'_k)_y,
\]

可联合移除一个词的子词分量，并累加多输出 token 的 log 概率差。研究使用 OPT-125M，主要分析高正贡献；该删除发生在分解后的输出，不能等同输入重跑或识别 QK 路由原因。[全文 §3、Eq.18–24](https://aclanthology.org/2023.acl-long.562.pdf)

### EAP-IG：便宜筛选之后必须检验整个子图

Hanna、Pezzelle、Belinkov，*Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms*，COLM 2024，arXiv:2403.17806。**UNVERIFIED（helper unavailable）**。

§2–3：EAP 用 \((z'_u-z_u)^\top\nabla_vL\) 近似替换边的效应；EAP-IG 固定该激活差，平均 clean/corrupted **输入 embedding 插值路径**上的梯度。它不是对每条内部边做精确积分。§4 用删去子图外全部边后的任务表现检验忠实性，保留绝对分数大的正、负边；论文明确独立边排序仍会失败。实验在 GPT-2 small 六任务中，EAP-IG 整体优于 EAP，但不是每个任务达到真实 patching。[正文 §2–4、Eq.1–3](https://arxiv.org/html/2403.17806v2)

### Circuit Tracing：重要的是输出方向与误差记账，不是必须训练图网络

Lindsey、Ameisen 等，*Circuit Tracing: Revealing Computational Graphs in Language Models*，Transformer Circuits Thread，2025 技术报告。**UNVERIFIED（helper unavailable）**。

“Local Replacement Model / Attribution Graphs”用跨层 transcoder 替代 MLP，加入逐点重构误差，冻结 attention 与归一化分母，再计算有符号的 activation×Jacobian 边。观测 logits 可精确复现，但原文明确这不保证机制相同。该图遗漏影响 attention 模式的 QK 路径；supernode 分组仍需人工判断。我们可借鉴输出方向、误差节点、组干预验证，不能把这一版直接当自动路由解释器。[方法、限制与 supernode 小节](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

### QK 扩展：定位“为什么看这里”还需连接下游效果

Kamath、Ameisen 等，*Tracing Attention Computation Through Feature Interactions*，Transformer Circuits Thread，2025-07-31 技术报告。**UNVERIFIED（helper unavailable）**。

“QK attributions”将 query/key 的特征、bias、误差分别相乘，把 pre-softmax score 写为双线性分量之和；“Head loadings”追踪哪些头承载 feature 边。跨多层头路径数量增长，作者用 attention-output SAE、逐层 residual SAE 或 multi-token transcoder 做中间节点。原文提醒 RoPE 需位置相关变换，归一化需线性化；这些分解解释 score，并不独自证明某证据应适用或被下游采纳。[全文对应方法与 Future work](https://www.transformer-circuits.pub/2025/attention-qk/index.html)

## 2. 官方代码核对后的移植边界

读取 `deep-spin/interp_llm` 提交 `a43b3834c878a84f3bb5bd4d833ed041c3eea675`（2024-10-17）：`src/contributions.py` 的 `l_transform` 虽区分 RMSNorm 是否减均值，但分母仍从 `torch.var` 得到；`get_values_weights` 又按 attention heads 重排 V，未适配我们 32 query heads / 8 KV heads 的 GQA。`utils_contributions.py::compute_alti` 明确先截断并归一化、再矩阵乘。因此只借用方法定义，不直接复制该实现到 Llama3.1。这里是所读提交的代码检查结论，不是对论文结果的重审。[固定版本代码](https://github.com/deep-spin/interp_llm/blob/a43b3834c878a84f3bb5bd4d833ed041c3eea675/src/contributions.py)

`hannamw/eap-ig-faithfulness` 链接作者 EAP-IG 库；已读取当前 `e24bafd3d22af2c59ac5a83e0a6581b89b5c05dd` 的 `attribute.py`，确认其实现依赖 TransformerLens 的模块 hook、clean/corrupted 激活差及 backward 评分。我们已有原生 HF 干预代码，无需为使用同一公式迁移整个模型加载框架。[作者库固定版本](https://github.com/hannamw/EAP-IG/blob/e24bafd3d22af2c59ac5a83e0a6581b89b5c05dd/src/eap/attribute.py)

## 3. 对我们方法的直接设计结论（以下为本项目推导）

不能再用“attention 质量 × hidden cosine”充当最终幻觉分数。应实现一个**当前陈述条件下的测量器**：同一冻结骨干同时提供高维消息、路由位置、输出方向；每项分数都指明对应的可执行干预。图是这一计算的组织结构，第一版不增加 GNN 或 graph adapter。

### 3.1 三个概念必须分开建模

1. **适用性**：来源陈述是否给当前实体、动作、阶段、条件提供约束。它是语义归属问题。
2. **实际使用**：该来源经过哪些 query/layer 对当前输出产生了什么方向的作用。它是模型机制问题。
3. **正确性**：实际输出是否符合适用证据。高“使用”不能自动回答高“正确”。

归因图本身没有适用性真值。模型可以正确地把错误对象的信息复制到输出，产生非常清晰、强正向的路径。必须独立提供包含实体、谓词、时间/阶段、否定、条件及 **NULL/未给出该约束** 的陈述—证据对应关系。即使使用同一 LLM 的另一种提示生成这种对应关系，也只是可错的估计，不能把它当评测真值。

### 3.2 输出目标：指向当前陈述，而不是 residual 自身

设当前待解释陈述 token 集为 \(C\)，真实生成的 token 为 \(y_t\)。无正确答案标签时，最基础目标是

\[
F_C=|C|^{-1}\sum_{t\in C}\log p(y_t\mid x,y_{<t}).
\]

它回答“哪些信息推动模型说出这段话”，不回答这段话是否正确。一个明确的竞争陈述 \(\tilde y\) 可由自动证据归属模块产生时，可以用两条完整、多 token 陈述的条件 log 概率差，不能固定 12/14、也不能用标注选对比项。目标模板、长度处理和候选产生方式需要冻结。

log 概率在高置信度位置可能饱和。可保存并预先选择 token log-odds \(z_{y_t}-\log\sum_{v\ne y_t}\exp z_v\) 作为另一种目标；它和 log 概率不是同一量，应比较局部干预误差后选择，不能看到结果再改方向。多 token 目标必须记录每个 token 的贡献，防止少数格式 token 掩盖约束词。

该目标从已生成陈述出发，天然是回溯检测。若宣称在首错之前实时触发，只能使用当时已经生成的前缀，不能把未来完整陈述的梯度当实时特征。

重放时还固定了真实回答的离散 token 历史。此时测得的是“给定这段历史后，内部路径怎样影响当前条件分布”，并不包含某个早期干预改变已生成 token、再改变全部后续文本的自由生成路径。验证连续错误的生成范围时，需要另行区分固定历史的条件效应与重新生成的总效应，不能混报。

### 3.3 有符号内容与路由效应的最小公式

省略 layer/head 下标，令

\[
\bar v_i=\sum_jA_{ij}v_j,\qquad
m_{ij}=W_O(A_{ij}v_j),\qquad
g_i^C=\frac{\partial F_C}{\partial o_i},
\]

其中 \(o_i\) 是该层 attention 的真实输出，\(g_i^C\) 必须穿过剩余真实网络计算，保留所有下游 residual、RMSNorm、SwiGLU、attention/QK 导数。

**内容 gate。** 给指定来源消息乘 \(\lambda_{ij}\)，在 \(\lambda=1\) 时：

\[
\frac{\partial F_C}{\partial\lambda_{ij}}
=(g_i^C)^\top m_{ij}.
\]

正值表示微弱抑制该消息会降低当前陈述目标；负值表示该消息在当地反对该目标。这是局部方向，不是 100% 删除效果或事实支持。

**路由 gate。** 在 pre-softmax score \(s_{ij}\) 加 \(\rho_{ij}\)，其精确局部导数为

\[
\frac{\partial F_C}{\partial\rho_{ij}}
=A_{ij}(g_i^C)^\top W_O(v_j-\bar v_i).
\]

这一竞争项不可省略：提高某条边意味着挤出其他边；若所有候选 value 下游效果相同，改变路由就不改变当前输出。对证据 span \(S\) 加统一 logit bias，导数为该式对 \(j\in S\) 求和。不要将 signed 值做 min-max、softmax 或强制行和为 1。可以分别保存正、负总量、净量及绝对量，但绝对量只用于筛选，不能解释为采纳。

实际 query/key 使用 RoPE 后向量。由链式法则，路由 score 导数可继续传回 Q、K；GQA 中每个 KV head 接收其对应 query heads 的梯度和。只替换 A 的 E 干预并不能证明“就是 K 出错”，需要 Q 与 K 的分离方向和重跑验证。

**残差、MLP。** 同样给实际 residual 分支、MLP 输出设有名字的 gate，记录对 \(F_C\) 的导数。不能将同一信息沿途多个节点的全局效应相加后称作一份守恒的“信息质量”；显式联合 gate 的导数和是该联合干预的一阶效应，但不是唯一分摊的因果贡献。

### 3.4 非线性与联合回看：从候选到经过验证的位置组

单点梯度用来筛选，不用来宣布发现回看节点。对给定 query×layer×证据组定义连续 gate \(\boldsymbol\lambda\)，其余网络始终重新计算。沿同一组 gate 的路径积分

\[
I_G=\int_0^1\nabla_{\boldsymbol\lambda}F_C(\boldsymbol\lambda(\alpha))^\top
\frac{d\boldsymbol\lambda(\alpha)}{d\alpha}\,d\alpha
=F_C(\boldsymbol\lambda(1))-F_C(\boldsymbol\lambda(0))
\]

是有清楚端点的整体效应；有限积分需报告闭合误差。这是本项目可实现的 gate 路径定义，不能标作 EAP-IG 原论文公式。零 gate、uniform-source gate 或替代来源各自代表不同干预，需分别记录，不能把零基线包装成自然反事实。

自动选择应寻找**位置组**及其作用区间，而不是全样本唯一 token：

- 从当前陈述反向获得所有过去 query 的有符号 source/history/MLP 作用轮廓；先按结构邻接和轮廓相似度产生合并候选，不截掉跨段来源。
- 以保留组内、干预组外后的输出保持程度筛选充分子图；再干预选中组，测试必要性。两者是不同问题，需要同时保留。
- 逐步删/合并候选组直到再缩小就明显损失预定义的保真度；保存多个近等价组。O5 已出现末位置充分与前面联合位置充分，这种情况不应该被强制压成唯一回看点。
- 在独立来源检验输出保持、组删除效应、组大小、跨重述稳定性。attention-rise top-k 命中某点不是充分评价。

早期节点只影响后续 query 的 QK 时，上述全链反传仍有机会找到它；单纯冻结 attention 的 OV 图会系统性漏掉它。高饱和或备份路径还可能让一阶分数低，因此保真度搜索必须允许合并候选并检验联合干预。

### 3.5 分段与幻觉判定应共享同一图，却使用不同读出

图的节点至少含 token/span、layer、source/history 角色及高维表征；消息边含实际 A/V、输出方向和 gate 定义。分段读出使用**证据归属分布与因果作用轮廓的变化**，而不是只用 entropy 或标点。跨 span 边始终保留。

幻觉读出必须比较“适用于当前陈述的证据”和“实际推动当前陈述的证据”：适用来源被路由却下游反对、非适用来源主导、NULL 情况下仍生成新约束，应分别保留为不同模式。过去错误信息被强烈引用却以负方向作用，可能是在纠正，不能统一标作错误传播。语义适用性尚未验证时，只能输出机制测量，不能输出自称有保证的幻觉风险。

## 4. 最小实现接口与接受标准

建议接口分四层，均可使用同一冻结骨干：

1. `ClaimTarget`：当前 span、每 token 的目标、竞争陈述来源、在线/回溯标记。
2. `EvidenceAssignment`：完整证据 span 与角色、适用性置信度、NULL，保留未确定项。
3. `SignedRoutingTrace`：全链 output covector、分开保存 V / pre-softmax routing / residual / MLP 效应，明确位移基线。
4. `CausalGroupSelection`：自动位置组、保留/删除两类真实重跑结果、替代组与影响区间。

第一版使用直接计算与受限组搜索；只有测量器在独立来源有效而速度成为瓶颈时，才考虑从无标签干预数据蒸馏轻量读出或 GNN。不能先堆 GNN 再以其分数代替未解决的归属定义。

工程上无需优化器和第二套 8B 权重，但要激活梯度；冻结参数不等于关闭 autograd。长输入应使用逐层重计算/checkpoint 与按目标分批 VJP，不能为了省显存 detach 过去历史而丢失回看路径。当前完整回答一次 forward 的缓存不足以事后恢复完整 Jacobian，需要重放；O4/O5 的缓存可用于检查输入、A/V 和局部公式，不能伪称已具备跨层全链归因。

接受标准集中在真实欠缺处：自动适用约束检索（含 NULL）在独立来源是否有效；位置组选中后能否保持/改变**当前**陈述，且不无差别改变其他陈述；signed effect 是否预测纠正/延续的干预方向；图分段能否改善 span 级定位。任何一项失败都应修改对应模块，而不是继续验证“关系会影响生成”。
