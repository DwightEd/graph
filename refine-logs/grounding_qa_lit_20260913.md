# QAFactEval / FactGraph：以关系问题固定规范证据锚点

日期：2026-09-13。接续 `grounding_architecture_lit_20260913.md`，仅精读指定两篇论文和官方源码；未运行 GPU、未下载模型、未改实验代码。

书目状态均为 **UNVERIFIED（deterministic helper unavailable）**；已人工读取原始论文方法、附录与固定提交源码，不等于科学审查通过。

## 1. 论文与代码事实

### QAFactEval，Fabbri et al.，NAACL 2022

**论文方法。** 最优配置从 summary 选 noun-phrase chunks，以答案和上下文为条件，由 QA2D 训练的 BART-large 出问题；Electra-large 回答。先在 summary 上过滤不可回答的问题，并要求答案 F1≥约 0.60；在 source 上不可回答则匹配得分置零。可回答时用 LERC(QuIP) 估计语义答案匹配，再平均问题分数。LERC 使用 MOCHA 的人工答案质量监督。作者明确指出：同一答案虽出现在 source 中，但属于另一语境，强制最佳答案会让匹配分数误判；answerability 是独立必要部件。这里没有证明 QG 不泄漏答案，也没有生成器内部定位。[§3.2、§5.1、Appendix A.3](https://arxiv.org/html/2112.08542v2)

**固定代码。** `salesforce/QAFactEval@01177f11cc05f8b3e75511a0cfa54bac19324d09`。`get_filter` 实际要求 `answer_probability > null_probability` 且 F1 **严格大于** 0.60。随后仅用 source 回答保留问题。LERC 输入顺序为问题、原选答案、source 预测答案、source context；概率不超过 null 时直接记零。匹配器上限 512 token，会截断输入。核心 QA/QG 由未固定版本的 `qaeval` 依赖提供，不能把当前包装代码当全部模型代码已复现。[过滤及调用代码](https://github.com/salesforce/QAFactEval/blob/01177f11cc05f8b3e75511a0cfa54bac19324d09/qafacteval.py#L15)，[LERC 代码](https://github.com/salesforce/QAFactEval/blob/01177f11cc05f8b3e75511a0cfa54bac19324d09/lerc_quip.py#L11)

### FactGraph，Ribeiro et al.，NAACL 2022

**论文方法。** AMR 关系标签变为关系节点，再展开到 subword 图。文本/图共享冻结 ELECTRA-base，以不同的 32 维 adapter 编码；结构 adapter 使用区分入/出方向的 RGCN。句图 mean pooling，summary 图作 query 对文档句图加权，与文本 CLS 拼接分类；只选相似度最高五句图。FactGraph-E 将 summary 图与每个来源句图成对编码，把每个边端点的文本和图特征合并后分类。边的负标签来自人工幻觉 span 与节点对齐；任一边负则 summary 负。AMR 节点预测预训练是自监督，但最终事实/边级分类不是。AMR 会抽象掉部分时态信息。[§3.2–3.5、§4.2、§5.2–5.3、Appendix B](https://aclanthology.org/2022.naacl-main.236.pdf)

**固定 adapter 代码。** `amazon-research/fact-graph@165285a483ebf2fc4ffe2202c65349b99dea9626`。普通 adapter 是 `LN→d→32→ReLU→d→残差`；结构 adapter 是 `LN→RGCN(d,32,2方向)→ELU/dropout→d→残差`。实际代码把普通 graph adapter 放 FFN 前、结构 adapter 放 FFN 后，与论文所写顺序相反。需区分论文结构和此版本代码。[modeling_electra.py:580–686](https://github.com/amazon-research/fact-graph/blob/165285a483ebf2fc4ffe2202c65349b99dea9626/src/transformers/models/electra_adapter/modeling_electra.py#L580)

**固定边分类代码。** 每条边拼接 `text_head, graph_head, text_tail, graph_tail`，经 `Linear(4d,2)`；图特征跨有效来源图取均值。交叉熵监督人工衍生的边标签；优化器选入 adapter、classifier/pooler 与 embeddings，词嵌入旧行梯度被清零。它输出语义关系是否事实一致，不是原 LLM 哪个 token 承担回看任务。[models.py:222–338](https://github.com/amazon-research/fact-graph/blob/165285a483ebf2fc4ffe2202c65349b99dea9626/src/models.py#L222)，[main_edgelevel.py:168–245](https://github.com/amazon-research/fact-graph/blob/165285a483ebf2fc4ffe2202c65349b99dea9626/src/main_edgelevel.py#L168)

## 2. 对 round0 的明确收缩建议（以下是我们的设计，不是论文已证实结果）

**建议采用 QA 锚点，暂不从约两千弱例同时学习自由分段、三类语义与多证据归属。** 将“应该采用什么”改成可直接审计的接口：

`当前命题的目标槽 → 去掉该槽答案的关系问题 → 全文来源回答/NULL + 引证区间 → 答案及限定条件一致性`

这减少了几个难以识别的隐变量。每个失败都有对应对象：没覆盖目标槽、问题问错了、来源回答错了、引用不足、答案不一致；不用以最终 loss 下降反推内部确实学到归属。语义锚点第一版冻结，只训练计算图的采用/传播头。图学习的职责是沿真实信息路径解释并传播已有锚点，而不是再造一个同时完成抽取、问答和事实判断的大系统。

| 阶段 | 输入与固定输出 | 必须保持的边界 |
|---|---|---|
| 命题/槽提取 | response 的关系/属性片段，返回原字符范围、待检查值和其余限定条件 | 数字、主体、关系、否定、时态、地点均可能出错；不能只检查名词 |
| 问题构造 | 目标值先遮蔽，生成保留其他条件的问题 | source QA 不接收原目标值、完整 response 或原生 logits |
| 问题自洽 | 用原 claim 回答，检查能恢复所选槽且问题未遗漏决定性条件 | 只说明问题忠实于 claim，不证明 claim 正确 |
| 来源回答 | 全文 source + 无目标答案的问题，输出 answer / NULL / unresolved，附 quote ranges | 不能强迫输出最相似 span；引用须覆盖绑定条件而不只是数值 |
| 对照 | 分别检查类型/单位、值、条件作用域，再映射回原片段 | source 自身有多个合法答案时保留集合，不能只选一项 |
| 计算图 | 把这些引用区间和回答片段锚到原 native 图 | 规范引用与真实采用关系分开，不把引用直接当因果路径 |

无目标答案泄漏要落到输入契约：生成 source QA 请求的函数只接受 `source, question`，不能接受 claim、selected_answer 或其缓存。问题文本排除目标值及直接改写，避免“为什么时长是十二分钟”式预设；源引用是输出，不能由原模型的 attention 预先限制到 top-k。原目标值只在 source QA 完成后交给答案比较器。语义防泄漏不能仅靠字符串查重，必须允许问题无效而弃权。

对原案例，应该问完整阶段条件下洋葱步骤的时长，source QA 应能够返回无支持时长，而不是在来源数字 12 和 14 中强行择一。这个例子用于定义接口，不建议再扩一组手工 Before/After 小实验。

## 3. 不可忽略的失败边界

1. **问对了不代表答对了。** summary round-trip 是内部一致性；source QA 仍可能把错误阶段的同一数字当答案。必须单列来源可回答性和条件覆盖，不能直接把 QA 结果当无噪声真值。采用同一个 8B 生成、提问、回答、验证会有相关错误；增加独立验证也不构成真值保证。
2. **NULL 不能兼任所有失败。** 真正无来源支持、QA 不确定、问题不成立、来源检索遗漏，需要不同状态。只有已覆盖全文且问题有效，才有资格把明确不可回答用于“不支持”证据；不确定时返回 unresolved。NULL 表示提供材料没有支持，不能宣称世界事实为假。
3. **问题覆盖决定可定位范围。** NP 选择会漏关系、否定、时间和量词；没有产生有效问题不能得满分。必须输出总槽数、有效槽数、弃权范围。只检查数值所得的低风险，不能推广到整句正确。
4. **一个错误前提会影响多个问题。** 若 claim 同时把主体和时间写错，针对数值的问题可能因主体不存在而 NULL；不能由此精确宣称只有数值 span 错。需要保留问题依赖的前提范围，报告局部化歧义，不能把问题目标槽当唯一事实错误位置。
5. **多句/多跳不能降成最佳单句。** 同一证据链可能跨 source 句甚至文档；引证集合需要共同覆盖条件。处理长 source 时应记录全量覆盖与跨块组合，不能用 chunk max 冒充全文问答。source 内冲突同样要保留多答案/冲突状态。
6. **答案相同不等于关系相同。** 即使 exact-match=1，也要检查所在事件、主体和作用域；语义 overlap 高更不能保证约束吻合。对区间、单位、近似词和否定不能用裸字符串匹配。
7. **规范证据不等于采用证据。** QA 引文只描述独立 reader 的支持依据。native 分支仍需实测有符号影响，才能称错误路由或历史错误延续；仅 QA 不一致只能称候选不支持。
8. **QA 是离线语义接口。** 它使用已完成 claim，不能作为提前定位首错的证明。回看组仍应由该 claim 的实际计算效应独立定位；不追求一个没有真实监督保证的“100%唯一回看点”。

第一版可把 question/claim/source-citation 看成有文本和高维特征的锚点节点，把引用、前提依赖、native message 与历史传播作为不同边类型。证据关系直接保留到读出，避免全图池化成一个向量。自适应分段先根据共享命题/前提和采用路径做可重叠分组，保留全部跨组边；等这些可审计锚点稳定，再考虑学习自由边界。

这使方法的可实现性与失败可诊断性强于 round0，但不构成已验证收敛。最先需要闭合的是无泄漏问题覆盖、source answerability 与带条件引证；如果这三项仍不可靠，扩大图头不会补救它们。
