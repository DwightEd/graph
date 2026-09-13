# Source self-supervised owner pointer head 审查

日期：2026-09-13  
模式：`research-refine` 有界方法评估；真实 collaboration GPT-5.5 fallback，Codex MCP 不可用。范围：评估是否恢复旧 `unsupervised_detector_lit_20260913.md` 第 5 节中的“适用约束头”，但只保留一个 source self-supervised owner/condition pointer head，不训练 hallucination classifier。未写代码、未启 GPU、未改变正在运行的 v3。

## 结论

可以**有条件采纳**，但只能作为 `source_owner_pointer_pretrainer`：一个给 response role/event 提供 source candidate 的轻量指针检索器。它不能输出 supported/conflict/N，不能替代 relation verification，不能替代 A/B contrast，也不能被称为 hallucination detector。

这比继续扩大 Qwen sourceQA 更有意义。v3 运行中已看到的失败不是单纯 JSON framing：source 类请求 parsed 139，其中 answerable 80，但 42 个 `source_answer/source_answer_quote` 仍为 null；134 个 invalid JSON 中 115 个是根后非 framing 内容；样例还会把带 `<MISSING_SLOT>` 的问题句当 `source_quote`，或把 `Additionally,` 这种未解析片段当必要条件后标 missing。一个 source-only pointer head 可以删除“开放问答生成 answer/quote”的接口，让模型只在冻结 source graph 的候选节点中选位置。

但它只解决“规范证据候选供给”。它不解决事实真假、条件适用、模型是否采用该证据、错误是否沿历史传播。若训练或推理时把 owner score / null score 改名成 support/conflict/N，就是回到旧三头方案的不可识别问题。

## 可识别性判断

该头的 source-side 训练目标是可识别的：给定一个来源事件/字段，把目标 value/role 从 query view 中遮掉，保留 entity/action/stage/negation/condition，要求在同一 source candidate view 中恢复原始 source position。正例来自原文坐标，不来自 RAGTruth，不来自合成 hallucination binary label，也不来自当前模型输出。

它在 response inference 上的语义迁移不是自动保证。训练学到的是“在 source 表达内部，哪个 source value 属于这个 owner/context”，而 response 可能是自然改写、漏条件、多个 role 同时错、或把 Data2txt field 改写成谓词句。因此这个头的合格 claim 只能是：提高 candidate supply/top-k recall，降低开放 sourceQA 失败。自然事实准确率仍需 relation verifier + contrast + native 证书或人工评估。

## 防泄漏数据视图

训练必须使用两个分离视图，不能从原 source hidden state 中直接读被 mask 的 target。

### Query view

- 从 source event/field 构造 canonical masked query：目标 role/value 替换为 `<MASK_ROLE>`，其它 roles、entity、action/stage、negation、condition 保留。
- query encoder 只能看 masked query，不能看未遮盖 source sentence、answer surface、source candidate ID、candidate rank、native effect、E/C/N。
- 对 target aliases 做全字段泄漏检查：同 surface、normalized number/date/unit、字符串去引号、bool polarity alias、range endpoint alias、Data2txt display value。若 target alias 出现在 query view 的非目标字段里，实例标 `query_target_alias_leak`，不能训练为普通正例。
- 不给绝对 source char/token position。若使用 transformer hidden，优先在 canonical local serialization 上取向量，而不是完整 source 文档同位置 hidden；否则模型会学“候选 span 就在 mask 原位置附近”。

### Candidate view

- candidate view 是冻结 source graph/raw inventory 的节点列表，包含正例、hard negatives、NULL、UNKNOWN。
- candidate 可以保留原文 value 和 source provenance；这是检索目标。但 scorer 不得使用“candidate 与 query 的绝对位置相同/相邻”作为特征。
- Data2txt candidate 以 `record_id, field_path, value_id, value_type, raw_literal, display_value` 表示。field path 是 provenance feature，不是 predicate truth。

### Response inference view

- 对 response graph 的每个 role 构造同样的 canonical masked query。若 response role 未抽出或 condition residue 未解析，输出 candidate 仍可给出，但 `downstream_allowed=relation_verification_only` 或 `target_dependence_only`。
- response target surface 不进入 query。若为了找冲突需要比较 response value 和 source value，该比较只能在 relation verifier/contrast compiler 中发生，不能进入 owner head 的候选选择。

## 正例、多正例和 ambiguous 的定义

训练实例单位是 `(source_event_or_record, target_role)`。

正例集合 `P(q)`：

- 同一个 source event/record/hyperedge 中，与被遮盖 role/field 绑定的原始 mention/value 节点；
- 若同一事实在 source 中有重复 mention 或同一 field 有多个等价渲染，全部为 multi-positive；
- 若一个 list/record 中 coordination 明确给出多个合法 values，也可多正例，但必须共享同一 owner/context。

同值不同事件默认不是多正例。它是 hard negative 或 ambiguity control。只有当 source graph 已把两个 occurrence 证明为同一事实的重复证据时，才可加入 `P(q)`。如果 identity context 不足以区分两个等值事件，该训练样本标 `ambiguous_positive_context`，不用于普通 cross-entropy。

这条必须写死，否则模型会把“值相同”学成“owner 相同”，正好制造用户担心的错误归属。

## Hard negatives

每个 query 的候选池必须包含同源高混淆负例，而不是随机负例：

- same source、same role/value type、不同 event/record；
- same value different event；
- same field name/path other record；
- same entity different stage/time/condition；
- same predicate/action different subject/object；
- bool polarity opposite field；
- numeric same unit but different range/interval；
- `None`/null literal 与具体 value 的配对；
- hidden-cosine top-k 但 relation provenance 不同；
- unassigned/condition residue 附近的 raw-token decoys。

若一个 batch 里没有 hard negatives，只能训练 representation warmup，不能用于宣称 owner disambiguation。

## NULL / deletion 训练

可以训练 `unmatched_under_sourceview`，但必须明确它不是 N。

构造方式：对同一个 masked query，从 candidate view 中删除所有正例和经 source graph 证明的重复正例，同时保留同源同类型 hard negatives、raw tokens、相同 candidate count bucket、NULL/UNKNOWN。模型应把 NULL 排在前面。

防止删除痕迹：

- query view 不包含“删除发生”的任何字段；
- candidate pool 大小、source unit 覆盖、role type 分布与正例任务匹配；
- 不能让 NULL 只出现在候选更少、没有同类型竞争项、或 graph coverage 更低的样本中；
- 删除 source constraint 的训练标签只能叫 `sourceview_unmatched`，不能进入 not-stated、missing fact 或 unsupported 统计。

## 最小模型与 loss

不建议上 GNN 主线。首版用 role-aware bilinear + 小 pairwise scorer 足够，便于审计和对照。

### 特征

对 query `q_i`：

- masked role hidden pool：冻结同一 observer、固定 layer set，canonical query serialization 上的 `<MASK_ROLE>` / event anchor pooling；
- owner context pools：subject/entity、predicate/action、stage/time、negation、condition；
- role type、value type、field-path tokens；
- condition residue flags。

对 candidate `c_j`：

- source value/mention hidden pool，同 observer、同 layer set；
- candidate event/record context pool；
- role/field type、value type、normalized scalar features；
- provenance graph features：same record, field sibling, event membership, condition attachment。

不得使用 hallucination labels、reader E/C/N、B/A F、native Δ、RAGTruth span label 或 synthetic binary hallucination label。

### Scoring

```text
s(q, c) = q^T W_role c
        + u^T phi_type(q, c)
        + v^T phi_provenance(q, c)
        + b_role
```

可选 1 层或 2 层 event-local message passing，只在 source candidate graph 和 query context graph 内传播，不跨 response labels，不读 native effects。若 bilinear 与 hard negatives 已足够，不加 message passing。

### Loss

多正例 InfoNCE：

```text
L_pos(q) = -log  Σ_{c in P(q)} exp s(q,c) / Σ_{c in C(q) ∪ {NULL,UNKNOWN}} exp s(q,c)
```

NULL 删除任务：

```text
L_null(q) = -log exp s(q,NULL) / Σ_{c in C_deleted(q) ∪ {NULL,UNKNOWN}} exp s(q,c)
```

Hard-negative margin 可作为辅助：

```text
L_margin = mean_{n in H(q)} max(0, m + s(q,n) - logsumexp_{p in P(q)} s(q,p))
```

总损失：

```text
L = L_pos + λ_null L_null + λ_margin L_margin + λ_entropy H_reg
```

`H_reg` 只防止所有分数塌到 NULL 或单一高频 field，不用于校准 truth probability。

## 划分与验收

训练/验证必须按 source 划分，不按 role instance 随机划分。否则同一 source 文档、同一 Data2txt record 或同一 field path 会泄漏。

建议三层 split：

1. `source-heldout`：文档/source_id 完全不重叠，用于 checkpoint selection。
2. `schema-heldout`：Data2txt field path / table schema 部分不重叠，用于检验是否只记 field key。
3. `style-heldout`：自然语言 source 与 Data2txt source 分开报告；不要用 Data2txt 成功覆盖自然文本失败。

自然 RAGTruth/当前 36 条只能做最终诊断或开发集报告，不能选 checkpoint、调 NULL cost 或挑 hard negatives。若已有六个 source 被反复用于修设计，不能叫未见测试。

最小验收指标不是 hallucination F1，而是候选供给：

- source self-supervised top-1/top-5 owner recall；
- same-value wrong-event error rate；
- same-field other-record error rate；
- deletion NULL false positive/false negative；
- surface-ablation 后的 top-k recall；
- position-randomization/canonicalization 后的 recall；
- downstream relation-verified candidate supply 是否高于 rule matcher。

如果只在普通 source reconstruction 上高分，但 same-value/different-event 或 position-randomization 崩溃，应否决进入主线。

## 对当前失败的直接缓解能力

能缓解：

- sourceQA 生成 `source_answer/source_answer_quote=null`：owner head 不生成 quote，只输出已有 node ID。
- source quote 编造或把 `<MISSING_SLOT>` 问题句当 source quote：candidate 由 source graph/raw inventory 反查，不能引用问题文本。
- Qwen 把 `Additionally,` 当必要条件：owner head 不复制逐字 condition table；它只提出候选，condition residue 仍由 finite verifier 处理。
- native0 因没有语义风险目标：可先给 candidate supply 和 blind target-dependence 输入，但不提供 correctness direction。

不能缓解：

- source graph 本身漏抽或 field graph 错；
- response event segmentation 错；
- predicate/condition applicability 需要语义判断；
- conflict 中 source 相关性很高但值/条件错误；
- source V 已混合上下文导致 origin 不能由 candidate ID 直接证明；
- 历史错误传播方向。

这些仍需要 reference_integrity、relation_verification、contrast_validity、native donor/input mediation 和 continuation edge verification。

## 是否会学到 copy / 位置 / 词面捷径

会，除非前述视图和 negatives 写死。最大风险有三个：

1. **copy shortcut**：query 中残留 target surface 或 alias，模型学字符串匹配。必须 mask aliases、数值单位、range endpoints、bool polarity aliases。
2. **position shortcut**：source self-supervised query 和 candidate 来自同一原文位置，hidden state 包含位置。必须用 canonical local serialization，禁用绝对 char/token position，并做 position shuffle 测试。
3. **field shortcut**：Data2txt 中 field key 与 answer type 强相关。必须做 schema/field-path heldout，并保留 same-field other-record hard negatives。

如果这些测试不过，该头没有可识别的域迁移基础，应否决。不能通过加 GNN 或加更多字段补救。

## 推理接口

推理时对每个 response event role：

```json
{
  "owner_head_version": "source_self_supervised_owner@1",
  "query_event_id": "...",
  "query_role_id": "...",
  "query_view_hash": "...",
  "source_graph_hash": "...",
  "feature_manifest_hash": "...",
  "candidate_rankings": [
    {
      "source_node_id": "...|NULL|UNKNOWN",
      "source_event_or_record_id": "...|null",
      "score": 0.0,
      "rank": 1,
      "basis": ["hidden", "role", "field_path", "event_context"],
      "same_value_competition": false,
      "same_field_competition": false
    }
  ],
  "status": "candidate_supplied|null_under_sourceview|unknown_coverage|ambiguous",
  "downstream_allowed": "relation_verification_only"
}
```

它只能 feed `RelationVerifier(frozen IDs)`。Verifier 通过后才能 compile contrast；contrast 通过后才能 native correctness certificate。若只进入 blind target-dependence，输出仍是 dependence，不是 support/conflict/N。

## 采纳条件与否决条件

采纳条件：

- rule matcher 或 sourceQA candidate supply 在 v3/v4 后仍是主瓶颈；
- owner head 在 source-heldout 和 schema-heldout 上通过 hard-negative 指标；
- top-k 候选显著增加 relation verifier 的可用输入；
- 不降低或污染 native auditor 的分母和 scope。

否决条件：

- 需要 RAGTruth 或 synthetic hallucination binary labels 才能工作；
- 主要收益来自 target surface 泄漏、source 绝对位置、field key 记忆；
- NULL score 被用作 not-stated；
- relation verifier 仍同等卡死且 owner head 只增加未验证候选；
- 输出被改名成 hallucination classifier、adoption error 或 route-derived correctness。

## 最终建议

不要立即把它作为主方法替换现有 matcher/native。把它作为 **candidate supply fallback** 放在非训练 rule matcher 之后：如果 rule/event-local matcher 仍无法给 Qwen verifier 提供稳定候选，再训练这个单头 owner pointer。它是旧三头方案中唯一可能保留的可识别训练目标，因为监督来自 source 自身坐标；但它只学 source ownership retrieval，不学事实正确性、不学模型采用、不学连续错误。

最小版本不需要 GNN：role-aware bilinear + event-local pairwise features + hard negatives 足够先判断是否有价值。只有在 source-heldout/schema-heldout 上证明 bilinear 对关系组合不够，才考虑 1-2 层 message passing。若这一步仍失败，应回到 schema-first deterministic contrast bank / alignment-only dependence，而不是继续堆模块。

