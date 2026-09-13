# Event graph → matcher → verifier → contrast → native 端到端失败审查

日期：2026-09-13  
模式：`research-refine` 有界方法规格审查；真实 collaboration GPT-5.5 fallback，Codex MCP 不可用。范围：推演 v3 后条件方案，检查 source-only pointer/field graph、冻结高维 features、role top-k + event provenance beam、冻结 Qwen 有限选项关系核验、精确 single/joint-event 对照、现有 native 的端到端接法。未改代码、未启 GPU、未改变正在运行的 v3。

## Anchor check

`FINAL_PROPOSAL` 的问题锚点仍是：在不以幻觉标签训练的条件下，测量冻结 LLM 中来源和历史信息的实际传递、聚合与输出采纳；区分错误沿用、正确纠正和新陈述；事实区间和影响范围分开评价。

用户三个关键缺口仍是：

1. 自动回看位置；
2. route-derived hallucination / continuous spans；
3. 条件归属与错误传播。

因此下一版不能靠“更容易跑通的格式实验”替代核心问题。matcher 可以增加候选供给，但不能把匹配质量、field-path 合法性或 blind logP dependence 改名成事实采纳、路由错误或正误检测。

## 总体判定

端到端管线有必要，但只有在两个条件下才值得作为 v4：

- matcher 负责**候选供给**，不是事实判别；
- contrast 构造规则先定义清楚，避免 “整事件改完后自证 same-event”。

当前最小可实现设计应分成两条输出：

1. `target_dependence`：对真实 claim 的原始完整 logP 做盲 native 来源/历史组搜索，给出哪些消息影响生成该 claim。它可以低弃权、覆盖全部 claim，但只回答“依赖哪里”，不回答“该依赖是否正确采用”。
2. `correctness_certificate`：只有 relation verification + exact contrast 通过的子集，才能用 A/B F 发 single/joint/event-level 方向性证书。

如果只做第 1 条并把后置 relation 标签贴上去，方法会退化成“语义 verifier 判真，native 外包解释”。这不是用户要求的完整正误决策，也不是 route-derived hallucination detection。它可作为候选和覆盖报告，不能作为主成功条件。

## 端到端最小接口

建议固定以下合同，避免各阶段互相回填：

```text
SourceInventory / SourcePointerGraph
  -> FeaturePack(same observer, same layer set, no labels/native effect)
  -> EventLocalMatcher(candidate-only)
  -> RelationVerifier(finite choices over frozen IDs)
  -> ContrastCompiler(exact edit/withholding or unresolved)
  -> NativeAudit(F_text target-dependence and/or A/B correctness certificate)
  -> MergeReport(scope-separated denominators)
```

每个阶段只读前一阶段 frozen artifact。Relation verifier 发现新来源 span 不能回填同一 matcher；只能产生新版本并重算分母。Native 不能反向改变 relation status 或 candidate pool。

## Data2txt scalar 的对照构造规则

Data2txt 来源是 Python literal / 字典 repr，不是自然谓词句。field path 和 scalar value 可以提供强 provenance，但不能自动证明自然语言 predicate 正确。对照构造要按 scalar 类型分支。

### 1. 字符串或带引号标量

- Source graph 保存 `raw_literal_surface` 和 `display_value`。引号是 provenance 的一部分，但 response edit 使用 `display_value`。
- 只有 relation verifier 判断 response predicate/role 与该 field path applicable，且 identity basis 包含 record ID / entity field / stable condition，才允许 `scalar_slot_replacement`。
- A 的 edit script 只替换 response target span；source value 若在同 record 多处出现，必须保留 same-value ambiguity 和 controls。

### 2. 数字、单位、hours-range

- 先程序标准化为 typed value：`number`, `unit`, `interval`, `range_closed/open`, `source_unit_from_field_path`。
- 若 source 是 range，response 是点值：
  - 点值在 range 外：可进入 conflict candidate。
  - 点值在 range 内但比来源更具体：不是普通 conflict，应标 `overspecified_within_range`；可构造 `specific_commitment_vs_range`，但不能称 source 给出该精确点值。
  - range 部分重叠：`partial_overlap_unresolved`，不进正误证书。
- `hours`、`duration_hours`、`opening_hours` 等单位来自 field path 只是 typed rendering input，不是 predicate truth；仍需 relation verification。

### 3. `None` / null literal

- `None` 表示结构字段未知、未提供或空值，不能编译成 `not_stated`，也不能作为“来源说没有”。
- 若 response 对该 field 作具体承诺，最多进入 `field_null_commitment_vs_withholding`：B=具体承诺，A=固定非承诺模板或字段未知模板。该分支不能和 single-role N 准确率合并。
- 若 field path 本身不 applicable，直接 unresolved。

### 4. bool

- Bool 只在 relation verifier 明确 predicate-polarity mapping 后使用。
- `true/false` 不能自动自然化成肯定/否定谓词。需要 `boolean_polarity_contrast`，记录 predicate phrase、negation span、field path、render template。
- 若 response 没有可定位 polarity span，不能用整句自由改写生成 A。

### 5. 句中未抽出条件

- Response graph 的 `unassigned_spans` 若含数字、时间、否定、比较级、阶段词、介词短语或括号说明，contrast compiler 不能忽略。
- 不要求全弃权：把这些 span 作为 `condition_residue` 传给 finite verifier，让 verifier 在固定选项中判断 `condition_preserved|condition_extra_unsupported|nonsemantic_residue|condition_unresolved`。
- 只有 `condition_preserved` 或 `nonsemantic_residue` 才能继续；`condition_extra_unsupported` 可进入 event-level unsupported，但不能发 slot-level certificate；`condition_unresolved` 只发 target-dependence。

## 多角色同时错误时的最小 joint 规则

Joint contrast 的危险是：把 response 的多个角色替换成 source 值后，A 自然会像来源，但这不能证明原 B 和 source 是同一事件。必须先有未编辑的 identity basis。

允许 joint 的最低条件：

1. 所有替换角色来自同一个 source event/record/hyperedge；不能拼接不同 source events。
2. 至少一个 identity basis 未被编辑，且该 basis 不是纯 surface 同值：可为 record ID、entity field、predicate/action family、stage/time condition、明确 coreference link。
3. identity basis 与非编辑条件通过 relation verifier；若全部身份角色都在 editable_roles 里，降为 `same_event_family_only` 或 `identity_unresolved`。
4. A edit 是 deterministic：替换指定 spans，保留其它字符；若需要自由改写谓词/语序才能成句，不能进入 native correctness certificate。
5. 输出 scope 是 `joint_event`，不能拆成单个数字/实体的 route attribution。

同义改写没有字面未编辑 identity 时，不应直接弃权。可以用冻结 hidden/field-path/coreference 提出 identity candidate，再让 finite verifier 判 `same_event_with_editable_roles`。但 verifier 必须列出 identity basis IDs；没有 IDs 的“看起来同义”不能过关。

## 原文不同事件同值的防假阳性规则

同值是 Data2txt 和自然文本里最容易制造假适用性的来源。

Required：

- 每个 source scalar occurrence、list element、field value、自然文本 mention 都保留不同 ID。
- matcher 输出 same-value competing candidates；relation verifier 必须知道它们存在。
- 如果 top-1 和 top-2 只差 event/record 而 value 相同，输出 `same_value_event_ambiguous`，不能构造 correctness certificate。
- native controls 至少包含 `same_value_wrong_event` 或 `same_field_other_record`，若存在但未测，证书降级。

这条规则比继续加更多 Qwen 字段更重要，因为它直接防止“值对了但事件错”的假成功。

## Blind F_text / 原始 claim logP 简化的评估

这个简化应**部分采纳**，但要严格限名。

定义：对每个真实 claim/event B，计算完整原始 claim 序列的 `log P_theta(B | source, prefix)`，不做长度均值；对 source/history 消息组做盲搜索，得到 `Δ_text(G)=logP_base(B)-logP_G(B)` 或等价符号。候选组确定后，再由 relation verifier 给出该组文本与当前 claim 的 applicable/support/conflict 状态。若后续能构造 A/B，再发方向性 certificate；不能构造 A/B 时只发 `target_dependence`。

优点：

- 不要求先构造精确 A/B，能避免 native 因 semantic contrast starvation 变成 0 调用。
- 可覆盖 supported、unsupported、ambiguous、multi-role 和 derived cases 的“生成依赖位置”。
- 有助于用户的自动回看需求，尤其能暴露 source/history 哪些位置推动了当前 claim 文本。

不能升级的原因：

- `logP(B)` 只测模型生成 B 的依赖，不测 B 相对正确 A 的偏好。
- 如果 relation verifier 后置判 source conflict，然后 native 发现 source group 推动 B，事实错误方向仍来自 verifier；native 只是解释依赖。
- source V 可能混合上下文，删除 source group 降低 B 可能来自实体、格式、主题或共现，而非错误证据采纳。
- 对 supported claim，强 dependence 是正常；对 unsupported claim，强 history dependence 不等于错误传播，除非 previous slot/event、错误方向和 origin mediation 都过关。

结论：`blind_target_dependence_audit` 是可采纳的 Stage 0/auxiliary output，不能作为完整检测方法。报告语言应是：

```text
This claim has a finite native dependence on source/history group G.
Relation verifier labels G as applicable/support/conflict/unresolved.
No correctness-direction certificate is issued unless a valid A/B contrast exists.
```

如果为了 coverage 把它叫 `adoption_error`、`route_hallucination` 或 `correctness decision`，就是改名完成目标，必须否决。

## 如果 matcher + finite verifier 仍被 sourceQA 同等卡死

若 v4 仍主要卡在 relation verifier，而不是 candidate supply，那么不要继续加 Qwen 表格字段。更小替代结构是按数据类型拆两条：

### A. Data2txt schema-first contrast bank

对结构化来源，先不用 Qwen 判断 source answer。由 AST literal graph 生成 field/value contrast bank：

```json
{
  "field_id": "...",
  "record_id": "...",
  "value_id": "...",
  "typed_value": "...",
  "render_candidates": ["canonical value", "value with unit", "range rendering", "withholding rendering"],
  "applicability_features": ["field_path_tokens", "record/entity match", "value_type", "unit"]
}
```

matcher 将 response role 与 field/value candidates 对齐；若 finite relation verifier 无法确认 predicate applicability，仍可运行 `target_dependence`，但 correctness certificate 保持 unresolved。对于字段名语义非常明确的开发子集，可预注册一小组 schema rules，例如 `duration_hours -> duration`、`is_* -> boolean attribute`；这些 rules 是 dataset schema assumptions，不能泛化到自然文本，也不能叫 reader-free semantic truth。

### B. Natural text alignment-only fallback

自然语言来源没有 schema truth。若 finite verifier 失败，只能用 frozen-feature matcher + blind native 给 `candidate_evidence_dependence`，并把 correctness 留给人工/RAGTruth evaluation 或后续 reader。不要用最高相似 event、attention OT 或 field-like heuristic 自行判 support/conflict。

这个替代结构仍是完整可运行系统，但它的输出分成：全量 dependency map + 子集 correctness certificates。它不能独立完成全量正误检测；若实验只得到前者，就必须报告主科学目标未完成。

## 面向用户三缺口的最小成功条件

1. 自动回看位置：`blind_target_dependence` 可提供全量候选回看；若有 matched controls 和 origin mediation，可升级为 finite position/origin witness。
2. route-derived hallucination / continuous spans：只有 A/B 或 relation-verified unsupported branch 通过时才可称 hallucination-related route span；否则只是 generation-dependence span。
3. 条件归属与传播：必须有 condition_residue verdict、identity_basis、previous slot/event link、错误/纠正方向和 origin mediation。matcher proximity 或 history attention 不够。

## 必须报告的分母

每批结果至少分开：

- `claims_total`、`events_total`、`roles_total`；
- `matcher_candidate_supplied`、`matcher_null_top1`、`matcher_unknown`、`same_value_ambiguous`、`identity_unresolved`；
- `relation_verified_applicable`、`relation_failed_condition`、`relation_failed_field_path`、`condition_residue_unresolved`；
- `contrast_single_valid`、`contrast_joint_valid`、`contrast_withholding_valid`、`contrast_unavailable`；
- `blind_target_dependence_executed`、`target_dependence_only`；
- `native_correctness_certificate`、`origin_mediated`、`continuation_certified`、`unresolved`。

不能只报 native 证书子集，也不能把 target-dependence only 从失败分母拿掉。

## 最终 Required

若要避免 v4 变成又一轮格式修补，进入下一版前需要写死以下规则：

1. matcher 只供给候选，不判 support/conflict/N；Qwen verifier 只读 frozen IDs，不生成新 source answer。
2. Data2txt scalar 按 string/number-range/None/bool 分支构造 deterministic contrast；field path 只作 provenance compatibility。
3. joint_event 必须有未编辑 identity_basis；无 basis 时最多 event-family dependence，不发 correctness certificate。
4. condition_residue 必须显式进入 relation verifier；未抽条件不能 silent drop。
5. same-value different-event ambiguity 必须保留候选和 controls。
6. `blind_target_dependence_audit` 可先跑以避免 native0，但只输出 dependence，不输出正误方向。
7. A/B correctness certificate 仍是主 claim 的必要子集；没有 A/B 的结果不能改名成 adoption error。
8. 如果 finite verifier 仍卡死，Data2txt 走 schema-first contrast bank，自然文本走 alignment-only dependence；二者都不得冒称完整事实检测成功。

这个规格能让下一版成为完整可运行的方法：全 claim 有候选/依赖分母，relation+contrast 子集有方向性机制证书。它不会把更多 prompt 字段当修复，也不会用覆盖率换掉用户要的准确回看、归属和传播。
