# surface-slot owner matcher revision review — 2026-09-13

范围：有界方法设计复核。未修改已冻结的 `soft_graph_v1` 47 个代码文件，也未改当前 bridge 4 个 prospective 文件；未启动 GPU。依据是已完成的 source-pointer bridge prepare 分母与 10 个 candidate draft 抽查、当前 `source_pointer_contrast.py` 的真实依赖、`graph_boundaries.py` 的 base/slot/region 三层设计，以及此前分段/匹配文献笔记。

## 判定

应删除 Qwen 自由 token-index SRL 作为 strict bridge 的硬前置。保留它只能作为可选 advisory anchor，且不能决定 target span、role 类型或 claim 边界。

这不是“再修 prompt”。当前 1704 条 B edges 里只有 10 个 candidate，且候选文本出现 `she doesnon misses...`、`the cafe 4.0...`、整段 review 插入句中、`Wintergreen essential` 被改成 `essential` 等。根因不是 JSON 坐标完整性，而是 Qwen 给出的 role/span 虽然坐标合法，却没有可靠的语义槽位边界与 owner 绑定。继续围绕这些 role 做拒绝规则会把大多数自然样本留在 `role_requires_whole_event_rewrite` 或产生不可读 A。

下一版最小可行结构是：

```text
complete BaseUnit(sentence/safe clause)
  -> deterministic SurfaceSlot proposals
  -> SourceOccurrence + ParentContext graph
  -> masked-target owner/occurrence matcher
  -> finite local verifier over fixed candidates
  -> strict single-slot contrast / soft dependence output
```

这里的 surface slot 是候选可编辑/可核验片段，不是 subject/predicate/object 语义角色。owner/event/record 归属由 source parent context + masked target context + finite verifier 条件化判断，不由 Qwen SRL 直接给出。

## 为什么 mask target 有必要

如果匹配器直接用 response 目标值的文本或 hidden state，错误值会牵引到“同错误值/相似值”的错误来源 owner。例如 `4.0` 可能拉到 review star 而不是 business star，`on` 可能来自任意介词，整段 quoted review 可能因 surface 完全相同被插进非 review predicate。目标值越错误，越会把 owner 也带偏。

因此首要 query 应是 masked target context：把待核验 surface span 从 base 里删去或替换为固定类型占位，只用非目标上下文、base owner cues、邻近 predicate words、field/path lexical cues、高维上下文来找 source occurrence。目标 surface 只用于类型检查、原/改文本构造和 finite verifier，不用于 owner retrieval 的主分数。

这不能完全消除 wrong-owner：如果非目标上下文本身错误或缺失 owner，matcher 仍可能错。它的作用是避免错误 value 成为 owner 选择的主因；最后仍必须由 finite local verifier 核验 selected source occurrence 的 owner/context。

## 最小接口

### 1. BaseUnit：完整可读上下文

直接复用下一版 boundary 层思想。每个 response word 属于一个完整 sentence/safe-clause base。raw address unit 和 Qwen anchor endpoint 不参与 base 边界。

```json
{
  "base_id": "sha256...",
  "span": [start, end],
  "text": "exact sentence or safe clause",
  "kind": "sentence_envelope|safe_clause",
  "address_unit_ids": ["u0", "u1"],
  "indivisible_for_semantic_scoring": true,
  "scope": "readable_context_not_single_fact"
}
```

Strict bridge 后续可以只编辑 base 内的 surface slot；soft detector 可以给 base bag risk；词级输出仍只能把 localized risk 赋给具体 slot/fact span。

### 2. SurfaceSlot：程序枚举，不假装 SRL

在每个 BaseUnit 内用确定性规则枚举候选 span。首版只覆盖高价值、可替换、可核验的 surface 类型：

- `number`: 整数、小数、百分比、货币、带逗号数字；
- `date_time`: 日期、年份、时间、时间范围、年龄/持续时间表达，如 `1,350-year-old`、`8 AM to 6 PM`；
- `explicit_quote`: 引号内完整内容，作为核验候选；strict edit 是否允许另由 surface policy 决定；
- `proper_noun`: 连续大写/标题式名称、带内部 `&`/`-`/`.` 的实体名；
- `common_noun_chunk`: 保守 noun-ish chunk，长度 1-6 content tokens，停在介词/动词/标点，不跨 base。

每个 slot 只保存类型与坐标，不保存 subject/predicate/object 判断：

```json
{
  "slot_id": "...",
  "base_id": "...",
  "span": [a, b],
  "quote": "8 AM to 6 PM",
  "surface_type": "date_time",
  "proposal_reason": "regex_time_range",
  "edit_policy": "single_slot_allowed|quoted_edit_closed|needs_verification|not_editable",
  "mask_text": "The business hours are from <TIME_RANGE> every day, and they offer free WiFi.",
  "non_target_context_span": [[base_start, a], [b, base_end]],
  "not_semantic_role": true
}
```

Program rules should aggressively reject punctuation-only, quote-only delimiter, pure stopword, coordinator, preposition, determiner, and spans that begin/end with open function words. This alone blocks examples like replacing a clause with `on` or leaving `doesn` + `on` glued together.

### 3. SourceOccurrence + ParentContext

Source side should have two schemas but one matcher interface.

For Data2txt/literal sources:

```json
{
  "occurrence_id": "...",
  "source_kind": "literal_field",
  "span": [a, b],
  "raw_surface": "4.0",
  "display_surface": "4.0",
  "value_type": "float",
  "field_path": ["business_stars"],
  "record_id": "...",
  "parent_id": "record_or_field_parent",
  "parent_context": {"field_path_tokens": ["business", "stars"], "sibling_fields": [...], "record_keys": [...]},
  "epistemic_status": "observed_literal|unknown_null"
}
```

For natural source text:

```json
{
  "occurrence_id": "...",
  "source_kind": "source_surface",
  "span": [a, b],
  "surface": "Wintergreen essential oil",
  "surface_type": "proper_noun|number|date_time|explicit_quote|common_noun_chunk",
  "parent_id": "source_sentence_or_clause_id",
  "parent_span": [s, e],
  "parent_text": "complete source sentence/clause",
  "owner_context": "parent with occurrence masked"
}
```

Raw source tokens can remain coverage fallbacks, but they should not feed strict bridge unless promoted to a `source_surface` occurrence with parent context. A lone raw token like `on` is not a valid source answer candidate for strict A/B.

### 4. Masked-target owner/occurrence matcher

The matcher takes one response `SurfaceSlot` and all source occurrences with compatible surface type. It outputs candidate pairs, not truth labels.

Minimum feature inputs:

- response masked-base representation: frozen observer features or frozen text encoder features for base text with slot replaced by a type placeholder;
- source occurrence representation: value/surface span feature;
- source parent representation: source sentence/field-record context with occurrence masked;
- lexical features from non-target base context vs source parent/field path;
- type compatibility: `surface_type`, value_type, unit/range flags;
- record/event membership and same-value competition counts;
- optional native/soft signal only as separate downstream evidence, not as candidate truth.

A fixed score is enough; no hallucination-label training is needed:

```text
cost(slot, occ) =
  0.45 * d(masked_response_base, masked_source_parent)
+ 0.20 * d(response_non_target_context, source_parent_context)
+ 0.15 * type_or_value_type_penalty
+ 0.10 * field_path_or_owner_lexical_penalty
+ 0.10 * same_value_or_parent_ambiguity_penalty
```

The exact weights can be frozen constants for v1; they are not learned truth parameters. For Data2txt, `field_path_or_owner_lexical_penalty` compares base context words to normalized path tokens and sibling key names. For natural sources, it compares base context to the source sentence/clause with the occurrence masked. Do not score support/conflict here.

Top-k contract:

- retain top 4 source occurrences per response slot;
- retain all same-surface/same-value competitors in denominators, even if outside top-k;
- event/record beam may prefer candidates sharing a parent across multiple slots in the same base, but cannot force source one-to-one capacity;
- if several candidates have near-tie owner/context score, mark `owner_ambiguous` and require stronger finite verification before strict contrast.

### 5. Finite local verifier over fixed candidates

After B freezes candidates, C receives only finite-choice questions about fixed IDs/spans. It does not generate an answer and cannot add new source candidates.

For each `(slot, occurrence)` pair:

- `applicability`: same_owner_context / different_owner_or_event / unknown;
- `original_relation`: selected occurrence supports / contradicts / unrelated / unknown with respect to the original slot value in the base;
- `edited_relation`: selected occurrence supports / contradicts / unrelated / unknown with respect to the deterministic edited base;
- `preservation`: pass / fail / unknown for non-target context, grammar, unit, scope, owner;
- `condition_residue`: preserved / extra_unsupported / nonsemantic / unresolved.

Strict bridge may proceed only when applicability is same-owner, original is C or N, edited is S, preservation passes, and condition residue is preserved/nonsemantic. Otherwise the pair remains soft evidence or unresolved.

## Can Qwen SRL be removed as hard precondition?

Yes. The hard dependency should be removed.

What remains from Qwen-like extraction:

- optional pointer anchors nested under base units;
- optional role hints if complete and non-truncated;
- optional finite verifier judgments over fixed IDs.

What must not remain:

- target span decided by a free-form role JSON;
- role type deciding whether a source occurrence is allowed;
- incomplete anchor endpoint deciding event/claim boundary;
- Qwen-proposed subject/predicate/object frame serving as same-event identity.

This reduces the current failure dependency directly. Instead of 1394 whole-event role rejections caused by SRL role labels, the system will enumerate concrete surface spans and let finite verification reject only failed candidate pairs. The likely coverage gain should be reported as candidate supply coverage, not factual accuracy.

## Does masking avoid wrong-value -> wrong-owner collapse?

It helps, but it is not sufficient by itself.

Required anti-collapse rules:

1. The primary owner score cannot use target quote lexical overlap or target-span hidden states.
2. Same-value and same-surface occurrences must all be counted; near-tie candidates set `owner_ambiguous`.
3. For Data2txt, record/field path and sibling context must be part of the parent context; a scalar value alone is never enough.
4. For natural text, source parent sentence/clause must be part of the candidate; a source surface occurrence without parent context is not strict-bridge eligible.
5. Finite C must validate the selected occurrence, not merely full-source support somewhere else.
6. If original non-target context is itself unsupported or wrong-owner, single-slot edit cannot proceed to a correctness certificate; report `multi_slot_or_owner_context_unresolved`.

The mask prevents the observed wrong value from dominating retrieval. The finite owner/context check prevents a semantically wrong parent from becoming a certificate.

## Deterministic edit policy

For strict single-slot A/B:

Allowed first:

- number/date/time/duration/location/entity/attribute surface spans where replacing the exact span with the source display surface preserves spacing and grammar;
- Data2txt numeric/string fields when value type and units are explicit or preserved by unchanged response text;
- natural source surface spans when the selected source parent supports the edited base.

Closed in first implementation:

- subject replacement;
- bool verbalization;
- quoted/code target edits;
- predicate, negation, condition rewrite;
- full review-text insertion into a sentence;
- range-to-point or unit conversion;
- multi-slot repair;
- source `None` as supported replacement.

For `explicit_quote`, keep it as a surface proposal for matching and soft relation evidence, but strict edit should stay closed unless there is a separate quoted-slot surface policy. This avoids repeating the prior “quoted content pasted into grammar frame” failure.

## Relation to native bridge

This revision changes only candidate supply. It does not change the native estimand.

If a pair passes strict finite validation, construct the same B/A complete base event and use the existing `CausalOracle` path from the native bridge spec:

- source keys are the selected source occurrence keys;
- source parent/record ID is retained for controls and ambiguity reporting;
- raw-origin controls must be frozen for the full selected source occurrence before C or reported as subspan/unresolved;
- pairedR and MLP run only under the validated F, not under target-only dependence.

If no strict contrast is available, the system may still emit soft graph target-dependence/local relation output, but it cannot call it route-derived hallucination or error adoption.

## Denominators to report

A next run should not only report “candidate count.” It should report:

- `base_units_total`;
- `surface_slots_total` by type;
- `slots_with_source_topk`;
- `same_value_or_owner_ambiguous`;
- `candidate_pairs_finite_checked`;
- `same_owner_applicable`;
- `edited_support_passed`;
- `preservation_passed`;
- `strict_contrast_available`;
- `native_position_executed`;
- `origin_mediated_executed`;
- `target_dependence_only`;
- `unresolved_multi_slot_or_owner_context`.

This prevents the method from hiding whether the new surface enumerator truly fixed candidate supply or merely moved failures into C.

## Minimum CPU tests before implementation is connected

1. A complete sentence crossing a raw address boundary still yields one BaseUnit, and surface slots are enumerated inside it.
2. `doesn't miss her co-hosts` does not produce a candidate edit that replaces a large clause with `on`.
3. `Wintergreen essential oil` is enumerated as one noun/proper-noun surface; replacing only `Wintergreen essential` is rejected because the slot boundary is not a complete surface span.
4. `business hours are from 8 AM to 6 PM` enumerates one time-range slot, not separate `8`, `AM`, `6`, `PM` slots for strict edit.
5. A review text field may be a source occurrence, but cannot be inserted into a non-review predicate sentence unless finite preservation passes; by default long quoted/review text insertion is rejected.
6. Same value in multiple records keeps all competitors and marks owner ambiguity unless the selected parent is validated.
7. Target masking changes owner retrieval: candidate ranking cannot depend on exact target quote lexical overlap.
8. Qwen pointer/SRL failure or truncated anchor does not prevent surface-slot enumeration.
9. A passing local verifier can identify a slot-level localized span without assigning bag risk to the whole base.
10. `None`, bool, subject, quoted/code, predicate, negation, and condition spans produce explicit rejection statuses for strict bridge.

## Final recommendation

Implement the surface-slot owner matcher before another strict bridge run. The smallest useful change is not a new GNN or learned detector; it is removing the unreliable Qwen SRL hard dependency and replacing it with deterministic surface proposals plus masked-context source owner matching. This preserves the no hallucination-label-training constraint, improves natural candidate supply, and keeps factual support/contrast claims behind finite validation and independent evaluation.

The expected first success criterion is not native certificate count alone. It is whether the run moves from “1704 edges -> 10 mostly malformed candidates” to a high surface-slot candidate supply rate with honest C-stage rejection reasons. Only the subset that passes edited-S/original-CN/preservation can proceed to the old strict native mechanism claims.
