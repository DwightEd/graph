# surface owner architecture follow-up — 2026-09-13

范围：`research-refine` 有界方法复核。只读 `next_iteration/surface_graph.py`、`next_iteration/surface_owner.py`、`tests/test_surface_graph.py`、当前 bridge prepare drafts 与前序设计文档；未修改冻结 soft_graph/bridge 文件，未启动 GPU，未运行新实验。

## 结论

这版方向是对的：它实质删除了 Qwen 自由 token-index SRL 作为 strict bridge 的硬前置，并且把 wrong-value 牵引 owner 的主路径切断了。`surface_graph` 从完整 sentence envelope 内程序枚举 surface slots，`semantic_role_extraction_required=False`；`surface_owner` 对 response/source 都用 masked context 建 document，`target_surface_used_in_ranking=False`，owner cost 不读原错误值或目标值 hidden state。content_run 也明确只是保守 lexical proposal，不是假 noun/SRL role。

但这还不能直接作为下一轮 GPU strict bridge。GPU 前需要补两个最小接口约束；都不需要新网络或幻觉标签训练。

## 已解决的关键问题

1. **SRL 硬依赖已移除。** 旧 `source_pointer_contrast.py` 从 `claim["source_terms"]` 的 Qwen target role 取 span/role，导致坐标合法但语义错的替换。新 `surface_graph.py` 不依赖 pointer/SRL，完整句跨 raw address boundary 保留，`doesn't`、`on`、编号 marker、punctuation-only、quote-only 等已被测试约束。

2. **wrong-value owner collapse 已显著缓解。** `match_owners` 只比较 slot 的 masked base context、source occurrence 的 masked parent context、path lexical；target quote 不进入 ranking。source 端的 literal field 也把 value 从 masked context 中移除，并保留 record parent。这正是避免 `4.0` 自动拉向任意 review star、`on` 拉向介词、review text 被 surface overlap 牵引的必要改动。

3. **strict/native 边界仍清楚。** `edit_candidate` 只产出 `candidate_not_semantically_verified`，`semantic_edit_kind="unknown_requires_finite_subject_predicate_condition_gate"`，不把 surface match 当支持/冲突，不读取标签，不发 certificate。辅助 masked 3-layer encoding 也标明不是原始生成轨迹；原生定位仍应使用完整 B/A F 的 `CausalOracle`。

## Required before next GPU strict bridge

### R1. finite edit-type gate 必须成为显式 frozen artifact，而不是只存在于 `semantic_edit_kind` 字符串里

当前实现已经把 subject 硬禁改成“需要 finite gate”，这是正确方向；但在接口层还需要写死这一个 gate 的输入、标签、阈值和产物，否则 content_run 仍可能把 subject、predicate、condition 或 owner-context 改写伪装成 value edit。

最小 gate：

```text
EDIT_TYPE_GATE(original_base, edited_base, slot_span, source_occurrence, parent_context)
labels:
  V = value_only: only the selected surface value changed; subject, predicate/action, condition, owner, unit/scope, tense, coreference, and grammar are preserved
  X = subject_predicate_condition_or_owner_change: edit changes or corrupts any non-target factual role/scope, or requires a rewrite to be grammatical
  U = unknown
pass: P(V) >= 0.8
```

This gate can replace or precede the old preservation prompt, but it must be recorded as a separate frozen decision because it is now the only protection after deleting SRL role labels. Strict contrast/native may proceed only when this gate passes plus the existing finite relation checks pass. If it fails, output `not_value_only_edit` / `edit_type_unknown`; do not try another slot or widen the edit in the same artifact.

This is the one necessary new gate. It is not a new model component; it is a finite label interface over an already frozen edit.

### R2. deterministic surface typing needs two small coverage fixes for the actual Data2txt failure modes

The current exact type gate is good, but the surface type rules are still too brittle for known natural/Data2txt sources:

- `8:0-16:0` from source hours will not match `_CLOCK` because the minute regex requires two digits. It falls back to content_run, while response `8 AM to 6 PM` is `time_range`, so the intended hours candidate will be blocked before finite validation.
- names with internal connectors such as `Crushcakes & Cafe` are likely split into `Crushcakes` and `Cafe` response slots because `&` counts as a separator. That weakens owner matching and can reintroduce subspan edits.

Concrete fix without new machinery:

- extend the deterministic time/range recognizer to accept `H:M` and `HH:M` forms as source time ranges, without converting them to natural language;
- allow content runs to include a small whitelist of internal name connectors (`&`, apostrophe, hyphen already partly covered, possibly `.` inside abbreviations) when both sides are content tokens, so `Crushcakes & Cafe` is one surface proposal;
- keep these as `surface_type`/span proposal rules only, not semantic role claims.

If this is not fixed, the new matcher can still be architecturally cleaner but may fail the exact examples that motivated the revision, moving failure from SRL rejection to type-mismatch/subspan rejection.

## Important non-blocking constraints

1. **Same-value ambiguity should preserve IDs where feasible.** `match_owners` records `same_source_value_occurrences` as a count and near-tie IDs. For final auditability, selected candidates should also retain the parent IDs of same-surface/value competitors when cheap to enumerate. The count is enough for denominator, but IDs make C-stage ambiguity review and error analysis much clearer.

2. **Lexical-only mode must stay preflight.** The code labels `lexical_only_preflight` clearly. Do not use it as the production owner matcher unless dense masked features fail closed; otherwise wrong-owner retrieval will become mostly field/path lexical matching.

3. **content_run is not a noun phrase.** The report/output should repeat this in manifests. A content_run candidate can become strict only through the finite edit-type gate and selected-source verifier; it is not an entity, object, or subject by construction.

4. **Natural text multi-hop/compositional source support remains outside scope.** Surface occurrence matching can find a candidate sentence/span, but it cannot prove paraphrase, arithmetic, range containment, or multi-sentence synthesis. These should remain `requires_relation_verification` or unresolved, not be solved by lowering the type gate.

## Minimal downstream interface

A strict candidate should move forward only with this frozen packet:

```json
{
  "slot_id": "...",
  "base_id": "...",
  "slot_surface_type": "number|time_range|duration|date|content_run|explicit_quote",
  "source_occurrence_id": "...",
  "source_parent_id": "...",
  "owner_match": {
    "cost": 0.0,
    "features": {
      "masked_context_cosine": 0.0,
      "masked_context_lexical": 0.0,
      "path_lexical": 0.0
    },
    "owner_ambiguous": false,
    "near_tie_occurrence_ids": []
  },
  "edit_candidate_sha256": "...",
  "edit_type_gate": {"V": 0.0, "X": 0.0, "U": 1.0},
  "source_occurrence_relation": "S|C|I|U",
  "original_global": "S|C|N|U",
  "edited_global": "S|C|N|U",
  "strict_contrast_status": "available|not_value_only_edit|owner_ambiguous|not_supported|unknown"
}
```

Only `strict_contrast_status="available"` can enter the old B/A `CausalOracle`. Everything else may still contribute to soft candidate/dependence reporting but not to route-derived hallucination, adoption, pairedR, or MLP mechanism claims.

## Early-stop note

Stopping soft v1 before expensive D is methodologically justified. A36/B36/C27 already showed the front-end structure did not satisfy the precondition for meaningful target-only D: fixed address/incomplete SRL boundaries, bag risk spreading, and 1704 B edges producing only 10 mostly malformed surface edits. Preserving partial artifacts and marking `design_invalid_early_stop` is cleaner than spending GPU on a native stage whose output would not answer the user’s key correctness/window/route-vs-aggregation questions.

## Final recommendation

Proceed with this SRL-free surface owner direction after R1 and R2 are fixed. Do not add a GNN, learned head, or hallucination-label training. The next run’s first success metric should be candidate-supply and C-stage reason distribution: base units, surface slots, owner top-k pairs, finite value-only pass, edited-S/original-CN pass, and strict B/A availability. Native certificates should remain a downstream subset, not the criterion for whether the new surface front-end works.
