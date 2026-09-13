# Cloze v3 后续结构审查：single-role mask 是否足够

日期：2026-09-13
审查模式：真实 collaboration GPT-5.5 fallback；Codex MCP 不可用，未冒充外部后端。本文件是 `research-refine` 方法审查，不是 v3 实证结论，也不是代码审计通过书。

范围：阅读了当前 `route_graph/cloze_anchor.py`、`route_graph/cloze_prompts.py`、`route_graph/evidence_anchor.py`、`route_graph/frozen_reader.py`、`docs/NATIVE_METHOD_MODEL_20260913.md`、`refine-logs/structured_alignment_followup_20260913.md`，并使用 v2 full36 的 A/cache 失败轮廓作为诊断背景。未改实现、未启动 GPU、未挑选标签有利样本。

## 结论

v3 cloze 是必要的接口修复，但不是完整结构修复。它解决 v2 暴露的主要表层问题：结构化 JSON 问题不能让 reader 稳定恢复当前 claim 的短槽位，导致 119 题中 93 invalid、26 uncertain、全部 abstain、native0。当前 cloze 设计已经把目标槽位落回原 claim 表面文本，并把条件表变成 sidecar，这是对 v2 失败点的最小修复。

但 single-role mask + “其余全部条件必须支持”有一个不可由提示词消除的识别边界：当自然 hallucination 是整事件错配、多角色同时错误、谓词/主体/条件一起漂移、或来源只支持一个抽象改写时，任意单一目标槽位都会把其它错误角色放进条件表。此时 source reader 必然不能支持这些非目标条件，C/E/N 都必须降为 U。这个保守拒绝是正确的，不能放宽成最近事件匹配；但若这种情况占比高，方法就没有进入用户核心问题所需的自动回看/归属/传播测量。

因此 v3 full36 的主要判据不应是“格式是否好看”，而应是失败质量分解：cloze 是否把 v2 的 self mismatch 降下来；剩余失败是否主要来自多角色事件级错配。如果后者成立，下一步最小结构改动应是事件级 source alignment / joint-mask fallback，而不是继续微调 cloze prompt。

## 当前 v3 接口的已闭合点与残余风险

已闭合或基本正确的点：

- `cloze_anchor.py` 生成 `masked_claim`、`target_role`、`role_bindings`、`condition_roles`、`condition_bindings`、`hidden_aliases` 和 sidecar hash，source-visible payload 不含 unmasked claim 或 answer_quote。
- `evidence_anchor.py` 已在 SELF_QA payload 中加入 `current_assertion`，source/verify 通过 `cloze_sidecar` 绑定条件位置，不再要求条件文本必须存在于自然 question 字符串里。
- `cloze_prompts.py` 明确 SELF 只是 mask recoverability，不是 factuality；SOURCE/VERIFY 要求每个 condition key 一行、同一事件、不可强选最近事件。
- `json_framing` 已把完整首 root + fence/尾随符号记为 `recovered`，并记录 raw span/suffix；这比 v2 strict JSON 直接丢弃更适合当前 reader 行为。

仍需注意的接口风险：

1. **SELF 的 question 字符串仍写着 “using the source”。** SELF prompt 要求用 current assertion 恢复 mask，但 question 文本本身是 source 任务措辞。若 v3 self 仍出现整句/错角色答案，应拆出 `self_question` 与 `source_question`，或把公共 cloze 文本改成中性：“Fill `<MISSING_SLOT>` in this assertion.” 这不是新增方法，只是避免自检目标和问题文字冲突。

2. **unassigned context 会提高保守性。** 当前编译器把角色之间未解析的字母数字片段也加入条件表。这能防止漏掉限定条件，但复杂句里可能把转折、例子标记、指代残片都变成必须 source 支持的条件。若 v3 大量失败为 `unparsed_context_* missing/conflicting`，不能靠删除这些条件来提升覆盖；应把它计入 frame granularity / event alignment 问题。

3. **single-role source C 仍要求精确 source answer quote。** 对抽象摘要、改写、合取拆分和源中分散证据，source 可支持但没有可直接替换的短 quote。此时不要伪造 grounded_replacement；可以进入 event-level 或 withholding-only 分支，但不能算 slot-level C 证书。

4. **若 hidden alias 只在后加入的 unassigned segment 中暴露，source payload 会拒绝，但要保证该拒绝记录为 question failure 而不是运行崩溃。** 方法上必须把这类情况计入 `cloze_leak_rejected`，而不是 silent exception 或丢样本。

## 为什么 single-role mask 在多错事件中会结构性拒绝

设 response frame 有角色 `{subject, predicate, object, time, condition}`，真实来源没有这个事件，或只有一个相似事件但 subject/predicate/time 同时不同。当前 single-role 方案每次只隐藏一个角色，其余角色全部作为条件。若隐藏 object，错误 subject/predicate/time 会被要求在来源中支持；若隐藏 subject，错误 object/predicate/time 会被要求支持。于是所有 mask 都会因为非目标条件不成立而 U。

这不是 reader 太笨，也不是 prompt 不够强，而是定义要求如此。E/C 只有在非目标条件均被来源同一事件支持时才可判定；N 也只有在非目标条件成立而目标缺失时才可判定。多角色错配缺的是“这个 response event 与哪个 source event 对齐、哪些 role 错”的上层结构，而不是某个槽位答案。

SRLScore 的角色 tuple 和 FENICE 的 claim/premise alignment 都说明同一事件对齐是事实结构的前置步骤；但它们的 max-over-source 或 weighted role similarity 不能直接移植成证据，因为最近 event 不是 native 信息采用证书。本项目需要的是保守的事件级对齐失败/成功状态，而不是放宽为“最相似来源事件”。

## 最小结构改动建议

保留 v3 cloze 作为主路径。只在以下触发条件满足时，增加一个 **event-level source alignment fallback**；不要替换掉 cloze，也不要新增训练组件。

### 1. 添加 frame-level source event alignment

对每个 response atomic frame，在 source 侧生成或核对候选 source event 表：

```json
{
  "response_frame_id": "...",
  "candidate_events": [
    {
      "source_event_id": "...",
      "event_identity": "same_event|near_event|unrelated|uncertain",
      "role_table": [
        {"role": "subject", "status": "equivalent|conflicting|missing|extra|uncertain", "response_quote": "...", "source_quote": "...", "source_spans": [[0, 0]]}
      ],
      "joint_evidence_quotes": ["..."],
      "unresolved_alternatives": ["..."]
    }
  ],
  "decision": "slot_ready|multi_role_event_mismatch|event_absent_estimate|ambiguous|unresolved"
}
```

这一步可以仍由冻结 Qwen 完成，但必须保持约束：不读 RAGTruth labels，不读 native effects，不强选最近事件；所有 source quotes 精确定位；多个相关候选互相冲突时 `ambiguous/unresolved`。

### 2. 只从 event alignment 派生两类可测目标

- `slot_ready`：存在同一 source event，且除一个目标 role 外，其余条件等价并有引用。走现有 cloze single-slot C/N/E 路径。
- `multi_role_event_mismatch`：存在同一或明确近邻 source event，但两个以上 response roles conflicting/missing。不要把它拆成多个假 slot C；构造 `joint_event_contrast`。

`joint_event_contrast` 的 A/B 规则应保守：

- 若所有错误 roles 都有唯一精确 source spans，A 是按 role 同时替换后的完整事件；替换必须通过 `only_changed_roles`、语法和条件保持检查。
- 若 source 只说明不存在/未说明该事件，A 是固定 event-level withholding，例如 “the source does not specify this event”，单列 `event_commitment_vs_withholding`。
- 若需要 reader 自由改写整个句子才生成 A，则不进入 native 证书；只保留 `event_alignment_proxy_unresolved`。

这保持共同 prefix 完整事件 F，不退回首 token，也不把复杂事实归因伪装成精确单槽定位。

### 3. 把自动回看/传播绑定到 event 或 slot scope

当前连续错误规则要求当前槽位映射到前一句具体答案槽位。对 event-level fallback，应输出两种 scope：

- `slot_continuation`: 前后同 slot/同问题，且 origin selection 是 mapped previous answer span。
- `event_continuation`: 前后为同一 event family 或明确 elaboration，且 native origin 只能定位到 previous event frame，不可称 previous answer slot。

如果只拿到 event_continuation，事实传播结论必须是“事件级历史依赖”，不是准确 slot 级回看。这样不会因为多角色事件导致全拒绝，也不会把整句历史依赖写成具体答案传播。

## 失败触发条件

v3 full36 后应按下列触发条件决定下一步，不要用少数好例子推进论文 claim：

1. **格式仍主导失败**：`frame_json_unrecovered` 或 `source_json_unrecovered` 占 reader calls > 10%。先修 parser/stopper；不要进入机制 claim。

2. **SELF 仍主导失败**：`self_wrong_role_or_span` 占 compiled questions > 20%。先拆 `self_question`/`source_question` 或改 current-assertion recoverability；不要增加 native 结构。

3. **single-mask collapse**：同一 frame 的两个以上 role-mask 均因非目标条件 missing/conflicting/uncertain 而 U，且这类 frame 覆盖 > 20% assertion words 或 > 30% compiled frames。进入 event-level source alignment；继续 prompt 修补是低价值。

4. **unparsed-context collapse**：失败条件主要来自 `unparsed_context_*`，且人工读样本显示这些片段是结构边界/修辞/列表编号而非事实限定。此时需要更好的 frame segmentation；不能简单删除 unparsed context 提高覆盖。

5. **derived evidence collapse**：source/verify 都认为同事件支持，但缺 exact `source_answer_quote`，导致 grounded replacement 无法构造。将其从 slot C 分母移到 `derived_support_no_exact_contrast`；只有可构造 event-level A/B 才进 native。

6. **source ambiguity**：多个 source candidates 满足部分角色且给出不同目标。保留 `ambiguous_event_alignment`，不选最高相似或最长引用。

7. **native starvation**：semantic valid anchors 仍不足以选择风险 claims，导致 B/C/D forward 近零。该结果说明 A anchor 仍未给机制层供给目标，不能解释为“内部没有信息”。

## 必须报告的覆盖门槛与分母

沿用最终方案中已冻结的主门槛，但增加结构分解。门槛不是调参目标，而是 claim language gate。

主门槛：

- 有效语义问题覆盖 ≥ 80% assertion words。
- 可构造 A/B 对比 ≥ 70% risk claims。
- 至少一种完整内部选择性证书 ≥ 50% risk claims。
- 若低于任一门槛，不能写“完整自然方法有效”；只能写对应失败阶段。

新增结构分母：

- `frame_parse`: strict / recovered / unrecovered。
- `frame_compile`: accepted frames、duplicate-role rejects、overlap rejects、long-role rejects、missing-subject/predicate rejects。
- `cloze_compile`: candidates、hidden-alias rejects、marker collision、ambiguous claim occurrence、unassigned-context-added。
- `self_recoverability`: exact pass、wrong role/event、container span、null、leak flag、ambiguous-within-claim。
- `source_condition`: all conditions supported、condition missing/conflicting、condition table invalid、source quote invalid、unresolved candidates。
- `single_mask_scope`: slot_ready、multi_role_event_mismatch、event_absent_estimate、derived_support_no_exact_contrast、ambiguous_event_alignment、unresolved。
- `contrast`: grounded_replacement valid、event_joint_replacement valid、withholding valid、edit invalid、no exact contrast。
- `native_supply`: selected risk claims、B proposed, C labeled, D validated, actual forward calls, native tokens。
- `mechanism_scope`: finite_position_effect、origin_mediated_effect、event_level_origin_only、slot_continuation、event_continuation、pairedR、MLP interaction、unresolved。

All denominator rows must include absolute counts over the whole frozen 36-response roster, per task, and per C/N/event-level branch. Report all words, assertion words, annotated error words, scored words, and abstained words. Do not remove invalid/unknown frames from the denominator.

## Decision for immediate v3

Run v3 full36 if engineering checks close, because it is the correct test of whether v2 was primarily a cloze/self-recoverability interface failure. But pre-register the interpretation now:

- If v3 obtains high self/source slot coverage and native claims execute, then judge mechanisms on their own certificates.
- If v3 mostly fails via single-mask collapse, stop prompt iterations and add event-level source alignment / joint-mask fallback.
- If v3 still yields native0, the result is an anchor/contrast supply failure, not evidence that the frozen model lacks internal source/history information.
- If v3 only succeeds on simple one-wrong-role cases, claim only slot-level coverage for that subset and report multi-role event failures as unresolved.

This keeps the project on the user's core problem: accurate automatic lookback, attribution, and propagation under natural constraints. It avoids turning the next cycle into another format experiment or a hand-picked relation-impact demonstration.
