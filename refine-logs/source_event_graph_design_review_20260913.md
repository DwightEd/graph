# Source-event graph 条件方案审查

日期：2026-09-13  
模式：`research-refine` 方法审查；真实 collaboration GPT-5.5 fallback，Codex MCP 不可用。本审查只针对 `docs/SOURCE_EVENT_GRAPH_DESIGN_20260913.md` 的 v3 后条件方案，不改变正在运行的 v3 协议，不是实证通过结论。

## 结论

这个方向值得作为 v3 后的最小结构候选：先冻结 source-only 事件坐标，再让 response alignment 只引用固定 ID，确实能减少当前 sourceQA 改写 `condition_role`、编造 `source_quote`、逐问题重复抽取造成的接口脆弱性。它也保留了“不以幻觉标签训练”的主约束：Qwen 仍是冻结语义编译器/核验器，不是用幻觉标签训练出的 detector。

但当前草案还不能直接进入实现。核心缺口有三类：第一，source graph 仍过度依赖 Qwen 一次性字段输出，若字段漏抽或关系连错，后续合法 ID 会给出虚假的确定感；第二，`same_event` 在多角色错配时没有定义身份依据，可能把语义上已经不是同一事件的东西强行 joint edit；第三，ID 合法性只证明“文本片段存在”，不证明“片段之间的角色、谓词、条件、推导关系正确”。这些不是实验后可调阈值问题，而是接口定义问题。

判定：**REVISE before implementation**。不阻塞 v3 继续跑，但作为 v4/vNext 结构需要先补以下 Required。

## Required 1：把 source graph 降级为坐标化候选库，不得作为来源全集或事实真值

草案已写“抽取失败与未覆盖单元仍在清单中，禁止把没有抽到事实当作来源不存在事实”，但接口还没有强制这一点。`compile_source_graph(...) -> {nodes, events, relations, coverage, failures}` 需要把 raw source inventory 作为一等对象，否则后续 `align_events` 很容易只在抽取图上查不到就进入 N/withholding。

最小接口修订：

```python
SourceGraph = {
  "source_sha256": str,
  "raw_units": [
    {"unit_id": str, "char_span": [int, int], "text": str, "token_span": [int, int]}
  ],
  "mentions": [
    {"mention_id": str, "unit_id": str, "char_span": [int, int], "surface": str}
  ],
  "event_assertions": [
    {
      "event_id": str,
      "predicate_mention_id": str,
      "role_edges": [{"role": str, "mention_id": str, "edge_id": str}],
      "modifier_edges": [{"type": str, "mention_id": str, "edge_id": str}],
      "status": "verified|reader_estimated|partial|unresolved"
    }
  ],
  "unparsed_units": [
    {"unit_id": str, "reason": str, "may_contain_fact": bool}
  ],
  "coverage": {"token_covered": int, "token_total": int, "unparsed_token_total": int}
}
```

`mention_id` 应由 source SHA + unit/span/surface 生成，**不要把 reader 的 role label 放进稳定坐标 ID**。role/predicate/condition 是可错的关系断言，应作为 `edge_id` 和属性保存。这样 Qwen 改了 role label 不会改写坐标身份，也能单独审计“引用合法但关系错误”。

N/whole-event missing 不能由 `event_assertions` 缺失触发。它至少需要：

1. raw source 全文双盲 not-stated 核验；
2. 与 response event lexical/embedding 检索相关的 raw units 均被检查或被标为 unresolved；
3. source graph 的 unparsed/partial units 没有覆盖该 event family 的未决候选。

达不到这三点只能是 `graph_extraction_absence_unresolved` 或 `fulltext_absence_unresolved`，不能进入确定 N 或 native withholding 证书。

## Required 2：限制 Qwen 输出为指针和有限枚举，禁止自由生成 source_quote / 关系文本

当前设计仍让冻结 reader 产生复杂事件、角色、关系字段。即便程序验证 quote，关系本身仍可能是幻觉。最小抗脆弱接口应把 Qwen 从“写事实图”改成“在给定文本坐标上选指针并打有限标签”。

建议拆成两步：

```python
propose_source_mentions(source_units) -> mention spans only
bind_source_events(source_units, mention_ids) -> event_assertions using mention_id + role enum + modifier enum
```

所有 downstream payload 只允许引用 `mention_id/event_id/edge_id`，不能再发送任意 `source_quote` 字符串。需要显示文本时由程序从 ID 反查原文。Qwen 输出中若出现未登记 ID、重复 key、跨 unit 不合法 span、同一 role 多个互斥值且无 coordination 标记，直接 `invalid_graph_field`，保留到分母。

这不会把系统改成监督 detector，也不改变 frozen-reader 约束；它只是把 reader 的自由文本通道压缩为可审计指针通道。

## Required 3：`same_event` 必须有身份依据；多角色错配不能默认 joint

草案的 “同一事件 / near-event / unknown + 多角色替换” 是对 single-mask collapse 的正确方向，但 `same_event` 本身需要先定义。如果 response 的 subject、object、time、condition 中多个角色都和 source 冲突，那么“同一事件”可能已经语义自相矛盾。例如同一 predicate 下 subject/object 对调，或阶段、时间和数值都变了，joint replacement 生成的 A 可能是另一个事件，而不是对 B 的事实修复。

最小修订：`align_events` 必须输出 `event_identity_basis`，列明哪些未被替换的角色/谓词/条件支撑 same-event 判断，哪些冲突角色被允许作为 edit targets。

```python
EventAlignment = {
  "response_event_id": str,
  "candidate_source_events": [
    {
      "source_event_id": str,
      "identity": "same_event|same_event_family|near_event|unrelated|unknown",
      "identity_basis": [
        {"response_edge_id": str, "source_edge_id": str, "basis_type": "predicate|participant|time|condition|coreference"}
      ],
      "conflicting_edges": [...],
      "missing_edges": [...],
      "unresolved_competitors": [...]
    }
  ]
}
```

`joint_event_contrast` 只能在 `identity=same_event` 且身份依据不全部来自将被替换的角色时生成。若只能判为 `same_event_family` 或 `near_event`，可以记录 `event_family_mismatch`，但不能发单事件机制证书。若多个 source event 分别支持不同 role，必须形成已验证 hyperedge；不能把来自不同事件的 role 拼成一个 A。

## Required 4：ID 合法性、关系正确性、对比可构造性必须三段判定

冻结 ID 解决的是“引用是否存在”，不是“引用是否适用”。后续必须避免把 `event_id` 引用成功当作 relation proof。

每个可进入 native auditor 的 contrast 至少需要三段状态全部通过：

1. `id_integrity`: 所有 mention/event/edge ID 属于冻结 source graph，span 可反查，hash 匹配。
2. `relation_verification`: 两次独立核验认为这些 edge 在同一事件内支持指定 predicate/role/condition，且保留竞争 event。
3. `contrast_validity`: A/B edit 只改变允许的 role 或固定 withholding；A 由同一个 source event/hyperedge 支持；B 与 A 在目标范围内不兼容；共同 prefix 与完整事件 F 可计算。

建议新增接口：

```python
verify_event_relation(source_graph, event_alignment, visible_raw_source) -> RelationVerification
compile_event_contrast(alignment, verification, response) -> ContrastSpec | Unresolved
```

`RelationVerification` 不应返回自由解释文本作为证据，只返回 ID、状态、失败原因和必要的短 rationale。rationale 可读，但不参与程序判定。

## Required 5：derived_support 与 whole-event missing 不能混进机制证书

草案把 `derived_support`、`whole-event missing`、`joint` 都接入 native auditor 的范围描述是合理的，但要把 scope 写得更硬。

- `derived_support`：若没有精确 A/B 对比，只能是 semantic-only support/unsupported 状态，不能进入 position/origin/pairedR/MLP 证书。若能构造 event-level A/B，必须显式记录 `derived_exact_contrast=true` 和支持它的 source hyperedge。
- `whole-event missing`：只能是 `commitment_vs_withholding`，不能称来源给出正确答案。N 还要依赖全文缺失核验，而不是 source graph 未抽到 event。
- `joint_event`：证书 scope 只能是 event-level。即使 native group 对 F 有效，也不能拆成某个数字/实体的单槽归因。

输出 JSON 应至少包含：

```json
{
  "semantic_scope": "single_slot|joint_event|event_commitment_vs_withholding|derived_semantic_only|unresolved",
  "native_scope_allowed": "slot|event|none",
  "why_not_slot": "multi_role_edit|derived_no_exact_contrast|missing_full_event|identity_unresolved|null"
}
```

## Required 6：response-event graph 也要坐标冻结，不能把 source graph 的字段反向修正 response

草案说 response 值可参与比较但不能反向修改 source graph；还需反过来规定 source graph 不能重写 response frame。否则 alignment 阶段可能用 source event 的 role schema 重新解释 response，掩盖原回答里的错误谓词、条件和指代。

最小接口：response frame 先独立编译为 `ResponseGraph`，保存 exact response spans、atomic frame id、role mentions、condition mentions、previous-link candidates。`align_events` 只建立跨图边，不修改任一侧节点。若发现 response role segmentation 错，只能输出 `response_frame_invalid` 或 `frame_granularity_unresolved`，不能偷偷替换成 source-friendly frame 后进入证书。

## Required 7：连续错误/恢复边必须验证关系，不只引用前后事件 ID

模块4 的方向正确：slot 目标对应 slot，joint 目标对应 event。但传播结论还需要一个关系核验层。`previous_event_id -> current_event_id` 的合法 ID 只说明两段文本存在，不说明当前陈述派生自前一错误，也不说明它是纠正。

建议输出：

```python
ContinuationEdge = {
  "edge_type": "slot_continuation|event_continuation|supported_recovery|history_dependency_only|unresolved",
  "previous_response_ids": [...],
  "current_response_id": str,
  "semantic_link_status": "verified|conflicting|uncertain",
  "direction_status": "error_reused|error_corrected|new_topic|unknown",
  "origin_mediation_status": "mapped_previous_slot|mapped_previous_event|not_mediated|unresolved"
}
```

只有 `semantic_link_status=verified`、错误方向明确、且 native origin 选择映射到 previous slot/event 时，才能发布 continuation/recovery。否则只能说历史依赖，不能说错误传播。

## 最小架构是否仍过度依赖 Qwen

是，但可接受为当前约束下的冻结语义编译器，前提是把它的权限收窄到：

- 在程序提供的 source/response 坐标上选择 span/ID；
- 从有限枚举中标 predicate/role/condition/relation status；
- 提供可审计失败原因；
- 不生成自由 source quote，不生成新 source node，不根据 native effect 或风险分数选证据。

这样仍依赖 Qwen 做语义估计，但不会让 Qwen 的自由字段成为不可追踪真值。它符合“不以幻觉标签训练”：没有用 hallucination labels 训练检测器，也没有把 reader 输出当 GT；所有 reader 判断都作为模型估计进入分母和失败表。

## 可执行的最小接口顺序

建议把 vNext 固定为下面的顺序，避免后续互相污染：

1. `build_raw_source_inventory(source)`: 程序生成 source units/token spans。
2. `compile_source_graph(source_units)`: Qwen 仅输出 mention pointers 和 event relation pointers；程序冻结 graph hash。
3. `compile_response_graph(response)`: 独立冻结 response event/role/condition spans。
4. `align_events(source_graph, response_graph, raw_source)`: 只创建跨图 alignment，不改两侧图。
5. `verify_event_relation(...)`: 对候选 event/role/condition 做双盲关系核验，保存竞争候选。
6. `compile_event_contrast(...)`: 生成 `single_slot`、`joint_event`、`event_commitment_vs_withholding` 或 `unresolved` 的 exact A/B。
7. `native_audit(ContrastSpec)`: 沿用当前 E/V/R/pairedR/origin/MLP，但按 `native_scope_allowed` 限制 claim language。
8. `merge_outputs(...)`: 报告 raw inventory、graph extraction、alignment、contrast、native 分母；不得从有效分母中删除 invalid/unresolved。

## 触发条件与报告要求

v3 完成后再启用该结构。若 v3 失败主要来自字段格式或 source quote 编造，优先实现 Required 1/2 的 pointer graph；若失败主要来自 same-event 多角色错配，实现 Required 3/4/5 的 event alignment 与 joint contrast；若连续错误仍是假阳性，优先实现 Required 7。

任何结果报告必须写清：

- source graph coverage 是抽取覆盖，不是来源事实覆盖；
- `graph_absent` 不是 `not_stated`；
- legal ID 不是 relation support；
- `same_event_family` 不是 `same_event`；
- joint/event-level certificate 不提供单槽定位；
- supervised probe 如存在仍只能是信息可解码上界，不进入当前无标签主方法。

## 最终 Required 状态

当前草案的方向保留，但进入实现前有 7 个 Required：

1. raw source inventory 与 extraction absence 分离；N 不可由 graph absence 触发。
2. Qwen source graph 输出改为 pointer + finite enum，禁止自由 source_quote/新节点。
3. `same_event` 定义 identity basis；多角色冲突不能默认 joint。
4. ID integrity、relation verification、contrast validity 三段分离。
5. `derived_support`、whole-event missing、joint-event 的 native scope 明确降级。
6. response graph 独立冻结，不由 source graph 反向修正。
7. continuation/recovery 边需要关系与方向核验，不只引用前后 ID。

修完这些后，该方案才是一个可实现的 v3 后最小结构，而不是另一个由冻结 reader 字段输出驱动的脆弱 QA 管线。

---

## 快速复审补记：7项 Required 修订后是否允许 CPU 模块实现

日期：2026-09-13 追加。范围只复核 `docs/SOURCE_EVENT_GRAPH_DESIGN_20260913.md` 新增的“审查后的具体接口修订（实施前）”，不评价实证通过，不要求启动 native 或 GPU。

结论：**可以进入最小 CPU 指针编译 / 引用完整性模块实现**。新增 7 条已经把前一版最危险的语义漂移压住了：raw inventory 与 event extraction 分离，模型只返回局部 token pointer 和有限枚举，quote/ID 由程序生成，response graph 独立冻结，`identity_basis` 成为 same-event 前置条件，reference / relation / contrast 三段分离，joint/derived/whole-event 都被明确限制 scope，历史边也保留语义、方向、native 介导三重核验。特别是第 67 行写明“首个 source-graph 实现只支持 reference_integrity 和对齐记录，joint/missing 的 native 扩展须后续单独实现审查”，这足够防止 CPU 编译器被误用成事实判断器。

剩余 Required 降为一个实现前接口细节，而不是方法阻塞：Data2txt 的 Python 字典 repr 必须有独立输入类型和白名单解析协议。建议写成：

```python
compile_data2txt_literal_source(raw: str) -> RawInventory
```

实现约束应固定为：

- 只用 `ast.parse(..., mode="eval")` 后遍历 `ast.Expression / Dict / List / Tuple / Constant`；禁止 `eval`、`literal_eval` 之外的执行语义、Name、Call、Attribute、Subscript、BinOp、JoinedStr、Bytes 等节点。
- key path、container path、list index、原始字符区间全部保存；无法定位到唯一原文区间的字段只能做 `structure_field_unlocalized`，不能生成可引用 mention。
- `None` 保留为 `unknown/null_literal`，不得编译成 not-stated 或 missing fact。
- 字典 key 只是字段 provenance，不自动等于 predicate 或 relation truth。Data2txt 可以生成 `field_edge` / `record_edge`，但若要声称自然事件关系，仍须进入后续 `relation_verification`。
- 对重复 key、非字符串 key、混合类型 list、过深/过长 literal、解析失败分别记录失败类型，保留 raw source 全文分母。

如果上述白名单被落实，`source_event_graph.py` 的第一版合理边界应是：构造 raw inventory、mention pointer、event/field pointer graph、hash/provenance、reference_integrity 测试，以及 response/source 双图不可互改的单元测试。它不应输出 E/C/N、same_event 事实判断、joint contrast、whole-event missing 或 native candidate。这样实现不会违反当前“不以幻觉标签训练”的约束，也不会把冻结 reader/结构字段当 GT。

当前 Required 状态：前 7 项方法 Required 已足够关闭，允许 CPU 模块实现；新增 Data2txt 白名单解析为实现 Required。完成后再审代码，而不是在设计层继续扩大结构。
