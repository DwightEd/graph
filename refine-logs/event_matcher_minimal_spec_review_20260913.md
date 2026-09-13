# Event-local matcher 最小规格审查

日期：2026-09-13  
模式：`research-refine` 有界方法规格审查；真实 collaboration GPT-5.5 fallback，Codex MCP 不可用。范围仅评估下一版候选层：event-local、source 可重复利用、半松弛图匹配。未写代码、未启 GPU、未改变冻结 v3。

## 结论

首版不应实现完整 `partial/unbalanced FGW`。更小且更稳的目标是 **逐角色候选 + 事件联合约束**：每个 response role 独立保留 top-k source candidates 和 null/unknown，再用 event-level pairwise relation consistency 做局部 beam/ILP rerank。FGWEA 的启发保留在“语义 cost + 结构/关系一致性 + progressive anchors”，但不要把全局 coupling、source 侧边际约束或 dense GW refine 搬过来。

原因很直接：FGWEA 对齐的是两张已经给定、同类、近似一对一的 KG。我们面对的是 response event graph 到 source text/field graph 的候选归属：source 事实可以被多个 response 事件复用；source 可能是 Data2txt 字典字段而非自然谓词；response 的多个 role 可能同时错误；null 表示 matcher 未找到或未搜索，不能表示事实不存在。完整 FGW 会把这些问题伪装成优化超参，导致 source 容量、拓扑不对称和 null 语义一起漂移。

判定：**采用 semi-relaxed role assignment with event-local structural reranking**。可选地把 FGW/GW 项写成局部 pairwise consistency penalty；不要写成全图最优传输或事实判别器。

## 对 FGWEA 原文的约束映射

FGWEA 的实际路线不是“直接全图 FGW”：它先用冻结语义 embedding 做 WD 初始匹配，取高置信 anchors，再用 anchors 近似结构/关系相似度，反复 multi-view OT，最后才做 GW refinement。论文还明确直接优化大稀疏图 FGW/GW 会不稳且低效。

映射到本项目：

- 可借鉴：progressive anchors、冻结表示 cost、关系/结构一致性作为 matching objective 的一部分。
- 不可照搬：一对一或近一对一 entity alignment 假设、同质 KG 拓扑、全局 source 边际约束、把高 coupling 当等价真值。
- 必须新增：role-level null、未搜索分母、source 可重复利用、Data2txt field-path 与自然语言 role 的 schema bridge、same-event identity basis、relation verification 与 contrast validity 的后置审计。

因此首版若叫 FGW，容易过度承诺。建议命名为 `event_local_semirelaxed_matcher`，把 FGW 只作为“关系一致性项的来源”，不作为论文主机制名。

## 最小目标函数

对每个 response event `e_r` 单独求解。设 response roles 为 `R = {r_i}`，每个 role 有候选集合：

```text
C_i = top_m_source_mentions_or_fields(r_i) ∪ {NULL_i, UNKNOWN_i}
```

候选来自冻结 source graph/raw inventory 和固定检索，不由 Qwen sourceQA 生成。source 侧不设容量约束：同一个 source mention/field 可被多个 response event 或多个 response mention 复用；仅在同一 event 内对互斥 role 作局部重复惩罚。

离散形式最清楚：

```text
score(z | e_r) = Σ_i U_i(z_i)
               + β Σ_(i,j) P_ij(z_i, z_j)
               + ρ IdentityPenalty(z)
               + κ AmbiguityPenalty(z)
```

其中 `z_i ∈ C_i`。取 cost 最低的 top-K assignments。

`U_i(c)` 是 role-level unary cost：

```text
U_i(c) = w_hid * d_hidden(r_i, c)
       + w_surf * d_surface(r_i, c)
       + w_type * d_type(r_i, c)
       + w_num  * d_number_unit(r_i, c)
       + w_path * d_field_path(r_i, c)
       + w_null * 1[c=NULL_i]
       + w_unk  * 1[c=UNKNOWN_i]
```

- `d_hidden` 使用冻结同一 observer、同一 layer set 的 span-pooled features；不得使用 RAGTruth label、reader E/C/N、B/A F 或 native effect。
- `d_surface` 只做字符串、日期、数字、单位、别名的程序特征。
- `d_type` 限制 role enum / value type / POS / field value type。
- `d_field_path` 只用于 Data2txt 或结构化来源；它是 provenance compatibility，不是语义真值。
- `NULL_i` 是“matcher 无合格来源候选”；`UNKNOWN_i` 是“候选池/抽取覆盖不足”。二者都不是 N。

`P_ij(c,d)` 是 event-local pairwise relation penalty：

```text
P_ij(c,d) = mismatch(
  response_edge_type(i,j),
  source_edge_type(c,d),
  record/event co-membership,
  condition attachment,
  order/proximity bucket,
  field-path sibling/parent relation
)
```

它是 FGWEA 的 relation/structure idea 的局部化版本：不做全图 `Σ_ijkl L(A_ij, A'_kl) π_ik π_jl`，只在一个 response event 的候选 beam 上比较 role-pair 关系。

若需要连续松弛，可写为半松弛 row-stochastic coupling：

```text
min_Π  Σ_iΣ_c Π_ic U_i(c)
     + β Σ_(i,j)Σ_cΣ_d Π_ic Π_jd P_ij(c,d)
     + ε Σ_iΣ_c Π_ic log Π_ic
s.t.  Σ_c Π_ic = 1 for every response role i
      Π_ic ≥ 0
      no source-column marginal constraint
```

这里没有 `Π^T 1 ≈ b`，因为 source 可重复利用；source 容量项会把合法复述挤进 null。实现上首版更建议离散 top-m + beam search，比 Sinkhorn/GW 更容易审计失败原因。

## Top-k 候选生成

每个 role 的候选必须先由固定、可复现的 retrieval 产生：

1. exact / normalized number-date-unit / string alias anchors；
2. Data2txt field value and key-path anchors；
3. frozen hidden cosine top-m；
4. same record/event neighborhood expansion；
5. optional base attention/message-location prior，但只作 tie-break 或附加 feature，不作 truth score。

每步记录 `searched_source_units`、`unsearched_source_units`、`candidate_pool_size`、`candidate_generation_reason`。如果 source graph 没抽到对应字段或自然事件，仍保留 raw inventory 检索结果和未覆盖分母。不得只把进入 top-k 的 source nodes 当全来源。

## Data2txt field-path 图与自然 role 图的不对称

Data2txt source 不是自然语言谓词图。首版应该把它编成 `record-field-value` provenance graph：

```json
{
  "record_id": "...",
  "field_path": ["team", "wins"],
  "value_id": "...",
  "value_type": "number|date|string|null|list|dict",
  "siblings": ["..."],
  "parent_path": "..."
}
```

自然 response role 图是 `predicate-role-condition`。二者的桥接只能通过有限 compatibility feature：field key tokens 与 predicate/role tokens 的 lexical similarity、value type、same record、sibling constraints、units。字段 key 合法不等于关系正确；例如 `duration=None` 不能支持“lasted three months”，只能给 `unknown/null_literal` 与 `relation_verification_required`。

对 Data2txt 不要硬造 predicate 句，也不要让 matcher 把 `field_path` 当自然事件身份。可输出 `source_structure_type=data2txt_field_record`，后续 relation verifier 才判断 response predicate 是否适用于该 field path。

## 同值、多事件与 source 复用

原文不同事件同值是高风险点。匹配不得只按 value surface 合并 source candidates。

Required：

- 每个 source value occurrence/field value 保持独立 ID，即使 surface 相同。
- role assignment 输出 `source_event_or_record_id`，不能只输出 value span。
- 若多个 candidate assignments 只有同值但不同 event/record，标 `ambiguous_same_value_different_event`，全部保留到 top-K 或 unresolved。
- controls 必须包含 same-value wrong-event 或 same-field unrelated record，如果存在。

source 可重复利用只跨 response events 放开；同一 response event 内，两个不同 semantic roles 映射同一 source mention 时必须满足 `allowed_coreference/reflexive/coordination`，否则加 `duplicate_role_target_penalty` 或直接 invalid。

## 多角色都错误时如何保持 event identity

matcher 可以支持 multi-role error，但不能强行把整事件错配叫 same_event。必须区分三个状态：

1. `same_event_with_editable_roles`：存在未被编辑的 identity basis，例如 predicate + subject、record ID + field family、stage + entity。joint edit 可进入后续 relation verification。
2. `same_event_family_only`：只知道属于同一实体/主题/表格记录附近，但 predicate、stage 或关键条件也冲突。只能输出 event-family candidate，不能编 joint contrast。
3. `identity_unresolved_or_absent`：所有 identity roles 也需要改，或只有相似值/邻近文本。不能进入 native contrast。

因此输出 assignment 时必须列出：

```json
"identity_basis": [
  {"response_role_id": "...", "source_role_id": "...", "basis_type": "predicate|subject|record_id|stage|condition", "changed": false}
],
"editable_roles": ["..."],
"identity_status": "same_event_with_editable_roles|same_event_family_only|identity_unresolved_or_absent"
```

`joint_event_contrast` 的前置条件是 `identity_status=same_event_with_editable_roles`，且 identity basis 不全来自将被替换的 roles。

## 输出合同

Matcher 输出只是一份候选合同，不是事实标签：

```json
{
  "matcher_version": "event_local_semirelaxed_v1",
  "source_graph_hash": "...",
  "response_graph_hash": "...",
  "feature_manifest_hash": "...",
  "response_event_id": "...",
  "source_reuse_policy": "unconstrained_across_response_events",
  "source_capacity_used": false,
  "role_candidate_pools": [
    {
      "response_role_id": "...",
      "role_type": "...",
      "candidates": [
        {
          "candidate_id": "...",
          "source_node_id": "...|NULL|UNKNOWN",
          "source_event_or_record_id": "...|null",
          "unary_cost": 0.0,
          "cost_terms": {"hidden": 0.0, "surface": 0.0, "type": 0.0, "number_unit": 0.0, "field_path": 0.0, "null": 0.0},
          "retrieval_reasons": ["exact_number", "field_path", "hidden_topk"],
          "searched_scope": "topk_pool|raw_inventory|graph_only"
        }
      ],
      "unsearched_count": 0,
      "null_candidate_present": true,
      "unknown_candidate_present": true
    }
  ],
  "event_assignments": [
    {
      "assignment_id": "...",
      "rank": 1,
      "total_cost": 0.0,
      "role_links": [
        {"response_role_id": "...", "candidate_id": "...", "mass_or_score": 1.0, "status": "matched|null|unknown"}
      ],
      "pairwise_penalties": [
        {"response_edge_id": "...", "source_edge_basis": "...", "penalty": 0.0, "reason": "same_record|edge_type_mismatch|condition_attachment_mismatch"}
      ],
      "identity_basis": [...],
      "identity_status": "same_event_with_editable_roles|same_event_family_only|identity_unresolved_or_absent",
      "changed_or_unmatched_roles": ["..."],
      "ambiguity_status": "low|same_value_competition|multi_event_competition|coverage_unresolved",
      "candidate_controls": [
        {"control_id": "...", "type": "same_value_wrong_event|same_field_unrelated_record|same_type_shuffled_structure"}
      ],
      "downstream_allowed": "relation_verification_only|contrast_possible_after_verification|unresolved_only"
    }
  ],
  "denominators": {
    "source_units_total": 0,
    "source_units_searched": 0,
    "source_units_unsearched": 0,
    "source_graph_nodes_total": 0,
    "response_roles_total": 0,
    "roles_with_non_null_candidate": 0,
    "roles_null_top1": 0,
    "roles_unknown": 0,
    "events_candidate_matched": 0,
    "events_ambiguous": 0,
    "events_identity_unresolved": 0
  }
}
```

Downstream must consume this as immutable input. Qwen relation verification may reject or refine statuses, but may not add new source nodes to the same matcher result. Any newly discovered span belongs to a separate rerun/version with a new denominator.

## 不应加入的步骤

- 不加 RAGTruth hallucination-label training、SiGHT-style synthetic hallucination binary labels、focal-loss detector 或 GAT 主线。
- 不做 dense full-source FGW/GW，也不调一个 source marginal capacity。
- 不用 matcher cost、null mass、attention concentration 或 OT distance 直接判 supported/conflict/N。
- 不让 Qwen 生成 free-form source answer/quote 后回填候选池。
- 不在 relation verification 失败后用最高相似 event 兜底。
- 不把 Data2txt key path 直接当自然谓词真值。
- 不在 matcher 阶段计算 B/A F 或 native intervention；否则候选选择会和机制证书循环。
- 不把多个 role 的 joint edit 拆成单槽归因。

## Required 状态

这份草案若按“半松弛图匹配”实现，还需要收窄为上述 `event_local_semirelaxed_matcher` 才足够最小。具体 Required：

1. 明确不用 source-column capacity；source 复用跨 response events 完全允许。
2. 把完整 partial/unbalanced FGW 降级为逐角色 top-k + event-local pairwise relation penalty。
3. matcher 的 null/unknown 只表示候选状态，不触发 N。
4. Data2txt 使用 record-field-value provenance graph，field_path 只作 compatibility feature。
5. 同值不同事件必须保留多候选和 ambiguity，不合并 ID。
6. joint contrast 必须依赖未编辑的 identity_basis；否则降为 family/unresolved。
7. 输出合同必须包含候选池、未搜索分母、cost terms、controls 和 downstream_allowed。
8. 关系核验、contrast、native 仍在 matcher 之后，不能反向改候选。

修完这些，下一版候选层是可实现的最小算法：它能降低 Qwen 多字段 sourceQA 的候选搜索负担，同时保持“不以幻觉标签训练”和“匹配不等于真假”的锚点。

## Sources used

- FGWEA: https://arxiv.org/abs/2305.06574 and https://arxiv.org/pdf/2305.06574
- Hallucination Span Detection with Input-Side Evidence Alignment: https://arxiv.org/html/2608.15804v1
- SiGHT: https://proceedings.mlr.press/v300/chen26d.html and https://raw.githubusercontent.com/mlresearch/v300/main/assets/chen26d/chen26d.pdf
- Layer-resolved OT caution: https://arxiv.org/abs/2606.13216
